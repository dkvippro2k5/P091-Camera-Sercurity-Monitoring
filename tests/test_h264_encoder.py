"""Tests for the fMP4 encoder behind the grid's WebSocket video transport.

The behaviours here are the ones whose failures are silent in a browser: a fragment that
splits across two muxer flushes and gets dropped, a late viewer handed a P-frame it cannot
decode, and a stalled viewer growing the server's memory. All are reachable without a
socket or a real camera.
"""

import numpy as np
import pytest

from src.vision.h264_encoder import (
    _SUBSCRIBER_QUEUE_LIMIT,
    H264StreamEncoder,
    _moof_starts_keyframe,
    _split_fmp4,
)

pytest.importorskip("av", reason="PyAV is what provides the H.264 encoder under test")


def frames(count: int, width: int = 320, height: int = 240, seed: int = 0):
    """Deterministic frames with real structure.

    Not `np.random`: noise is the worst case for a video encoder and produces fragments an
    order of magnitude larger than any real footage, which would make the queue-limit test
    below assert against a number nothing in production resembles. A moving block compresses
    like a camera scene does.
    """
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    for i in range(count):
        frame = background.copy()
        x = (i * 7) % max(1, width - 40)
        frame[height // 3 : height // 3 + 40, x : x + 40] = 220
        yield frame


def encode_all(encoder: H264StreamEncoder, count: int, **kwargs) -> None:
    for frame in frames(count, **kwargs):
        encoder.encode(frame)


# ---------------------------------------------------------------------------
# fMP4 box parsing
# ---------------------------------------------------------------------------


def test_split_separates_init_segment_from_media_fragments():
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    queue, _, _ = encoder.subscribe()
    encode_all(encoder, 20)

    init = encoder.init_segment()
    assert init is not None
    # MSE rejects an init segment that is not exactly ftyp+moov, and both boxes have to be
    # whole — a truncated moov fails at addSourceBuffer time with no useful error.
    assert init[4:8] == b"ftyp"
    assert b"moov" in init
    assert queue, "subscriber received no fragments"
    for fragment in queue:
        assert fragment[4:8] == b"moof"
        assert b"mdat" in fragment
    encoder.close()


def test_split_holds_back_a_fragment_whose_mdat_has_not_arrived():
    """A moof written without its mdat must not be emitted or counted as consumed.

    This is the split-flush case: the muxer can return mid-fragment, and emitting the moof
    alone would hand MSE a fragment header describing samples that are not there.
    """
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    queue, _, _ = encoder.subscribe()
    encode_all(encoder, 12)
    complete = queue[0]
    init = encoder.init_segment()
    assert init is not None
    encoder.close()

    moof_len = int.from_bytes(complete[:4], "big")
    moof_only = complete[:moof_len]

    _, fragments, consumed = _split_fmp4(moof_only)
    assert fragments == []
    assert consumed == 0, "an orphan moof must be carried forward, not consumed"

    # The case that actually happens on the wire: a flush ending mid-fragment, right after
    # the init segment. The init is consumed, the orphan moof is not, and the two are not
    # allowed to be conflated — consuming the moof here loses the fragment entirely.
    prefixed = init + moof_only
    returned_init, fragments, consumed = _split_fmp4(prefixed)
    assert returned_init == init
    assert fragments == []
    assert consumed == len(init), "the orphan moof must be left for the next flush"

    # Feeding the carried-forward remainder back with the mdat appended yields the whole
    # fragment, which is the round trip `_encode_frame` performs via `self._tail`.
    _, fragments, consumed = _split_fmp4(prefixed[consumed:] + complete[moof_len:])
    assert len(fragments) == 1
    assert fragments[0][0] == complete
    assert consumed == len(complete)


def test_split_leaves_a_truncated_box_header_for_the_next_flush():
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    queue, _, _ = encoder.subscribe()
    encode_all(encoder, 12)
    fragment = queue[0]
    encoder.close()

    # Four bytes is less than a box header, so nothing can be parsed and nothing consumed.
    _, fragments, consumed = _split_fmp4(fragment[:4])
    assert fragments == []
    assert consumed == 0


def test_keyframe_detection_marks_the_gop_boundaries():
    """Exactly one fragment per GOP starts at a keyframe.

    The subscriber gate depends on this being right in both directions: over-reporting
    hands a late viewer an undecodable fragment, under-reporting leaves it waiting forever.
    """
    gop = 5
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=gop)
    queue, _, _ = encoder.subscribe()
    encode_all(encoder, 21)
    fragments = list(queue)
    encoder.close()

    flags = [_moof_starts_keyframe(f) for f in fragments]
    assert flags[0] is True, "the first fragment a subscriber sees must be a keyframe"
    keyframe_positions = [i for i, is_key in enumerate(flags) if is_key]
    assert len(keyframe_positions) >= 2, "expected more than one GOP in 21 frames"
    # Successive keyframes sit one GOP apart. Checked as a set of gaps rather than exact
    # indices because the encoder is free to emit its first fragment slightly late.
    gaps = {b - a for a, b in zip(keyframe_positions, keyframe_positions[1:])}
    assert gaps == {gop}


# ---------------------------------------------------------------------------
# Subscriber contract
# ---------------------------------------------------------------------------


def test_late_subscriber_waits_for_the_next_keyframe():
    gop = 8
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=gop)
    early_queue, _, _ = encoder.subscribe()
    encode_all(encoder, 3)  # mid-GOP: the next fragments are P-frames

    late_queue, _, late_state = encoder.subscribe()
    assert late_state["waiting_for_keyframe"] is True
    encoder.encode(next(iter(frames(1))))
    assert len(late_queue) == 0, "a mid-GOP joiner must not be given a P-frame"

    encode_all(encoder, gop + 2)
    assert late_queue, "the late subscriber never received its keyframe"
    assert _moof_starts_keyframe(late_queue[0]), "first fragment delivered was not a keyframe"
    assert len(early_queue) > len(late_queue)
    encoder.close()


def test_stalled_subscriber_is_dropped_back_to_a_keyframe_rather_than_growing():
    """A viewer that never drains must not be able to grow the server's memory.

    The recovery matters as much as the cap: after dropping a backlog the subscriber is put
    back into keyframe-wait, because handing it the newest fragment alone would reference
    pictures it just lost.
    """
    gop = 10
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=gop)
    queue, _, state = encoder.subscribe()
    encode_all(encoder, _SUBSCRIBER_QUEUE_LIMIT * 2 + gop)

    assert len(queue) <= _SUBSCRIBER_QUEUE_LIMIT
    assert state["dropped"] >= 1
    if queue:
        assert _moof_starts_keyframe(queue[0]), "resync did not start at a keyframe"
    encoder.close()


def test_unsubscribe_stops_delivery():
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    subscription = encoder.subscribe()
    queue = subscription[0]
    encode_all(encoder, 12)
    delivered = len(queue)
    assert delivered > 0

    encoder.unsubscribe(subscription)
    encode_all(encoder, 12)
    assert len(queue) == delivered
    encoder.close()


def test_close_wakes_subscribers_so_their_handlers_can_exit():
    """Without this a WebSocket handler blocks on an event no producer will ever set."""
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    _, ready, state = encoder.subscribe()
    encode_all(encoder, 8)
    ready.clear()

    encoder.close()
    assert ready.is_set()
    assert state["closed"] is True
    assert encoder.available is False


def test_subscribing_after_close_returns_an_already_closed_entry():
    encoder = H264StreamEncoder("cam", fps=20.0)
    encoder.close()
    queue, ready, state = encoder.subscribe()
    assert state["closed"] is True
    assert ready.is_set()
    assert len(queue) == 0


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_resolution_change_restarts_the_encoder_and_mints_a_new_init_segment():
    """A new init segment invalidates what viewers are decoding against, so they are dropped
    and must reconnect — a stale SourceBuffer would render the new size as garbage."""
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    _, ready, state = encoder.subscribe()
    encode_all(encoder, 10, width=320, height=240)
    first_init = encoder.init_segment()
    assert first_init is not None

    encode_all(encoder, 10, width=640, height=480)
    second_init = encoder.init_segment()

    assert second_init is not None
    assert second_init != first_init
    assert state["closed"] is True, "viewers on the old init segment must be dropped"
    assert ready.is_set()
    assert encoder.stats()["width"] == 640
    encoder.close()


def test_odd_dimensions_are_rounded_to_an_encodable_size():
    """H.264 in 4:2:0 needs even dimensions, and the display resize can land on an odd one."""
    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=5)
    queue, _, _ = encoder.subscribe()
    encode_all(encoder, 10, width=321, height=241)

    assert encoder.available, "an odd frame size must not disable the encoder"
    assert encoder.stats()["width"] % 2 == 0
    assert encoder.stats()["height"] % 2 == 0
    assert queue
    encoder.close()


def test_encoder_failure_is_contained_rather_than_raised(monkeypatch):
    """The display loop's JPEG path — which is also the MJPEG fallback — must survive an
    encoder that cannot open, so encode() reports failure through `available`, not an
    exception."""
    import src.vision.h264_encoder as module

    monkeypatch.setattr(module, "_ENCODER_PREFERENCE", ("definitely_not_a_codec",))
    encoder = H264StreamEncoder("cam", fps=20.0)

    encode_all(encoder, 3)  # must not raise

    assert encoder.available is False
    assert encoder.init_segment() is None


def test_empty_and_none_frames_are_ignored():
    encoder = H264StreamEncoder("cam", fps=20.0)
    encoder.encode(None)
    encoder.encode(np.empty((0, 0, 3), dtype=np.uint8))
    assert encoder.available, "a skipped frame must not disable the encoder"
    encoder.close()


def test_h264_is_substantially_smaller_than_the_mjpeg_it_replaces():
    """The whole point of the transport change, asserted loosely.

    The threshold is far below what real footage achieves (~25x on the project's clips) so
    this stays a regression test for "inter-frame compression is on" rather than a
    benchmark that fails when a codec default shifts.
    """
    cv2 = pytest.importorskip("cv2")

    encoder = H264StreamEncoder("cam", fps=20.0, keyframe_interval_frames=40)
    queue, _, _ = encoder.subscribe()
    jpeg_bytes = 0
    for frame in frames(60, width=640, height=480):
        encoder.encode(frame)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        assert ok
        jpeg_bytes += len(buffer)

    h264_bytes = sum(len(f) for f in queue)
    encoder.close()

    assert h264_bytes > 0
    assert jpeg_bytes > h264_bytes * 3
