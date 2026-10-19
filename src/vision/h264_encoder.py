"""Per-camera H.264 encoder producing fragmented MP4 for the browser's MediaSource API.

Why this exists: the live grid used to ship MJPEG — one standalone JPEG per frame over
`multipart/x-mixed-replace`. That works, but every frame pays full intra-frame cost on the
wire and a separate CPU decode in the browser, so a server producing 20fps landed at 10-14fps
rendered. H.264 sends keyframes rarely and deltas in between (~8x less data at 480p here),
and the browser decodes it on the GPU, so the same 20fps arrives as 20fps.

The output is fMP4 in two parts, which is exactly the shape MediaSource Extensions wants:

  - **init segment** (`ftyp` + `moov`) — codec/resolution header. Every viewer needs this
    once, before any media, so it is captured separately and replayed to late joiners.
  - **media fragments** (`moof` + `mdat`) — one per frame (`frag_every_frame`), which is what
    keeps latency at roughly one frame instead of one fragment-duration.

A viewer that connects mid-stream gets the init segment, then must wait for the next
keyframe: an fMP4 fragment holding a P-frame references pictures the newcomer never saw, and
feeding it to MSE yields a decode error or a smeared image. `subscribe` handles that by
holding new subscribers until a keyframe arrives; `keyframe_interval_frames` bounds how long
that wait can be.

Encoding runs on the caller's thread (the worker's display loop), not a thread of its own:
the display loop already paces itself at `display_fps` and already holds the frame, so an
extra thread would only add a hand-off and a second copy of the frame.
"""

import logging
import threading
from collections import deque
from fractions import Fraction

import numpy as np

logger = logging.getLogger(__name__)

# fMP4 muxes in this timescale. Milliseconds are fine for video at these rates and keep the
# pts arithmetic in plain integers.
_TIME_BASE = Fraction(1, 1000)

# How many fragments a subscriber may fall behind before the oldest are dropped. A viewer on
# a slow link must not be able to grow the server's memory without bound; dropping video for
# a struggling client is the right trade in a live view, where the newest frame is the one
# that matters. At ~8KB per 480p fragment this caps a stalled subscriber at ~500KB.
_SUBSCRIBER_QUEUE_LIMIT = 60

# Codecs tried in order. NVENC keeps the encode off the CPU where the deployment has a
# usable NVIDIA GPU (the T4 does); libx264 is the portable fallback and is what CI and any
# CPU-only dev machine will actually use.
_ENCODER_PREFERENCE = ("h264_nvenc", "libx264")


class _Sink:
    """File-like target handed to PyAV; collects the muxer's writes in memory.

    PyAV needs something with `write`; it never reads back, so this is deliberately not a
    real `io` object. Writes arrive on the encoding thread only.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()

    def write(self, data) -> int:
        self.buffer.extend(data)
        return len(data)

    def take(self) -> bytes:
        out = bytes(self.buffer)
        self.buffer.clear()
        return out


def _moof_starts_keyframe(moof: bytes) -> bool:
    """Whether this fragment's first sample is a sync sample (an IDR frame).

    Finds the `trun` box inside `moof/traf` and reads the flags that apply to sample 0:
    `first_sample_flags` when present (0x000004), otherwise the first entry of the per-sample
    flags (0x000400). Bit 16 of a sample-flags word is `sample_is_non_sync_sample`, so a
    clear bit means a keyframe. An unrecognised layout conservatively reports "not a
    keyframe" and the subscriber simply waits for a fragment that can be proven safe.

    Reading the muxer's own flags beats re-parsing H.264 NAL units out of the `mdat`: it is
    a fixed offset walk rather than a scan, and it cannot be fooled by emulation-prevention
    bytes inside the payload.
    """
    idx = moof.find(b"trun")
    if idx < 0 or idx + 12 > len(moof):
        return False
    flags = int.from_bytes(moof[idx + 5 : idx + 8], "big")
    sample_count = int.from_bytes(moof[idx + 8 : idx + 12], "big")
    if sample_count == 0:
        return False

    cursor = idx + 12
    if flags & 0x000001:  # data_offset present
        cursor += 4
    if flags & 0x000004:  # first_sample_flags present — describes sample 0 exactly
        if cursor + 4 > len(moof):
            return False
        return not (int.from_bytes(moof[cursor : cursor + 4], "big") & 0x00010000)

    if flags & 0x000400:  # per-sample flags; sample 0's entry comes first
        if flags & 0x000100:  # sample_duration precedes it
            cursor += 4
        if flags & 0x000200:  # sample_size precedes it
            cursor += 4
        if cursor + 4 > len(moof):
            return False
        return not (int.from_bytes(moof[cursor : cursor + 4], "big") & 0x00010000)

    return False


def _split_fmp4(data: bytes) -> tuple[bytes, list[tuple[bytes, bool]], int]:
    """Split raw muxer output into (init segment, [(fragment, is_keyframe), ...], consumed).

    Walks the top-level box headers rather than scanning for magic bytes: `ftyp`/`moov`
    belong to the init segment, and each following `moof` opens a fragment that its `mdat`
    closes. `consumed` is how many bytes were parsed into complete boxes — the caller keeps
    the remainder, which is a box the muxer has only half-written, and prepends it next time
    so no bytes are invented or lost.

    A `moof` with no `mdat` yet is not counted as consumed, so a fragment split across two
    muxer flushes is emitted whole on the second.
    """
    init = bytearray()
    fragments: list[tuple[bytes, bool]] = []
    pos = 0
    consumed = 0
    pending_moof: bytes | None = None
    pending_is_key = False
    pending_start = 0

    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        box_type = data[pos + 4 : pos + 8]
        # A size of 0 means "to end of file" and 1 means a 64-bit size follows. Neither is
        # emitted by this muxer configuration; treating them as a truncated tail is safer
        # than guessing, because the alternative is an infinite loop on size 0.
        if size < 8 or pos + size > len(data):
            break

        box = bytes(data[pos : pos + size])
        if box_type in (b"ftyp", b"moov"):
            init.extend(box)
            consumed = pos + size
        elif box_type == b"moof":
            pending_moof = box
            pending_is_key = _moof_starts_keyframe(box)
            pending_start = pos
        elif box_type == b"mdat" and pending_moof is not None:
            fragments.append((pending_moof + box, pending_is_key))
            pending_moof = None
            consumed = pos + size
        else:
            # Anything else (`mfra`, `free`, ...) is muxer bookkeeping MSE does not want,
            # but it is fully parsed, so it still advances the consumed mark.
            if pending_moof is None:
                consumed = pos + size
        pos += size

    # An orphan `moof` means its `mdat` has not been written yet, so the pair must be carried
    # into the next flush together. With this muxer the `mdat` always follows immediately, so
    # nothing after the `moof` has advanced the mark and the rewind is a no-op — it is kept
    # because dropping a whole fragment is the failure it prevents, and a muxer that ever
    # interleaves a box between the two would trigger it silently.
    if pending_moof is not None:
        consumed = min(consumed, pending_start)

    return bytes(init), fragments, consumed


class H264StreamEncoder:
    """Encodes one camera's display frames to fMP4 and fans them out to WebSocket viewers.

    Thread model: `encode()` is called from a single producer (the worker's display loop).
    Subscribers are consumed by any number of WebSocket handlers. The lock guards only the
    subscriber registry and the cached init segment — encoding itself is never done under
    it, so a slow subscriber cannot stall the camera.
    """

    def __init__(
        self,
        camera_id: str,
        fps: float = 20.0,
        bitrate: int = 1_200_000,
        keyframe_interval_frames: int = 40,
    ) -> None:
        self.camera_id = camera_id
        self.fps = fps if fps > 0 else 20.0
        self.bitrate = bitrate
        self.keyframe_interval_frames = keyframe_interval_frames

        self._lock = threading.Lock()
        self._subscribers: list[tuple[deque[bytes], threading.Event, dict]] = []
        self._init_segment: bytes | None = None
        self._container = None
        self._stream = None
        self._sink = _Sink()
        self._tail = b""
        self._frame_index = 0
        self._width = 0
        self._height = 0
        self._failed = False
        self._closed = False
        self._codec_name: str | None = None

    # ----------------------------------------------------------------- producer
    def encode(self, frame: np.ndarray) -> None:
        """Encode one BGR frame and hand the resulting fragments to every subscriber.

        Never raises: the display loop's JPEG path must keep working even if H.264 does not,
        because that path is also the MJPEG fallback the frontend drops back to. A failure
        disables the encoder for this camera's session rather than being retried per frame,
        which would otherwise log once per frame forever.
        """
        if self._failed or self._closed or frame is None or frame.size == 0:
            return

        height, width = frame.shape[:2]
        try:
            if self._container is None:
                self._open(width, height)
            elif (width - width % 2, height - height % 2) != (self._width, self._height):
                # A source that changes resolution mid-stream invalidates the init segment
                # every viewer is already decoding against. Restarting the encoder mints a
                # new one; subscribers are dropped so their handlers resend it.
                logger.info(
                    f"[{self.camera_id}] Display resolution changed "
                    f"{self._width}x{self._height} -> {width}x{height}; restarting encoder."
                )
                self._close_container()
                self._drop_subscribers()
                self._open(width, height)

            self._encode_frame(frame)
        except Exception as exc:
            self._failed = True
            self._close_container()
            self._drop_subscribers()
            logger.warning(
                f"[{self.camera_id}] H.264 encoding disabled for this session ({exc}). Viewers fall back to MJPEG."
            )

    def _open(self, width: int, height: int) -> None:
        import av  # Imported lazily so a machine without PyAV can still run MJPEG-only.

        # H.264 requires even dimensions in 4:2:0. The display loop's resize can land on an
        # odd height for an unusual aspect ratio, so round down rather than fail to open.
        width -= width % 2
        height -= height % 2
        if width <= 0 or height <= 0:
            raise ValueError(f"frame too small to encode: {width}x{height}")

        last_error: Exception | None = None
        for codec_name in _ENCODER_PREFERENCE:
            sink = _Sink()
            container = None
            try:
                container = av.open(
                    sink,
                    mode="w",
                    format="mp4",
                    options={
                        # empty_moov + frag_every_frame is what makes this streamable: the
                        # header carries no sample table, and each frame closes its own
                        # fragment so nothing waits on a fragment-duration timer.
                        "movflags": "frag_every_frame+empty_moov+default_base_moof",
                    },
                )
                stream = container.add_stream(codec_name, rate=int(round(self.fps)))
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.bit_rate = self.bitrate
                stream.codec_context.time_base = _TIME_BASE
                stream.options = self._codec_options(codec_name)
                # `add_stream` only resolves the codec by name — it does not open it, so a
                # codec that is compiled in but unusable (NVENC with no NVIDIA device, or a
                # driver that refuses the session) fails later, on the first encode, long
                # after this loop could have fallen back. Opening it here moves that failure
                # into the loop where the fallback lives.
                stream.codec_context.open()
            except Exception as exc:
                last_error = exc
                logger.debug(f"[{self.camera_id}] Encoder '{codec_name}' unavailable: {exc}")
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
                continue

            self._container = container
            self._stream = stream
            self._sink = sink
            self._codec_name = codec_name
            self._width, self._height = width, height
            self._frame_index = 0
            self._tail = b""
            with self._lock:
                self._init_segment = None
            logger.info(
                f"[{self.camera_id}] H.264 stream opened with '{codec_name}' at {width}x{height} / {self.fps:.0f}fps."
            )
            return

        raise RuntimeError(f"no usable H.264 encoder: {last_error}")

    def _codec_options(self, codec_name: str) -> dict[str, str]:
        """Encoder flags chosen for live viewing rather than file size.

        Common to both: no B-frames and a fixed keyframe interval. B-frames would force the
        muxer to hold a frame back to reorder it, adding latency for no benefit at these
        bitrates, and a bounded GOP is what caps how long a joining viewer waits.
        """
        gop = str(self.keyframe_interval_frames)
        if codec_name == "h264_nvenc":
            return {
                "preset": "p1",  # fastest NVENC preset
                "tune": "ull",  # ultra-low latency
                "profile": "baseline",
                "g": gop,
                "bf": "0",
                "delay": "0",
                "zerolatency": "1",
            }
        return {
            "preset": "veryfast",
            "tune": "zerolatency",
            "profile": "baseline",
            "g": gop,
            "bf": "0",
        }

    def _encode_frame(self, frame: np.ndarray) -> None:
        import av

        if (frame.shape[1], frame.shape[0]) != (self._width, self._height):
            # Only reachable after the even-dimension rounding above trimmed a row/column.
            frame = frame[: self._height, : self._width]

        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        # Pace pts by the nominal display rate rather than the wall clock: the display loop
        # already runs on a fixed interval, and a wall-clock pts makes every dropped tick
        # show up as a timestamp gap that MSE renders as a stall.
        video_frame.pts = int(self._frame_index * 1000 / self.fps)
        video_frame.time_base = _TIME_BASE
        self._frame_index += 1

        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

        raw = self._tail + self._sink.take()
        init, fragments, consumed = _split_fmp4(raw)
        self._tail = raw[consumed:]

        if init:
            with self._lock:
                self._init_segment = init
        for fragment, is_keyframe in fragments:
            self._publish(fragment, is_keyframe)

    def _publish(self, fragment: bytes, is_keyframe: bool) -> None:
        with self._lock:
            for queue, ready, state in self._subscribers:
                if state["waiting_for_keyframe"]:
                    if not is_keyframe:
                        continue
                    state["waiting_for_keyframe"] = False
                if len(queue) >= _SUBSCRIBER_QUEUE_LIMIT:
                    # This subscriber is not draining. Drop its backlog entirely and make it
                    # resync on the next keyframe: handing it the newest fragment alone would
                    # reference frames it just lost.
                    queue.clear()
                    state["waiting_for_keyframe"] = True
                    state["dropped"] += 1
                    continue
                queue.append(fragment)
                ready.set()

    # ---------------------------------------------------------------- consumers
    def subscribe(self) -> tuple[deque[bytes], threading.Event, dict]:
        """Register a viewer. Returns its queue, its wake event, and its state dict.

        The subscriber starts held until the next keyframe — see the module docstring. The
        caller must pass the same triple back to `unsubscribe` when it disconnects.
        """
        entry: tuple[deque[bytes], threading.Event, dict] = (
            deque(),
            threading.Event(),
            {"waiting_for_keyframe": True, "dropped": 0, "closed": False},
        )
        with self._lock:
            if self._failed or self._closed:
                entry[2]["closed"] = True
                entry[1].set()
                return entry
            self._subscribers.append(entry)
        return entry

    def unsubscribe(self, entry) -> None:
        with self._lock:
            if entry in self._subscribers:
                self._subscribers.remove(entry)

    def init_segment(self) -> bytes | None:
        """The `ftyp`+`moov` header, or None until the first frame has been encoded."""
        with self._lock:
            return self._init_segment

    @property
    def available(self) -> bool:
        """False once encoding has failed or stopped for this session — the API uses this to
        tell a client to fall back to MJPEG rather than leave it on a silent socket."""
        return not (self._failed or self._closed)

    @property
    def codec_name(self) -> str | None:
        return self._codec_name

    def stats(self) -> dict:
        with self._lock:
            return {
                "available": not (self._failed or self._closed),
                "codec": self._codec_name,
                "width": self._width,
                "height": self._height,
                "subscribers": len(self._subscribers),
                "encoded_frames": self._frame_index,
                "dropped_backlogs": sum(s["dropped"] for _, _, s in self._subscribers),
            }

    # ----------------------------------------------------------------- teardown
    def _drop_subscribers(self) -> None:
        """Detach every viewer and wake it, so each handler notices and resyncs or exits."""
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for _, ready, state in subscribers:
            state["closed"] = True
            ready.set()

    def _close_container(self) -> None:
        container, stream = self._container, self._stream
        self._container = self._stream = None
        if container is None:
            return
        try:
            if stream is not None:
                for packet in stream.encode(None):
                    container.mux(packet)
            container.close()
        except Exception:
            # A close failure on a container that is being discarded anyway has nothing
            # left to affect; the buffers go with it.
            logger.debug(f"[{self.camera_id}] Ignoring error while closing H.264 container.")

    def close(self) -> None:
        """Stop encoding and wake every subscriber so their loops can exit."""
        self._closed = True
        self._close_container()
        self._drop_subscribers()
