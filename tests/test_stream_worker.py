"""Regression tests for the RTSP ingest path: reader loop, reconnect policy, and the
frame-buffer contract between the worker and the pipeline.

These use a fake VideoCapture rather than a real socket. The behaviours that broke in
production — a wedged camera that opens but never delivers, a file looping forever, a
worker mutating the buffer it was handed — are all reachable without one, and a test
that needs a live RTSP server does not get run.
"""

import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pytest

from src.vision import IncidentDetector, VisionPipeline
from src.vision.stream_worker import CameraStreamWorker


class FakeCapture:
    """Stands in for cv2.VideoCapture with a scripted grab() outcome.

    Args:
        grabs: How many successful grabs before grab() starts failing. None = endless.
        frame_count: What CAP_PROP_FRAME_COUNT reports. >0 marks a finite source (a
            file); live RTSP reports 0 or -1.
        opened: Whether the capture opens at all.
        fps: What CAP_PROP_FPS reports, used to pace finite sources.
    """

    def __init__(
        self,
        grabs: int | None = None,
        frame_count: float = 0.0,
        opened: bool = True,
        fps: float = 0.0,
    ):
        self._remaining = grabs
        self._frame_count = frame_count
        self._opened = opened
        self._fps = fps
        self.initial_grabs = grabs
        self.released = False
        self.props: dict[int, float] = {}
        self.resets = 0

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        if prop_id == cv2.CAP_PROP_POS_FRAMES and value == 0:
            self.resets += 1
            self._remaining = self.initial_grabs
        return True

    def isOpened(self) -> bool:  # noqa: N802 - matches the cv2 API
        return self._opened and not self.released

    def grab(self) -> bool:
        if self._remaining is None:
            return True
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True

    def retrieve(self):
        return True, np.full((48, 64, 3), 30, dtype=np.uint8)

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def set(self, prop: int, value: float) -> bool:  # noqa: A003
        self.props[prop] = value
        return True

    def release(self) -> None:
        self.released = True


@pytest.fixture
def mock_pipeline():
    """Pipeline with a scripted detector and no overlays: fast and deterministic."""
    return VisionPipeline(detector=IncidentDetector(model_path=None), draw_overlays=False)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll until predicate holds. Returns False on timeout rather than hanging."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Reconnect policy
# ---------------------------------------------------------------------------


def test_breaker_trips_when_stream_opens_but_never_delivers(mock_pipeline, monkeypatch):
    """A camera that accepts the connection then goes silent must eventually be given up on.

    The original loop only counted *open* failures and reset that counter on every
    successful open, so this wedged-encoder case retried forever.
    """
    opens = []

    def fake_open(self):
        cap = FakeCapture(grabs=0)  # opens fine, delivers nothing
        opens.append(cap)
        return cap

    monkeypatch.setattr(CameraStreamWorker, "_open_capture", fake_open)

    worker = CameraStreamWorker(
        "cam_wedged", "rtsp://fake/stream", pipeline=mock_pipeline,
        reconnect_interval_sec=0.01, max_reconnect_attempts=3,
    )
    worker.start()

    assert wait_until(lambda: worker._stop_event.is_set()), "breaker never tripped"
    worker.stop()

    assert len(opens) == 3, f"expected to stop after 3 attempts, made {len(opens)}"
    assert worker.get_stats()["stopped_reason"] is not None
    assert all(c.released for c in opens), "captures leaked on the failure path"


def test_finite_source_ends_instead_of_restarting(mock_pipeline, monkeypatch):
    """A video file reaching EOF is done, not disconnected.

    Treating EOF as a drop restarted the file from frame 0 and never terminated.
    """
    opens = []

    def fake_open(self):
        cap = FakeCapture(grabs=3, frame_count=3)  # a file: 3 frames then EOF
        opens.append(cap)
        return cap

    monkeypatch.setattr(CameraStreamWorker, "_open_capture", fake_open)

    worker = CameraStreamWorker(
        "cam_file", "video.mp4", pipeline=mock_pipeline, reconnect_interval_sec=0.01,
    )
    worker.start()

    assert wait_until(lambda: worker._stop_event.is_set()), "worker looped past EOF"
    worker.stop()

    assert len(opens) == 1, "source was reopened after a legitimate EOF"
    assert worker.get_stats()["stopped_reason"] == "end_of_source"


def test_finite_source_loops_when_explicitly_requested(mock_pipeline, monkeypatch):
    """The looping demo behaviour stays available behind an opt-in flag."""
    cap = FakeCapture(grabs=2, frame_count=2)

    def fake_open(self):
        return cap

    monkeypatch.setattr(CameraStreamWorker, "_open_capture", fake_open)

    worker = CameraStreamWorker(
        "cam_loop", "video.mp4", pipeline=mock_pipeline, reconnect_interval_sec=0.01,
        loop_finite_sources=True,
    )
    worker.start()

    assert wait_until(lambda: cap.resets >= 1), "loop_finite_sources did not restart the file"
    worker.stop()


def test_healthy_frames_rearm_the_breaker(mock_pipeline, monkeypatch):
    """A brief blip must not accumulate toward the breaker across a healthy period.

    The counter resets on a delivered frame, not on a successful open — an open that
    yields nothing should not be credited as recovery.
    """
    states = [FakeCapture(grabs=5), FakeCapture(grabs=5), FakeCapture(grabs=None)]
    opens = []

    def fake_open(self):
        cap = states[min(len(opens), len(states) - 1)]
        opens.append(cap)
        return cap

    monkeypatch.setattr(CameraStreamWorker, "_open_capture", fake_open)

    worker = CameraStreamWorker(
        "cam_blip", "rtsp://fake/stream", pipeline=mock_pipeline,
        reconnect_interval_sec=0.01, max_reconnect_attempts=2,
    )
    worker.start()

    # Two separate blips, each followed by real frames, then a healthy stream.
    assert wait_until(lambda: len(opens) >= 3, timeout=5.0)
    time.sleep(0.2)

    assert not worker._stop_event.is_set(), "breaker tripped despite frames arriving between blips"
    worker.stop()


# ---------------------------------------------------------------------------
# Read pacing
# ---------------------------------------------------------------------------


def test_file_source_is_paced_to_its_own_frame_rate(mock_pipeline, monkeypatch):
    """A file must be read at its native fps, not as fast as the disk allows.

    grab() on a file never blocks, so an unpaced reader drained a 75s clip in ~2s while
    the processor — throttled to target_fps — saw a single frame. Everything else was
    dropped by the single-slot buffer, so a demo looked like it had analyzed nothing.
    """
    def fake_open(self):
        return FakeCapture(grabs=None, frame_count=900, fps=30.0)

    monkeypatch.setattr(CameraStreamWorker, "_open_capture", fake_open)

    worker = CameraStreamWorker("cam_paced", "clip.mp4", pipeline=mock_pipeline)
    worker.start()
    time.sleep(1.0)
    read = worker.get_stats()["total_read_frames"]
    worker.stop()

    # At 30 fps, ~30 frames in a second. Unpaced this ran into the thousands.
    assert 5 <= read <= 90, f"read {read} frames in ~1s; expected pacing near 30"


def test_live_stream_is_not_paced(monkeypatch):
    """A live source is paced by the network; adding our own delay would drop frames."""
    worker = CameraStreamWorker(
        "cam_live", "rtsp://host/s",
        pipeline=VisionPipeline(detector=IncidentDetector(model_path=None)),
    )

    # frame_count 0 marks a live stream, so no interval regardless of reported fps.
    assert worker._source_frame_interval(FakeCapture(frame_count=0, fps=30.0)) == 0.0
    # A file with a broken fps report must not divide by zero either.
    assert worker._source_frame_interval(FakeCapture(frame_count=900, fps=0.0)) == 0.0
    assert worker._source_frame_interval(FakeCapture(frame_count=900, fps=30.0)) == pytest.approx(1 / 30)


# ---------------------------------------------------------------------------
# Transport configuration
# ---------------------------------------------------------------------------


def test_rtsp_gets_tcp_transport_and_timeouts(monkeypatch):
    """RTSP must use TCP and bounded timeouts; UDP corrupts frames and no timeout hangs stop()."""
    captured = {}

    def fake_videocapture(source, backend=None, params=None):
        captured["source"] = source
        captured["params"] = params
        captured["ffmpeg_opts"] = __import__("os").environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        return FakeCapture()

    monkeypatch.setattr(cv2, "VideoCapture", fake_videocapture)

    worker = CameraStreamWorker(
        "cam", "rtsp://host/stream", pipeline=VisionPipeline(detector=IncidentDetector(model_path=None)),
        open_timeout_sec=4.0, read_timeout_sec=6.0,
    )
    worker._open_capture()

    assert "rtsp_transport;tcp" in (captured["ffmpeg_opts"] or ""), "RTSP still defaults to UDP"

    params = captured["params"] or []
    assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params, "open timeout not passed at construction"
    assert 4000 in params and 6000 in params, f"timeout values not applied: {params}"


def test_local_file_skips_rtsp_only_options(monkeypatch):
    """A file source must not be handed RTSP transport params."""
    captured = {}

    def fake_videocapture(source, backend=None, params=None):
        captured["params"] = params
        return FakeCapture()

    monkeypatch.setattr(cv2, "VideoCapture", fake_videocapture)

    worker = CameraStreamWorker(
        "cam", "video.mp4", pipeline=VisionPipeline(detector=IncidentDetector(model_path=None)),
    )
    worker._open_capture()

    assert captured["params"] is None


# ---------------------------------------------------------------------------
# Frame buffer contract
# ---------------------------------------------------------------------------


def test_pipeline_does_not_mutate_the_shared_reader_buffer():
    """The worker hands the pipeline its live single-slot buffer without copying.

    With overlays on, the draw_* helpers composite in place, so the reader's buffer was
    being progressively painted over — the unmodified frame no longer existed anywhere.
    """
    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None), draw_overlays=True)
    shared_buffer = np.full((480, 640, 3), 30, dtype=np.uint8)
    pristine = shared_buffer.copy()

    out, _ = pipeline.process_frame(shared_buffer, frame_id=1, camera_id="cam_01")

    assert np.array_equal(shared_buffer, pristine), "process_frame mutated the caller's frame"
    assert out is not shared_buffer, "overlay output aliases the input buffer"


def test_repeated_processing_of_one_frame_leaves_no_residue():
    """Processing the same buffer twice must give identical output.

    The processor re-reads the single slot each tick. When overlays were blended in
    place, pass 2 drew on top of pass 1's output and the zone fills darkened cumulatively.
    A fresh pipeline per pass isolates the buffer from rule-engine state, so any
    difference here is residue left on the frame.
    """
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    now = datetime(2026, 1, 1, 12, 0, 0)

    def run() -> np.ndarray:
        pipeline = VisionPipeline(detector=IncidentDetector(model_path=None), draw_overlays=True)
        out, _ = pipeline.process_frame(
            frame, frame_id=1, camera_id="cam_01", timestamp=1.0, now=now
        )
        return out

    first = run()
    second = run()

    assert np.array_equal(first, second), "overlays accumulated on the shared buffer"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_stop_joins_both_threads(mock_pipeline, monkeypatch):
    """stop() must leave no live worker threads behind."""
    monkeypatch.setattr(CameraStreamWorker, "_open_capture", lambda self: FakeCapture())

    worker = CameraStreamWorker("cam_lifecycle", "rtsp://fake/stream", pipeline=mock_pipeline)
    worker.start()
    assert wait_until(lambda: worker.get_stats()["total_read_frames"] > 0)

    worker.stop()

    assert not worker.is_running()
    leaked = [t.name for t in threading.enumerate() if "cam_lifecycle" in t.name]
    assert leaked == [], f"threads leaked after stop(): {leaked}"


def test_callback_receives_processed_frames(mock_pipeline, monkeypatch):
    """The result callback fires with the camera id and a frame of the right shape."""
    monkeypatch.setattr(CameraStreamWorker, "_open_capture", lambda self: FakeCapture())
    seen = []

    worker = CameraStreamWorker(
        "cam_cb", "rtsp://fake/stream", pipeline=mock_pipeline, target_fps=30.0,
        on_result_callback=lambda cid, res, frame: seen.append((cid, res, frame)),
    )
    worker.start()
    got = wait_until(lambda: len(seen) > 0)
    worker.stop()

    assert got, "callback never fired"
    camera_id, result, frame = seen[0]
    assert camera_id == "cam_cb"
    assert result.camera_id == "cam_cb"
    assert frame.shape == (48, 64, 3)
