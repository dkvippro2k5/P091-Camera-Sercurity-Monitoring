import glob
import os
import threading
import time

import cv2
import numpy as np
import pytest

from src.vision import (
    BoundingBox,
    FrameAnalysisResult,
    IncidentDetection,
    IncidentDetector,
    MultiStreamManager,
    RTSPSimulator,
    VisionPipeline,
)


def test_vision_schemas():
    bbox = BoundingBox(xmin=10, ymin=20, xmax=100, ymax=200)
    assert bbox.xmin == 10
    assert bbox.ymax == 200

    detection = IncidentDetection(
        label="intrusion",
        track_id="person_105",
        confidence=0.95,
        timestamp=1715012345.67,
        camera_id="cam_hallway_01",
        bbox=bbox,
    )
    assert detection.label == "intrusion"
    assert detection.track_id == "person_105"
    assert detection.confidence == 0.95
    assert detection.camera_id == "cam_hallway_01"

    frame_result = FrameAnalysisResult(
        frame_id=1, camera_id="cam_hallway_01", detections=[detection], processing_time_ms=12.5
    )
    assert frame_result.frame_id == 1
    assert len(frame_result.detections) == 1


def test_detector_with_no_model_returns_nothing():
    """model_path=None is the null detector: it must report nothing, never invent it.

    Tests that only exercise the rule layer construct the detector this way. Returning
    fabricated detections here would make a surveillance system report incidents that
    have nothing to do with the camera feed.
    """
    detector = IncidentDetector(model_path=None)
    assert detector.model is None

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert detector.detect(dummy_frame) == []


def test_vision_pipeline_frame_process():
    pipeline = VisionPipeline()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    output_frame, result = pipeline.process_frame(dummy_frame, frame_id=42)

    assert output_frame.shape == dummy_frame.shape
    assert result.frame_id == 42
    assert result.processing_time_ms >= 0.0


def test_the_pipeline_never_modifies_the_caller_s_frame():
    """The live path hands in the reader thread's shared single-slot buffer.

    Overlays composite in place, so the pipeline copies before drawing. Without that copy
    the annotations land on the caller's array and the unannotated original stops existing
    anywhere — every later consumer of that buffer sees boxes burned into it.
    """
    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    before = frame.copy()

    pipeline.process_frame(frame, frame_id=1)

    assert np.array_equal(frame, before), "process_frame drew on the caller's frame"


def test_camera_id_propagates_into_detections():
    """camera_id must reach every detection, not just the frame result.

    Run against real footage rather than a blank frame: the detector stamps camera_id in
    the YOLO path, so a frame with nothing in it exercises none of that plumbing. `data/`
    is gitignored, so this skips in CI rather than failing on absent clips.
    """
    clips = sorted(glob.glob(os.path.join("data", "vision", "test", "*.mp4")))
    if not clips:
        pytest.skip("no test footage under data/vision/test/")

    cap = cv2.VideoCapture(clips[0])
    try:
        pipeline = VisionPipeline()
        result = None
        # People are not necessarily in the very first frame; give it a short budget
        # rather than a single shot, and keep it small so this stays fast.
        for frame_id in range(15):
            ok, frame = cap.read()
            if not ok:
                break
            _, result = pipeline.process_frame(frame, frame_id=frame_id, camera_id="cam_lobby_07")
            if result.detections:
                break
    finally:
        cap.release()

    assert result is not None, f"could not read a single frame from '{clips[0]}'"
    assert result.camera_id == "cam_lobby_07"
    assert result.timestamp is not None
    assert result.detections, f"no detections in the first 15 frames of '{clips[0]}'"
    assert all(d.camera_id == "cam_lobby_07" for d in result.detections)


def test_last_tracks_normalizes_person_boxes_to_the_frame_size():
    """`last_tracks` is what MonitoringService crops the VLM clip to a track with (see
    docs/plans/vlm-throughput-and-concurrent-events.md stage 3) -- it must report boxes as
    fractions of the frame, not pixels, since the ring reads them back at whatever
    resolution that particular frame happens to be at.
    """
    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None))
    # Bypass detection entirely -- this pins the normalization math in last_tracks, not
    # YOLO's output, so it needs no model weights and no real footage.
    pipeline._last_detections = [
        IncidentDetection(
            label="person",
            track_id="person_1",
            confidence=0.9,
            bbox=BoundingBox(xmin=100, ymin=50, xmax=300, ymax=450),
        ),
        IncidentDetection(
            label="backpack",
            track_id="bag_1",
            confidence=0.8,
            bbox=BoundingBox(xmin=10, ymin=10, xmax=20, ymax=20),
        ),
    ]
    pipeline._last_detections_frame_size = (400, 500)  # (width, height)

    tracks = pipeline.last_tracks()

    assert set(tracks) == {"person_1"}, "only person tracks are returned"
    assert tracks["person_1"] == pytest.approx((0.25, 0.1, 0.75, 0.9))


def test_last_tracks_is_empty_before_the_first_inference():
    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None))
    assert pipeline.last_tracks() == {}


def test_torch_thread_tuning_is_idempotent():
    """Tuning is process-global, so it must apply once and then no-op.

    Repeated calls would otherwise reset the pool mid-run, and torch raises if a parallel
    region has already started.
    """
    from src.vision import stream_manager

    # Reset the module guard so this test sees a first call regardless of ordering.
    stream_manager._torch_tuned = False

    first = stream_manager.tune_torch_for_multistream(threads_per_worker=2)
    second = stream_manager.tune_torch_for_multistream(threads_per_worker=2)

    assert second is False, "tuning applied twice"

    try:
        import torch

        if first and not torch.cuda.is_available():
            assert torch.get_num_threads() == 2
    except ImportError:
        pass


def test_manager_warns_when_cameras_exceed_cpu_capacity(caplog, monkeypatch):
    """Oversubscription must be surfaced: frames go stale silently otherwise."""
    import logging as _logging

    # The warning is deliberately CPU-only — _warn_if_oversubscribed returns early on a
    # GPU host because that capacity curve is different and the CPU budget would be a
    # wrong guess. This test is about the CPU path, so the host's actual GPU is pinned
    # out; without this it passes in CI (no GPU) and fails on any dev box that has one.
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ImportError:
        pass

    manager = MultiStreamManager(tune_torch_threads=False)
    budget = max(2, (os.cpu_count() or 4) // 2)

    with caplog.at_level(_logging.WARNING, logger="src.vision.stream_manager"):
        for i in range(budget + 2):
            manager.add_camera(f"cam_{i:02d}", f"rtsp://host/{i}", autostart=False)

    warnings = [r for r in caplog.records if "exceeds" in r.message]
    assert warnings, f"no oversubscription warning after {budget + 2} cameras"
    # Once, not once per camera.
    assert len(warnings) == 1

    manager.stop_all()


def test_process_video_rejects_undecodable_file(tmp_path):
    """A file that exists but cannot be decoded must raise, not report a successful run.

    os.path.exists only proves the path is there. A truncated or non-video file opens as
    a closed capture, and the old loop skipped entirely — returning a summary and writing
    a handover JSON that claimed a clean zero-frame job.
    """
    bogus = tmp_path / "not_really_a_video.mp4"
    bogus.write_bytes(b"this is not an mp4")

    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None))

    with pytest.raises(ValueError, match="Could not decode"):
        pipeline.process_video(str(bogus), output_json_path=str(tmp_path / "handover.json"))

    # And no misleading handover artifact is left behind.
    assert not (tmp_path / "handover.json").exists()


def test_process_video_still_raises_on_missing_file(tmp_path):
    """A genuinely absent path keeps its distinct error type."""
    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None))

    with pytest.raises(FileNotFoundError):
        pipeline.process_video(str(tmp_path / "nope.mp4"))


def test_detector_raises_when_weights_are_missing(tmp_path):
    """Missing weights must fail loudly at construction, not degrade silently.

    A detector that quietly loads nothing reports an all-clear on every frame, which on a
    surveillance system is indistinguishable from a camera watching an empty corridor.
    """
    missing = str(tmp_path / "absent_weights.pt")

    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        IncidentDetector(model_path=missing)


def test_rtsp_simulator_configs():
    configs = RTSPSimulator.generate_mock_camera_configs(count=10)
    assert len(configs) == 10
    assert configs[0]["camera_id"] == "cam_01"
    assert "rtsp://" in configs[0]["rtsp_url"]


def test_multi_stream_manager():
    manager = MultiStreamManager()

    # Add 4 simulated camera streams (autostart=False for test)
    configs = RTSPSimulator.generate_mock_camera_configs(count=4)
    for cfg in configs:
        manager.add_camera(cfg["camera_id"], cfg["rtsp_url"], autostart=False)

    status = manager.get_status()
    assert status["total_cameras"] == 4
    assert status["active_cameras"] == 0
    assert status["average_latency_p95_ms"] == 0.0
    assert status["processing_errors"] == 0

    # stop_all drops the workers: keeping stopped threads in the registry made
    # get_status report cameras that can never run again.
    manager.stop_all()
    assert len(manager.workers) == 0
    assert manager.get_status()["total_cameras"] == 0


def test_manager_coalesces_slow_frame_callbacks_without_blocking_workers():
    """Preview encoding may lag, but camera inference callbacks must return immediately."""
    callback_started = threading.Event()
    release_callback = threading.Event()
    seen: list[int] = []

    def slow_callback(camera_id, result, frame):
        callback_started.set()
        release_callback.wait(timeout=2.0)
        seen.append(result.frame_id)

    manager = MultiStreamManager(
        global_on_frame_callback=slow_callback,
        tune_torch_threads=False,
        max_concurrent_inferences=1,
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    manager._handle_worker_result(
        "cam_01", FrameAnalysisResult(frame_id=1, camera_id="cam_01"), frame
    )
    assert callback_started.wait(timeout=1.0)

    started = time.perf_counter()
    for frame_id in range(2, 12):
        manager._handle_worker_result(
            "cam_01",
            FrameAnalysisResult(frame_id=frame_id, camera_id="cam_01"),
            frame,
        )
    enqueue_elapsed = time.perf_counter() - started

    release_callback.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and (not seen or seen[-1] != 11):
        time.sleep(0.01)
    status = manager.get_status()
    manager.stop_all()

    assert enqueue_elapsed < 0.1
    assert seen == [1, 11]
    assert status["replaced_frame_callbacks"] == 9


def test_manager_shares_one_inference_limiter_across_camera_workers():
    manager = MultiStreamManager(
        tune_torch_threads=False,
        max_concurrent_inferences=1,
    )
    first = manager.add_camera("cam_01", "rtsp://host/1", autostart=False)
    second = manager.add_camera("cam_02", "rtsp://host/2", autostart=False)

    assert first.inference_semaphore is manager._inference_semaphore
    assert second.inference_semaphore is manager._inference_semaphore
    assert manager.get_status()["max_concurrent_inferences"] == 1
    manager.stop_all()
