import os

import numpy as np
import pytest

from src.vision import (
    BoundingBox,
    FaceAnonymizer,
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

    frame_result = FrameAnalysisResult(frame_id=1, camera_id="cam_hallway_01", detections=[detection], processing_time_ms=12.5)
    assert frame_result.frame_id == 1
    assert len(frame_result.detections) == 1


def test_detector_mock_mode():
    detector = IncidentDetector(model_path=None)
    assert detector.is_mock is True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)

    assert isinstance(detections, list)
    assert len(detections) > 0
    assert detections[0].confidence > 0.0


def test_anonymizer_blur():
    anonymizer = FaceAnonymizer()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test blur with manually specified face box (x, y, w, h)
    faces = [(100, 100, 50, 50)]
    anonymized = anonymizer.blur_faces(dummy_frame, faces=faces, method="gaussian")

    assert anonymized.shape == dummy_frame.shape
    assert isinstance(anonymized, np.ndarray)


def test_vision_pipeline_frame_process():
    pipeline = VisionPipeline()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    anonymized_frame, result = pipeline.process_frame(dummy_frame, frame_id=42)

    assert anonymized_frame.shape == dummy_frame.shape
    assert result.frame_id == 42
    assert result.processing_time_ms >= 0.0


def test_anonymizer_pixelate_face_clipped_at_frame_edge():
    """A face box running past the frame edge must not crash the pixelate path."""
    anonymizer = FaceAnonymizer()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Box starts at x=600 with w=80 -> clipped to 40px wide against the 640px frame
    anonymized = anonymizer.blur_faces(dummy_frame, faces=[(600, 100, 80, 80)], method="pixelate")

    assert anonymized.shape == dummy_frame.shape


def test_camera_id_propagates_into_detections():
    """camera_id must reach every detection, not just the frame result."""
    # Pinned to mock: this asserts camera_id plumbing, and the scripted mock is the only
    # thing that yields detections on a blank frame. With the default detector this would
    # load real weights, find nothing in black pixels, and fail for the wrong reason.
    pipeline = VisionPipeline(detector=IncidentDetector(model_path=None))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _, result = pipeline.process_frame(dummy_frame, frame_id=5, camera_id="cam_lobby_07")

    assert result.camera_id == "cam_lobby_07"
    assert result.timestamp is not None
    assert len(result.detections) > 0
    assert all(d.camera_id == "cam_lobby_07" for d in result.detections)


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


def test_manager_warns_when_cameras_exceed_cpu_capacity(caplog):
    """Oversubscription must be surfaced: frames go stale silently otherwise."""
    import logging as _logging

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


def test_detector_can_refuse_silent_mock_fallback(tmp_path):
    """Production callers must be able to demand real weights.

    Falling back to mock mode silently means a surveillance system reports plausible
    detections that have nothing to do with the camera feed.
    """
    missing = str(tmp_path / "absent_weights.pt")

    with pytest.raises(FileNotFoundError, match="allow_mock_fallback"):
        IncidentDetector(model_path=missing, allow_mock_fallback=False)

    # Default stays permissive so tests and demos keep working.
    assert IncidentDetector(model_path=missing).is_mock is True


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

    # stop_all drops the workers: keeping stopped threads in the registry made
    # get_status report cameras that can never run again.
    manager.stop_all()
    assert len(manager.workers) == 0
    assert manager.get_status()["total_cameras"] == 0
