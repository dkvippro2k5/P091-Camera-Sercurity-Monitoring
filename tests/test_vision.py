import numpy as np

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

    # Clean up
    manager.stop_all()
    assert len(manager.workers) == 4
