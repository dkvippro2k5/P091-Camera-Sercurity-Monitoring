import logging
from collections.abc import Callable

import numpy as np

from src.vision.pipeline import VisionPipeline
from src.vision.schemas import FrameAnalysisResult
from src.vision.stream_worker import CameraStreamWorker

logger = logging.getLogger(__name__)


class MultiStreamManager:
    """Scalable Parallel Multi-Stream RTSP Manager for urban camera networks.

    Allows managing 4, 10, 50, or hundreds of parallel RTSP camera streams.
    Each camera runs in its own Worker thread with automatic reconnection, frame skipping,
    and thread-safe handover event dispatching.
    """

    def __init__(
        self,
        shared_pipeline: VisionPipeline | None = None,
        global_on_incident_callback: Callable[[str, FrameAnalysisResult, np.ndarray], None] | None = None,
    ):
        self.shared_pipeline = shared_pipeline or VisionPipeline()
        self.global_on_incident_callback = global_on_incident_callback
        self.workers: dict[str, CameraStreamWorker] = {}

    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        target_fps: float = 15.0,
        autostart: bool = True,
    ) -> CameraStreamWorker:
        """Add a new camera RTSP stream to the manager.

        Args:
            camera_id: Unique string identifier for the camera (e.g. 'cam_lobby_01').
            rtsp_url: RTSP stream URL or local video path / device index.
            target_fps: Maximum processing FPS for this stream.
            autostart: Whether to start worker immediately.

        Returns:
            The created CameraStreamWorker instance.
        """
        if camera_id in self.workers:
            logger.warning(f"Camera '{camera_id}' is already managed. Updating RTSP URL...")
            self.remove_camera(camera_id)

        worker = CameraStreamWorker(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            pipeline=self.shared_pipeline,
            on_result_callback=self._handle_worker_result,
            target_fps=target_fps,
        )
        self.workers[camera_id] = worker

        if autostart:
            worker.start()

        return worker

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera stream by ID."""
        if camera_id not in self.workers:
            return False

        worker = self.workers.pop(camera_id)
        worker.stop()
        logger.info(f"Removed camera '{camera_id}' from MultiStreamManager")
        return True

    def start_all(self) -> None:
        """Start workers for all added cameras."""
        for worker in self.workers.values():
            if not worker.is_running():
                worker.start()

    def stop_all(self) -> None:
        """Stop all camera worker threads."""
        for camera_id, worker in list(self.workers.items()):
            worker.stop()
        logger.info("Stopped all camera streams in MultiStreamManager")

    def get_status(self) -> dict[str, object]:
        """Return system-wide statistics for all camera streams."""
        active_count = sum(1 for w in self.workers.values() if w.is_running())
        total_processed = sum(w._processed_count for w in self.workers.values())
        avg_fps = (
            round(sum(w._last_fps for w in self.workers.values() if w.is_running()) / max(1, active_count), 2)
            if active_count > 0
            else 0.0
        )

        return {
            "total_cameras": len(self.workers),
            "active_cameras": active_count,
            "average_system_fps": avg_fps,
            "total_processed_frames": total_processed,
            "cameras": [w.get_stats() for w in self.workers.values()],
        }

    def _handle_worker_result(
        self, camera_id: str, result: FrameAnalysisResult, frame: np.ndarray
    ) -> None:
        """Internal handler called by worker threads on every analyzed frame.
        Triggers global callback if incident detections are present.
        """
        # Only dispatch when there are active detections to avoid flooding callbacks
        if result.detections and self.global_on_incident_callback:
            try:
                self.global_on_incident_callback(camera_id, result, frame)
            except Exception as e:
                logger.error(f"Error in global incident callback for '{camera_id}': {e}")
