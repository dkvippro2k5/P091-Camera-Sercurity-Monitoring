import logging
import queue
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from src.vision.pipeline import VisionPipeline
from src.vision.schemas import FrameAnalysisResult

logger = logging.getLogger(__name__)


class CameraStreamWorker:
    """Individual Camera Worker thread that reads from an RTSP stream (or simulated video source),
    runs the Vision Pipeline (face anonymization + incident detection), and dispatches handover JSON metadata.

    Features:
      - Threaded non-blocking RTSP stream ingestion
      - Automatic frame-skipping buffer to prevent lag under heavy load
      - Auto-reconnection logic if RTSP stream drops
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        pipeline: VisionPipeline | None = None,
        on_result_callback: Callable[[str, FrameAnalysisResult, np.ndarray], None] | None = None,
        target_fps: float = 15.0,
        reconnect_interval_sec: float = 3.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.pipeline = pipeline or VisionPipeline()
        self.on_result_callback = on_result_callback
        self.target_fps = target_fps
        self.reconnect_interval_sec = reconnect_interval_sec

        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)  # Keep buffer small to prevent lag
        self._frame_count = 0
        self._processed_count = 0
        self._last_fps = 0.0

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name=f"CameraWorker-{self.camera_id}", daemon=True)
        self._thread.start()
        logger.info(f"Started CameraStreamWorker for '{self.camera_id}' ({self.rtsp_url})")

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info(f"Stopped CameraStreamWorker for '{self.camera_id}'")

    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    def get_stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "rtsp_url": self.rtsp_url,
            "running": self.is_running(),
            "total_read_frames": self._frame_count,
            "processed_frames": self._processed_count,
            "current_fps": self._last_fps,
        }

    def _worker_loop(self) -> None:
        cap = None
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        last_process_time = time.perf_counter()

        while self._running:
            # 1. Connect or Reconnect RTSP Stream
            if cap is None or not cap.isOpened():
                logger.info(f"[{self.camera_id}] Opening RTSP stream source: {self.rtsp_url}")
                # Handle numeric RTSP / camera indices or stream URLs
                source = int(self.rtsp_url) if self.rtsp_url.isdigit() else self.rtsp_url
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    logger.warning(f"[{self.camera_id}] Failed to open stream. Retrying in {self.reconnect_interval_sec}s...")
                    time.sleep(self.reconnect_interval_sec)
                    continue

            # 2. Read frame
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning(f"[{self.camera_id}] Frame read failed or stream end. Reconnecting...")
                cap.release()
                cap = None
                time.sleep(1.0)
                continue

            self._frame_count += 1
            now = time.perf_counter()

            # Throttle processing to target FPS
            if now - last_process_time < frame_interval:
                continue

            last_process_time = now

            # 3. Process frame with Vision Pipeline
            try:
                anonymized_frame, result = self.pipeline.process_frame(
                    frame, frame_id=self._frame_count, camera_id=self.camera_id
                )
                self._processed_count += 1

                # Update FPS calculation
                self._last_fps = round(1.0 / max(0.001, (time.perf_counter() - now)), 2)

                # 4. Trigger Callback if registered
                if self.on_result_callback:
                    self.on_result_callback(self.camera_id, result, anonymized_frame)

            except Exception as e:
                logger.error(f"[{self.camera_id}] Error in vision pipeline processing: {e}")

        if cap:
            cap.release()
