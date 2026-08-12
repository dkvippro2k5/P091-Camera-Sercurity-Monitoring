import logging
import os
import threading
from collections.abc import Callable

import numpy as np

from src.vision.detector import IncidentDetector
from src.vision.motion import MotionGate
from src.vision.pipeline import VisionPipeline
from src.vision.rule_engine import RuleEngine
from src.vision.schemas import FrameAnalysisResult
from src.vision.stream_worker import CameraStreamWorker

logger = logging.getLogger(__name__)

# Torch defaults to one intra-op thread per core. That is right for a single inference at
# a time, but with several camera workers each running their own inference the pools
# oversubscribe the CPU and fight. Measured on 8 cores with 4 cameras at 720p:
#   intra-op threads = 8 (default) -> 11.9 fps aggregate
#   intra-op threads = 2           -> 13.9 fps aggregate
#   intra-op threads = 1           -> 13.5 fps aggregate
# Two is the sweet spot here: enough parallelism per inference, little contention.
_TORCH_THREADS_PER_WORKER = 2

_torch_tuned = False
_torch_tune_lock = threading.Lock()


def tune_torch_for_multistream(threads_per_worker: int = _TORCH_THREADS_PER_WORKER) -> bool:
    """Cap torch's intra-op thread pool so concurrent camera workers stop fighting.

    Process-global and only applied once. No effect on GPU inference, where the
    bottleneck is elsewhere entirely.

    Args:
        threads_per_worker: Intra-op threads each inference may use.

    Returns:
        True if the setting was applied, False if it was already tuned, torch is
        unavailable, or CUDA is in use.
    """
    global _torch_tuned

    with _torch_tune_lock:
        if _torch_tuned:
            return False
        try:
            import torch
        except ImportError:
            return False

        _torch_tuned = True

        if torch.cuda.is_available():
            logger.debug("CUDA present; leaving torch thread settings alone.")
            return False

        try:
            # set_num_threads raises if a parallel region has already started, which is
            # why this wants to run before the first inference.
            torch.set_num_threads(threads_per_worker)
        except RuntimeError as e:
            logger.warning(f"Could not tune torch threads: {e}")
            return False

        logger.info(
            f"Capped torch intra-op threads to {threads_per_worker} for multi-stream CPU "
            "inference (measured ~17% aggregate throughput gain on 4 cameras)."
        )
        return True


class MultiStreamManager:
    """Parallel multi-stream RTSP manager: one worker thread per camera.

    Each camera runs in its own worker with automatic reconnection, frame skipping, and
    thread-safe handover event dispatching. Each worker gets its own VisionPipeline —
    YOLO's tracker state and the rule engine's dwell timers are per-stream and would
    corrupt each other if shared.

    CAPACITY — measured, not aspirational
    ------------------------------------
    Inference dominates everything else, so the honest limit is set by the hardware.
    Measured on 8 CPU cores (torch 2.13+cpu), yolo11n, 720p, full pipeline:

        1 camera  ->  5.5 fps aggregate
        2 cameras ->  9.1 fps aggregate (4.6 each)
        4 cameras -> 11.9 fps aggregate (3.0 each)

    Throughput plateaus around 12-15 fps *in total* regardless of how many workers are
    added: per-camera rate degrades linearly past that point. Practical limits:

      - CPU-only, this class as written:      3-4 cameras at a useful frame rate
      - Single mid-range GPU + batching:      20-50 cameras (needs a batched detector,
                                              which this class does NOT implement)
      - Hundreds of cameras:                  requires horizontal scaling — N processes
                                              each owning a slice of cameras, GPU-backed.
                                              That is a different deployment topology,
                                              not a tuning knob on this class.

    `target_fps` is a ceiling, not a promise. Setting it to 15 on 40 CPU cameras does not
    make them run at 15; it just means nothing throttles them below the hardware limit.
    Check `get_status()["average_inference_capacity_fps"]` against your target to see
    whether the box is actually keeping up.

    MOTION GATING — A/B, same box, 2026-08-05
    -----------------------------------------
    `MotionGate` (motion.py) defaults OFF, which is why the numbers above still stand. It
    buys throughput only on an idle scene, and the A/B says so plainly:

        busy footage (input_test.mp4, 200 frames, moving people):
            gate off -> 5.36 fps, 200 inferences, 0.0% skipped, 2 events
            gate on  -> 5.72 fps, 200 inferences, 0.0% skipped, 2 events

        idle 720p scene (static frame + sensor noise, 120 frames):
            gate off -> 5.62 fps,   120 inferences,  0.0% skipped
            gate on  -> 136.89 fps,   4 inferences, 96.7% skipped

    Two things worth reading off this. The gate never skips while anything is moving or
    any track is alive, so on busy footage it is a no-op — the 6% is noise, not a win, and
    both runs emit the *same 2 events*, which is the property that matters. The win is the
    3am corridor, where it is ~24x. It also shows the 21/255 threshold correctly treats
    sensor noise as "still" rather than churning inference on it.
    """

    def __init__(
        self,
        shared_pipeline: VisionPipeline | None = None,
        pipeline_factory: Callable[[], VisionPipeline] | None = None,
        global_on_incident_callback: Callable[[str, FrameAnalysisResult, np.ndarray], None] | None = None,
        global_on_frame_callback: Callable[[str, FrameAnalysisResult, np.ndarray], None] | None = None,
        tune_torch_threads: bool = True,
    ):
        """Initialize the multi-stream manager.

        Args:
            shared_pipeline: A single pipeline used as a template to create per-worker
                instances. Kept for backward compatibility — when provided without a
                factory, a factory is built that clones the shared pipeline's config.
            pipeline_factory: Callable that returns a fresh VisionPipeline for each worker.
                Takes priority over shared_pipeline when both are provided.
            global_on_incident_callback: Called only for frames carrying rule-engine
                events. This is the alerting path — keep it cheap.
            global_on_frame_callback: Called for *every* analyzed frame, incident or not.
                This is what a live preview or recorder needs; wiring those into the
                incident callback instead leaves the view frozen between events. Runs on
                the worker thread, so it must not block or touch a GUI directly.
            tune_torch_threads: Cap torch's intra-op threads for concurrent CPU inference.
                Process-global, so pass False if the host process manages its own torch
                threading or shares the interpreter with unrelated torch work.
        """
        if tune_torch_threads:
            tune_torch_for_multistream()

        self._pipeline_factory = pipeline_factory or self._make_default_factory(shared_pipeline)
        self.global_on_incident_callback = global_on_incident_callback
        self.global_on_frame_callback = global_on_frame_callback
        self.workers: dict[str, CameraStreamWorker] = {}
        # add/remove are called from API request threads while get_status/start_all/stop_all
        # iterate; without this the iteration raises "dictionary changed size".
        self._workers_lock = threading.RLock()
        self._oversubscription_warned = False

    @staticmethod
    def _make_default_factory(
        template: VisionPipeline | None = None,
    ) -> Callable[[], VisionPipeline]:
        """Build a factory creating independent pipelines that mirror the template.

        Each worker needs its own detector and rule engine — YOLO's tracker state and the
        engine's dwell timers are per-stream and would corrupt each other if shared. The
        template's *configuration* is carried across so a caller-supplied model path,
        confidence threshold, or zone set is not silently replaced by defaults.
        """
        if template is None:
            return lambda: VisionPipeline()

        detector = template.detector
        engine = template.rule_engine
        gate = template.motion_gate

        def factory() -> VisionPipeline:
            return VisionPipeline(
                detector=IncidentDetector(
                    model_path=detector.model_path,
                    conf_threshold=detector.conf_threshold,
                    target_labels=detector.target_labels,
                    use_tracking=detector.use_tracking,
                ),
                rule_engine=RuleEngine(
                    # Zones are immutable config and carry no per-stream state, so the
                    # manager instance is safe to share across workers.
                    zone_manager=engine.zone_manager,
                    fall_dwell_sec=engine.fall_dwell_sec,
                    fall_aspect_ratio=engine.fall_aspect_ratio,
                    fall_upright_aspect_ratio=engine.fall_upright_aspect_ratio,
                    fall_velocity_ratio=engine.fall_velocity_ratio,
                    fall_height_drop_ratio=engine.fall_height_drop_ratio,
                    fall_min_indicators=engine.fall_min_indicators,
                    abandon_sec=engine.abandon_sec,
                    owner_radius_ratio=engine.owner_radius_ratio,
                    stationary_tolerance_ratio=engine.stationary_tolerance_ratio,
                    track_ttl_sec=engine.track_ttl_sec,
                    track_stitch_window_sec=engine.track_stitch_window_sec,
                    track_stitch_radius_ratio=engine.track_stitch_radius_ratio,
                    alert_cooldown_sec=engine.alert_cooldown_sec,
                ),
                draw_overlays=template.draw_overlays,
                inference_size=template.inference_size,
                # The gate is NOT shared: it holds the previous frame, so one instance
                # across cameras would diff camera A against camera B and see motion
                # everywhere — silently disabling the gate it was added to provide.
                motion_gate=MotionGate(
                    enabled=gate.enabled,
                    pixel_threshold=gate.pixel_threshold,
                    min_changed_fraction=gate.min_changed_fraction,
                    heartbeat_sec=gate.heartbeat_sec,
                    diff_size=gate.diff_size,
                ),
            )

        return factory

    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        target_fps: float = 15.0,
        autostart: bool = True,
        max_reconnect_attempts: int = 0,
        loop_finite_sources: bool = False,
    ) -> CameraStreamWorker:
        """Add a new camera RTSP stream to the manager.

        Args:
            camera_id: Unique string identifier for the camera (e.g. 'cam_lobby_01').
            rtsp_url: RTSP stream URL or local video path / device index.
            target_fps: Maximum processing FPS for this stream.
            autostart: Whether to start worker immediately.
            max_reconnect_attempts: 0 retries forever; >0 gives up after that many
                consecutive failures to open or to read a frame.
            loop_finite_sources: When True, a file-based source restarts from the
                beginning at EOF instead of stopping.

        Returns:
            The created CameraStreamWorker instance.
        """
        with self._workers_lock:
            if camera_id in self.workers:
                logger.warning(f"Camera '{camera_id}' is already managed. Updating RTSP URL...")
                self.remove_camera(camera_id)

            worker = CameraStreamWorker(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                pipeline=self._pipeline_factory(),
                on_result_callback=self._handle_worker_result,
                target_fps=target_fps,
                max_reconnect_attempts=max_reconnect_attempts,
                loop_finite_sources=loop_finite_sources,
            )
            self.workers[camera_id] = worker

        if autostart:
            worker.start()

        self._warn_if_oversubscribed()
        return worker

    def _warn_if_oversubscribed(self) -> None:
        """Log once when camera count passes what CPU inference can plausibly serve.

        Silent oversubscription is the failure mode that matters here: every camera keeps
        "running", frames just quietly get stale, and nothing in the status output says
        the box stopped keeping up.
        """
        if self._oversubscription_warned:
            return

        try:
            import torch

            if torch.cuda.is_available():
                return  # GPU capacity is a different curve; don't guess at it.
        except ImportError:
            pass

        # ~3-4 cameras per 8 cores at a useful frame rate, from the measurements in the
        # class docstring. Scaled by core count so this is not hardcoded to one machine.
        budget = max(2, (os.cpu_count() or 4) // 2)
        if len(self.workers) > budget:
            self._oversubscription_warned = True
            logger.warning(
                f"{len(self.workers)} cameras on {os.cpu_count()} CPU cores exceeds the "
                f"~{budget} this host can serve at a useful frame rate. Frames will go "
                "stale rather than error. Check average_inference_capacity_fps in "
                "get_status(), and see MultiStreamManager's docstring for scaling options."
            )

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera stream by ID."""
        with self._workers_lock:
            worker = self.workers.pop(camera_id, None)

        if worker is None:
            return False

        worker.stop()
        logger.info(f"Removed camera '{camera_id}' from MultiStreamManager")
        return True

    def start_all(self) -> None:
        """Start workers for all added cameras."""
        with self._workers_lock:
            workers = list(self.workers.values())

        for worker in workers:
            if not worker.is_running():
                worker.start()

    def stop_all(self) -> None:
        """Stop all camera worker threads and drop them from the registry."""
        with self._workers_lock:
            workers = list(self.workers.values())
            self.workers.clear()

        for worker in workers:
            worker.stop()
        logger.info("Stopped all camera streams in MultiStreamManager")

    def get_status(self) -> dict[str, object]:
        """Return system-wide statistics for all camera streams."""
        with self._workers_lock:
            workers = list(self.workers.values())

        camera_stats = [w.get_stats() for w in workers]
        active_count = sum(1 for s in camera_stats if s["running"])
        total_processed = sum(s["processed_frames"] for s in camera_stats)
        # Averaging inference capacity, not achieved throughput — see CameraStreamWorker
        # .get_stats. This is headroom per camera, so it stays high on an idle system.
        avg_capacity = (
            round(
                sum(s["inference_capacity_fps"] for s in camera_stats if s["running"]) / max(1, active_count),
                2,
            )
            if active_count > 0
            else 0.0
        )

        return {
            "total_cameras": len(camera_stats),
            "active_cameras": active_count,
            "average_inference_capacity_fps": avg_capacity,
            "total_processed_frames": total_processed,
            "cameras": camera_stats,
        }

    def _handle_worker_result(self, camera_id: str, result: FrameAnalysisResult, frame: np.ndarray) -> None:
        """Internal handler called by worker threads on every analyzed frame.

        Fans out to two callbacks with different gating: every frame goes to the frame
        callback (preview, recording), while only frames carrying rule-engine events go
        to the incident callback (alerting).
        """
        # Every frame. A preview or recorder driven off the incident callback would sit
        # frozen between events, which reads as "the demo is broken".
        if self.global_on_frame_callback:
            try:
                self.global_on_frame_callback(camera_id, result, frame)
            except Exception as e:
                logger.error(f"Error in global frame callback for '{camera_id}': {e}")

        # Events only. Gating on `detections` too would fire on every frame containing
        # any person at all, which is the flooding this check exists to prevent.
        if result.events and self.global_on_incident_callback:
            try:
                self.global_on_incident_callback(camera_id, result, frame)
            except Exception as e:
                logger.error(f"Error in global incident callback for '{camera_id}': {e}")
