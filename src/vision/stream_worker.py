import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable

import cv2
import numpy as np

from src.vision.h264_encoder import H264StreamEncoder
from src.vision.pipeline import VisionPipeline
from src.vision.schemas import FrameAnalysisResult

logger = logging.getLogger(__name__)

# FFmpeg reads its RTSP options from this process-global env var, which has to be set
# before the capture is constructed. Every worker wants the same value, so writing it
# repeatedly is harmless; the lock only keeps concurrent writes from interleaving.
#
# Scope discipline matters here: cv2.VideoCapture(...) is a blocking call that can take up
# to open_timeout_sec, and it must never be made *inside* this lock. Every caller across the
# process shares one lock instance, so holding it around the open would serialize every
# camera start and every preview-frame request in the system behind whichever one is
# currently connecting — a large, easy-to-miss latency hit disguised as a one-line env var
# guard. Grab the lock only around the setdefault(); construct the capture after releasing it.
_FFMPEG_OPTS_LOCK = threading.Lock()
_RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp"
_LATENCY_WINDOW_SIZE = 100
# Mirrors monitoring.py's _PREVIEW_JPEG_QUALITY. Duplicated rather than imported: monitoring
# imports stream_manager which imports this module, so importing the other way would cycle.
_DISPLAY_JPEG_QUALITY = 65


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile for a small bounded telemetry window."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 2)


class CameraStreamWorker:
    """Individual Camera Worker that reads from an RTSP stream (or simulated video source),
    runs the Vision Pipeline (incident detection), and dispatches handover JSON metadata.

    Features:
      - Dedicated reader thread: decouples RTSP I/O jitter from inference latency
      - grab()/retrieve() pattern: skipped frames demux only (no decode), saving 80%+ CPU
      - Exponential backoff on reconnection with optional circuit-breaker
      - Graceful shutdown via threading.Event
      - RTSP transport optimization (TCP preference, open/read timeouts)
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        pipeline: VisionPipeline | None = None,
        on_result_callback: Callable[[str, FrameAnalysisResult, np.ndarray], None] | None = None,
        on_raw_frame_callback: Callable[[str, np.ndarray], None] | None = None,
        target_fps: float = 15.0,
        reconnect_interval_sec: float = 3.0,
        max_reconnect_attempts: int = 0,
        open_timeout_sec: float = 10.0,
        read_timeout_sec: float = 10.0,
        loop_finite_sources: bool = False,
        inference_semaphore: threading.Semaphore | None = None,
        display_fps: float = 20.0,
        display_max_height: int = 480,
        h264_enabled: bool = True,
        h264_bitrate: int = 1_200_000,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.pipeline = pipeline or VisionPipeline()
        self.on_result_callback = on_result_callback
        # Called from the display loop at display_fps with the raw (un-annotated) frame, so
        # the VLM pre-roll ring fills at a fixed cadence independent of how fast — or slow —
        # YOLO is keeping up. On a CPU host with several cameras the inference callback can
        # drop to ~1 fps, which left the ring too sparse for _collect_clip to build a clip.
        self.on_raw_frame_callback = on_raw_frame_callback
        self.target_fps = target_fps
        # Decoupled from target_fps on purpose: the live view can be smooth without paying for
        # YOLO at that rate — see docs/plans/multi-camera-streaming-scale.md's "phương pháp 0".
        # Not exposed through CameraCreate/the API: there is no second stream to configure,
        # this only redraws the same source's frames at a cheaper cadence.
        self.display_fps = display_fps
        self.display_max_height = display_max_height
        self.reconnect_interval_sec = reconnect_interval_sec
        # 0 means unlimited retries (original behavior); >0 is a circuit-breaker.
        self.max_reconnect_attempts = max_reconnect_attempts
        # Without these FFmpeg blocks indefinitely on an unreachable host, and neither
        # stop() nor the circuit-breaker can interrupt it.
        self.open_timeout_sec = open_timeout_sec
        self.read_timeout_sec = read_timeout_sec
        # A file or other finite source hits EOF legitimately. Treating that as a dropped
        # connection restarts it from frame 0 forever; opt in explicitly for demo loops.
        self.loop_finite_sources = loop_finite_sources
        self.inference_semaphore = inference_semaphore

        self._stop_event = threading.Event()
        self._process_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._display_thread: threading.Thread | None = None

        # Single-slot latest-frame buffer shared between reader and processor.
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id: int = 0
        self._frame_lock = threading.Lock()

        # Display loop's output. No lock: same reasoning as monitoring.py's self._frames —
        # a plain bytes-reference reassignment is atomic under the GIL, and readers only ever
        # want whatever is newest, so a lock would only add contention for no correctness gain.
        self._latest_display_jpeg: bytes | None = None

        # H.264 stream fed from the same display frames. Kept alongside the JPEG buffer
        # rather than replacing it: MJPEG stays the fallback for a browser without
        # MediaSource support and for the snapshot endpoint, and it costs ~1.2ms/frame.
        # Constructed eagerly but opened lazily on the first frame, so a host without a
        # usable encoder degrades to MJPEG instead of failing to start the camera.
        self._h264_bitrate = h264_bitrate
        self._h264_encoder: H264StreamEncoder | None = (
            H264StreamEncoder(
                camera_id,
                fps=display_fps,
                bitrate=h264_bitrate,
                # ~2s at the default display rate. This is the longest a joining viewer can
                # wait for its first picture, traded against the bandwidth of sending full
                # keyframes more often.
                keyframe_interval_frames=max(1, int(round(display_fps * 2))),
            )
            if h264_enabled
            else None
        )

        # Stats (read from main thread via get_stats, written from worker threads).
        self._lock_stats = threading.Lock()
        self._frame_count = 0
        self._processed_count = 0
        self._last_fps = 0.0
        self._last_latency_ms = 0.0
        self._latencies_ms: deque[float] = deque(maxlen=_LATENCY_WINDOW_SIZE)
        self._last_inference_wait_ms = 0.0
        self._inference_waits_ms: deque[float] = deque(maxlen=_LATENCY_WINDOW_SIZE)
        self._processing_errors = 0
        self._reconnect_count = 0
        self._stopped_reason: str | None = None

    def start(self) -> None:
        """Start the reader, processing, and display threads."""
        if not self._stop_event.is_set() and self._process_thread is not None:
            return
        self._stop_event.clear()

        # stop() closes the encoder permanently — a restart of this same worker therefore
        # needs a fresh one, or the display loop would encode into a closed container and
        # every viewer would be told to fall back to MJPEG. The manager currently builds a
        # new worker per (re)start, so this only matters for a direct stop()/start() pair,
        # which the tests do and a future caller reasonably might.
        if self._h264_encoder is not None and not self._h264_encoder.available:
            self._h264_encoder = H264StreamEncoder(
                self.camera_id,
                fps=self.display_fps,
                bitrate=self._h264_bitrate,
                keyframe_interval_frames=max(1, int(round(self.display_fps * 2))),
            )

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"CameraReader-{self.camera_id}",
            daemon=True,
        )
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name=f"CameraWorker-{self.camera_id}",
            daemon=True,
        )
        self._display_thread = threading.Thread(
            target=self._display_loop,
            name=f"CameraDisplay-{self.camera_id}",
            daemon=True,
        )
        self._reader_thread.start()
        self._process_thread.start()
        self._display_thread.start()
        logger.info(f"Started CameraStreamWorker for '{self.camera_id}' ({self.rtsp_url})")

    def stop(self) -> None:
        """Signal all threads to stop and wait for them.

        A reader blocked deep inside FFmpeg cannot be interrupted by the event alone —
        the open/read timeouts are what bound that wait. If a thread outlives the join
        it is reported rather than silently leaked.
        """
        self._stop_event.set()
        for thread in (self._reader_thread, self._process_thread, self._display_thread):
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    logger.warning(
                        f"[{self.camera_id}] Thread '{thread.name}' did not exit within 3s; "
                        "it is blocked in FFmpeg and will exit once its I/O times out."
                    )
        # After the display thread is joined, so nothing is mid-encode. Closing wakes every
        # WebSocket viewer, which is what lets their handlers notice the camera is gone
        # instead of blocking on a queue no producer will ever fill again.
        if self._h264_encoder is not None:
            self._h264_encoder.close()
        logger.info(f"Stopped CameraStreamWorker for '{self.camera_id}'")

    def is_running(self) -> bool:
        return (
            not self._stop_event.is_set()
            and self._process_thread is not None
            and self._process_thread.is_alive()
        )

    def get_latest_display_jpeg(self) -> bytes | None:
        """Newest frame produced by `_display_loop` — the high-cadence view, decoupled from
        target_fps. `None` until the display thread has encoded its first frame."""
        return self._latest_display_jpeg

    def get_h264_encoder(self) -> H264StreamEncoder | None:
        """This camera's fMP4 encoder, or None when H.264 is disabled for it.

        A returned encoder can still be unusable — check `.available`, which goes false when
        no encoder could be opened on this host. The video WebSocket uses that to tell the
        client to fall back to MJPEG rather than wait on a stream that will never arrive.
        """
        return self._h264_encoder

    def get_stats(self) -> dict:
        with self._lock_stats:
            latencies = list(self._latencies_ms)
            return {
                "camera_id": self.camera_id,
                "rtsp_url": self.rtsp_url,
                "running": self.is_running(),
                "total_read_frames": self._frame_count,
                "processed_frames": self._processed_count,
                # Inference throughput ceiling (1 / per-frame latency), NOT the achieved
                # rate — that is capped by target_fps and by how fast the source delivers.
                "inference_capacity_fps": self._last_fps,
                "last_latency_ms": self._last_latency_ms,
                "latency_p50_ms": _percentile(latencies, 0.50),
                "latency_p95_ms": _percentile(latencies, 0.95),
                "latency_p99_ms": _percentile(latencies, 0.99),
                "latency_window_frames": len(latencies),
                "last_inference_wait_ms": self._last_inference_wait_ms,
                "inference_wait_p95_ms": _percentile(list(self._inference_waits_ms), 0.95),
                "inference_wait_max_ms": (
                    max(self._inference_waits_ms) if self._inference_waits_ms else 0.0
                ),
                "processing_errors": self._processing_errors,
                "reconnect_count": self._reconnect_count,
                "stopped_reason": self._stopped_reason,
            }

    # ------------------------------------------------------------------ reader
    def _open_capture(self) -> cv2.VideoCapture | None:
        """Open a VideoCapture with RTSP-optimized settings."""
        source = int(self.rtsp_url) if self.rtsp_url.isdigit() else self.rtsp_url
        is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")

        if is_rtsp:
            # Both settings must be in place *before* the connection is attempted, so they
            # go through the constructor: an open timeout applied afterwards has nothing
            # left to bound. FFmpeg reads its transport choice from the env var only.
            # Default UDP silently corrupts frames on congested networks.
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(self.open_timeout_sec * 1000),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(self.read_timeout_sec * 1000),
            ]
            with _FFMPEG_OPTS_LOCK:
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _RTSP_FFMPEG_OPTIONS)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

        if cap.isOpened():
            # Keep only the newest frame; combined with the single-slot buffer below this
            # is what bounds latency. Ignored by some backends, so it is a hint, not a
            # guarantee.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap.release()
        return None

    def _is_finite_source(self, cap: cv2.VideoCapture) -> bool:
        """Whether this source has a known end (a file), as opposed to a live stream.

        Live RTSP reports a frame count of 0 or -1; a file reports its real length.
        """
        try:
            return cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        except cv2.error:
            return False

    def _source_frame_interval(self, cap: cv2.VideoCapture) -> float:
        """Seconds between frames for a finite source, or 0.0 for a live stream.

        A live camera paces itself: grab() blocks until the next frame is on the wire. A
        file does not — grab() returns as fast as the disk allows, so an unpaced reader
        consumes the whole file in seconds while the processor, throttled to target_fps,
        sees only a handful of frames. Everything in between is silently discarded by the
        single-slot buffer. Pacing the read to the file's own frame rate makes a file
        source behave like the camera it stands in for.
        """
        if not self._is_finite_source(cap):
            return 0.0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        except cv2.error:
            return 0.0
        if not fps or fps <= 0 or fps != fps or fps == float("inf"):
            return 0.0
        return 1.0 / fps

    def _reader_loop(self) -> None:
        """Dedicated thread: continuously grab frames into a single-slot buffer.

        Only the latest frame matters for live surveillance. Using grab()/retrieve()
        means skipped frames are demuxed but never decoded, saving significant CPU.
        """
        cap: cv2.VideoCapture | None = None
        backoff = self.reconnect_interval_sec
        max_backoff = 60.0
        consecutive_failures = 0
        # >0 only for files; live streams are paced by the network. See _source_frame_interval.
        read_interval = 0.0
        next_read_at = 0.0

        def trip_breaker(reason: str) -> None:
            logger.error(f"[{self.camera_id}] {reason} Giving up.")
            with self._lock_stats:
                self._stopped_reason = reason
            self._stop_event.set()

        while not self._stop_event.is_set():
            # 1. Connect or reconnect
            if cap is None or not cap.isOpened():
                logger.info(f"[{self.camera_id}] Opening source: {self.rtsp_url}")
                cap = self._open_capture()

                if cap is None:
                    consecutive_failures += 1
                    with self._lock_stats:
                        self._reconnect_count += 1

                    # Circuit-breaker
                    if (
                        self.max_reconnect_attempts > 0
                        and consecutive_failures >= self.max_reconnect_attempts
                    ):
                        trip_breaker(
                            f"Reached max reconnect attempts ({self.max_reconnect_attempts})."
                        )
                        break

                    logger.warning(
                        f"[{self.camera_id}] Failed to open stream. "
                        f"Retrying in {backoff:.1f}s (attempt {consecutive_failures})..."
                    )
                    self._stop_event.wait(backoff)
                    # Exponential backoff with cap
                    backoff = min(backoff * 2, max_backoff)
                    continue

                logger.info(f"[{self.camera_id}] Stream opened successfully")
                read_interval = self._source_frame_interval(cap)
                next_read_at = time.monotonic()
                if read_interval > 0:
                    logger.info(
                        f"[{self.camera_id}] Finite source: pacing reads at "
                        f"{1.0 / read_interval:.1f} fps to match the file's frame rate."
                    )

            # 2. Pace a file to its native rate so it behaves like a live camera.
            if read_interval > 0:
                sleep_for = next_read_at - time.monotonic()
                if sleep_for > 0 and self._stop_event.wait(sleep_for):
                    break
                next_read_at += read_interval

            # 3. Grab the latest frame (demux only, no decode)
            if not cap.grab():
                # A finite source reaching its end is not a failure. Reconnecting would
                # restart it from frame 0 and never terminate, so honour the EOF unless
                # looping was explicitly requested.
                if self._is_finite_source(cap) and not self.loop_finite_sources:
                    logger.info(f"[{self.camera_id}] Reached end of finite source.")
                    with self._lock_stats:
                        self._stopped_reason = "end_of_source"
                    self._stop_event.set()
                    break

                # Looping file: seek back to the start and continue immediately.
                if self._is_finite_source(cap) and self.loop_finite_sources:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info(f"[{self.camera_id}] Reached end of file, looping back to start.")
                    continue

                # A stream that opens but delivers nothing is the common camera failure
                # (wedged encoder, auth accepted then dropped). Counting it here is what
                # lets the breaker trip — the open path alone would reset the counter to
                # zero on every successful reconnect and spin forever.
                consecutive_failures += 1
                with self._lock_stats:
                    self._reconnect_count += 1

                cap.release()
                cap = None

                if (
                    self.max_reconnect_attempts > 0
                    and consecutive_failures >= self.max_reconnect_attempts
                ):
                    trip_breaker(
                        f"Stream delivered no frames after {self.max_reconnect_attempts} attempt(s)."
                    )
                    break

                logger.warning(
                    f"[{self.camera_id}] Grab failed. Reconnecting in {backoff:.1f}s "
                    f"(attempt {consecutive_failures})..."
                )
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            # 4. Decode and store in single-slot buffer
            ret, frame = cap.retrieve()
            if not ret or frame is None or frame.size == 0:
                continue

            # A frame actually arrived: the connection is healthy, so re-arm the breaker
            # and the backoff. Doing this on open() instead would credit a connection that
            # never produced a single frame.
            consecutive_failures = 0
            backoff = self.reconnect_interval_sec

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_id += 1

            with self._lock_stats:
                self._frame_count += 1

        if cap:
            cap.release()

    # --------------------------------------------------------------- processor
    def _process_loop(self) -> None:
        """Processing thread: pick the latest frame from the buffer, run the pipeline."""
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        last_processed_id = 0
        next_process_at = time.monotonic()

        while not self._stop_event.is_set():
            # Pace inference starts, not inference completions. Sleeping a full interval
            # after a 180ms inference at a 5fps target produces only ~2.6fps.
            wait_for = max(0.0, next_process_at - time.monotonic())
            self._stop_event.wait(wait_for)
            if self._stop_event.is_set():
                break

            # Grab the latest frame from the buffer
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id

            # Skip if no new frame since last processing
            if frame is None or frame_id == last_processed_id:
                next_process_at = time.monotonic() + frame_interval
                continue
            last_processed_id = frame_id

            # Process frame with Vision Pipeline
            queue_start = time.perf_counter()
            try:
                acquired = False
                if self.inference_semaphore is not None:
                    while not self._stop_event.is_set():
                        acquired = self.inference_semaphore.acquire(timeout=0.1)
                        if acquired:
                            break
                    if not acquired:
                        break

                inference_start = time.perf_counter()
                try:
                    output_frame, result = self.pipeline.process_frame(
                        frame, frame_id=frame_id, camera_id=self.camera_id
                    )
                finally:
                    if acquired:
                        self.inference_semaphore.release()

                elapsed = time.perf_counter() - inference_start
                inference_wait_ms = round((inference_start - queue_start) * 1000, 2)
                with self._lock_stats:
                    self._processed_count += 1
                    self._last_latency_ms = round(elapsed * 1000, 2)
                    self._latencies_ms.append(self._last_latency_ms)
                    self._last_inference_wait_ms = inference_wait_ms
                    self._inference_waits_ms.append(inference_wait_ms)
                    self._last_fps = round(1.0 / max(0.001, elapsed), 2)

                # Trigger callback if registered
                if self.on_result_callback:
                    self.on_result_callback(self.camera_id, result, output_frame)

            except Exception as e:
                with self._lock_stats:
                    self._processing_errors += 1
                logger.error(f"[{self.camera_id}] Error in vision pipeline processing: {e}")
            finally:
                # Do not run catch-up bursts after an overloaded interval; use the newest
                # frame at the next normal slot instead.
                next_process_at = max(next_process_at + frame_interval, time.monotonic())

    # ----------------------------------------------------------------- display
    def _display_loop(self) -> None:
        """Independent thread: redraw the newest raw frame at display_fps, no inference.

        Reads the same single-slot buffer the reader thread fills (as fast as the source
        delivers — already close to display_fps for most sources) and the processor thread
        also reads, but never runs YOLO itself: it just re-composites the last known
        zones/boxes (`VisionPipeline.overlay_last_known`) onto whatever frame is newest right
        now. This is what lets the live view be smooth without raising target_fps, which
        would mean paying for YOLO that often instead of just for a cheap resize + JPEG encode.
        """
        frame_interval = 1.0 / self.display_fps if self.display_fps > 0 else 0.05
        next_display_at = time.monotonic()

        while not self._stop_event.is_set():
            wait_for = max(0.0, next_display_at - time.monotonic())
            self._stop_event.wait(wait_for)
            if self._stop_event.is_set():
                break

            with self._frame_lock:
                frame = self._latest_frame

            try:
                if frame is not None:
                    # Feed the raw frame to the pre-roll ring before any drawing/resizing, at
                    # this loop's fixed cadence. Isolated so a slow ring write cannot stall
                    # the live view.
                    if self.on_raw_frame_callback is not None:
                        try:
                            self.on_raw_frame_callback(self.camera_id, frame)
                        except Exception as exc:  # noqa: BLE001
                            logger.error(f"[{self.camera_id}] raw-frame callback failed: {exc}")

                    annotated = self.pipeline.overlay_last_known(frame, self.camera_id)
                    height, width = annotated.shape[:2]
                    if height > self.display_max_height:
                        scale = self.display_max_height / height
                        annotated = cv2.resize(
                            annotated,
                            (max(1, round(width * scale)), self.display_max_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    # H.264 first: it is what the grid actually renders, and doing it before
                    # the JPEG means a slow encode delays the fallback rather than the
                    # primary path.
                    if self._h264_encoder is not None:
                        self._h264_encoder.encode(annotated)

                    ok, buffer = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _DISPLAY_JPEG_QUALITY]
                    )
                    if ok:
                        self._latest_display_jpeg = buffer.tobytes()
            except Exception as e:
                # No reconnect logic here, unlike the reader thread — one bad encode must not
                # be allowed to silently end the display feed for this camera's whole session.
                logger.error(f"[{self.camera_id}] Error in display loop: {e}")
            finally:
                next_display_at = max(next_display_at + frame_interval, time.monotonic())
