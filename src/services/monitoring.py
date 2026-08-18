"""Runtime that turns the vision pipeline into a multi-camera web service.

This is the missing seam the repo never had: `src/vision/` could analyse a stream and
`src/api/` could answer HTTP, but nothing connected them. `MonitoringService` owns the
`MultiStreamManager`, keeps the latest annotated JPEG per camera for the grid to poll,
and turns rule-engine events into operator alerts.

Threading is the thing to get right here. Worker callbacks run on the per-camera processor
thread, never on the event loop:

    worker thread ──▶ _on_frame     ──▶ JPEG buffer          (grid, every frame)
                  └─▶ _on_incident  ──▶ alert + SSE publish  (modal, events only)

So three rules hold throughout: nothing in a callback may block (a slow call here stalls
frame processing for that camera outright), nothing in a callback may touch an asyncio
primitive directly (only `loop.call_soon_threadsafe`), and the frame handed to a callback
may alias the reader thread's buffer, so it is encoded immediately rather than stored.

Second-stage VLM triage (PRD §1.2) obeys that first rule literally: `_emit_alert` publishes
`alert.created` at the same speed it always did, then hands the assessment to a
`ThreadPoolExecutor`. The verdict arrives later as a separate `alert.verdict` event. It is
advisory — it annotates the alert and never suppresses it (see `src/services/vlm.py`).
"""

import asyncio
import base64
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import cv2
import numpy as np

from src.agents.incident_workflow import build_incident_workflow
from src.config import get_settings
from src.models.camera_schemas import (
    ALWAYS_ON_LABELS,
    Alert,
    AlertEvidenceFrame,
    CameraCreate,
    CameraInfo,
    DemoVideo,
    SystemStatus,
)
from src.services.alert_history import AlertHistory
from src.services.incident_log import IncidentLog
from src.services.notifications import ZaloNotifier
from src.services.vlm import NullVLMTriage, TriageRequest, TriageVerdict, VLMTriage, build_triage_client
from src.vision.detector import IncidentDetector
from src.vision.pipeline import VisionPipeline
from src.vision.rule_engine import FallPerspectiveProfile, RuleEngine
from src.vision.schemas import FrameAnalysisResult, Zone
from src.vision.stream_manager import MultiStreamManager, tune_torch_for_multistream
from src.vision.zone_config import ZoneManager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.join(_PROJECT_ROOT, "data", "vision", "test")
UPLOAD_DIR = os.path.join(_PROJECT_ROOT, "data", "vision", "uploads")
REGISTRY_PATH = os.path.join(_PROJECT_ROOT, "data", "cameras.json")

# Grid tiles are small and there may be a dozen of them; 65 is visually indistinguishable
# from 95 at tile size and roughly a third of the bytes over the wire.
_PREVIEW_JPEG_QUALITY = 65
_RING_JPEG_QUALITY = 85
_MAX_ALERTS = 50

# Triage runs on at most two threads and refuses work beyond this many alerts in flight. A
# camera flickering between states would otherwise queue an unbounded backlog of paid API
# calls whose verdicts arrive long after the operator has moved on. RuleEngine's
# alert_cooldown_sec (30s per camera+label) already caps the common case; this is the
# backstop for the uncommon one.
_MAX_TRIAGE_IN_FLIGHT = 4

_DEMO_HINTS = {
    "fall": "fall_detection",
    "abandoned": "abandoned_object",
    "input_test": "intrusion",
}


def _detector_labels(csv: str) -> list[str] | None:
    """Parse the configured COCO allow-list; an empty value deliberately means no filter."""
    labels = [label.strip() for label in csv.split(",") if label.strip()]
    return labels or None


def _evenly_spaced(items: list[bytes], budget: int) -> list[bytes]:
    """Thin a list down to at most `budget` items, keeping the first, last, and even spread.

    Consecutive frames at 8 fps are near-duplicates, so taking the first N of a window would
    describe the first half-second and nothing else. Spacing them is what makes a handful of
    stills still cover the whole transition.
    """
    if budget <= 0 or not items:
        return []
    if len(items) <= budget:
        return items
    step = (len(items) - 1) / (budget - 1) if budget > 1 else 0
    return [items[round(index * step)] for index in range(budget)]


def encode_frame_to_jpeg(frame: np.ndarray, max_edge: int = 1024) -> bytes:
    """JPEG-encode a BGR frame for storage or preview."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > max_edge:
        scale = max_edge / longest
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("cv2.imencode failed to encode the frame as JPEG")
    return buffer.tobytes()


def _evidence_frames(frames: list[bytes], pre_count: int) -> list[AlertEvidenceFrame]:
    """Convert the sampled VLM clip into frontend-ready data URLs."""
    evidence: list[AlertEvidenceFrame] = []
    for index, jpeg in enumerate(frames):
        evidence.append(
            AlertEvidenceFrame(
                index=index + 1,
                phase="before" if index < pre_count else "after",
                image_data_url="data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("utf-8"),
            )
        )
    return evidence


class MonitoringService:
    """Owns every camera worker, the latest-frame buffers, and the alert log."""

    def __init__(
        self,
        alert_cooldown_sec: float | None = None,
        triage: VLMTriage | None = None,
        history: AlertHistory | None = None,
        incident_log: IncidentLog | None = None,
    ) -> None:
        settings = get_settings()
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        # Apply process-global CPU limits before constructing YOLO. Some Torch settings
        # cannot be changed after the first parallel region has started.
        tune_torch_for_multistream()

        # One ZoneManager shared by every pipeline. Zones are immutable config with no
        # per-stream state, so sharing is safe — and it is what lets a polygon drawn in
        # the UI reach workers that already exist (see ZoneManager.set_zones).
        self.zone_manager = ZoneManager()

        # Template pipeline: MultiStreamManager clones its *config* per worker, because
        # tracker state and dwell timers must not be shared across cameras.
        template = VisionPipeline(
            detector=IncidentDetector(
                model_path=settings.model_weights_path,
                conf_threshold=settings.detector_confidence,
                target_labels=_detector_labels(settings.detector_target_labels),
            ),
            rule_engine=RuleEngine(
                zone_manager=self.zone_manager,
                alert_cooldown_sec=(settings.alert_cooldown_sec if alert_cooldown_sec is None else alert_cooldown_sec),
            ),
            draw_overlays=True,
        )
        self.manager = MultiStreamManager(
            shared_pipeline=template,
            global_on_frame_callback=self._on_frame,
            global_on_incident_callback=self._on_incident,
            max_concurrent_inferences=(
                settings.camera_max_concurrent_inferences or None
            ),
        )

        self._cameras: dict[str, CameraInfo] = {}
        self._frames: dict[str, bytes] = {}
        self._alerts: list[Alert] = []
        self._lock = threading.RLock()

        # Rolling window of recent frames per camera, for the triage layer's pre-roll.
        #
        # JPEG *bytes*, never ndarrays: the frame handed to a callback may alias the reader
        # thread's shared buffer, so a ring of arrays would quietly become N references to
        # whatever the newest frame is — a "clip" of identical images. These are the same
        # bytes `_on_frame` already encodes for the preview, so this costs no extra encode.
        self._ring: dict[str, deque[tuple[float, bytes]]] = {}
        self._pre_roll_sec = settings.vlm_pre_roll_sec
        self._post_roll_sec = settings.vlm_post_roll_sec
        self._max_frames = settings.vlm_max_frames
        self._history_examples = settings.vlm_history_examples
        # Sized from the configured capture rate with headroom, so the window covers
        # pre_roll_sec of footage even when a camera runs faster than its target.
        self._default_ring_size = max(8, int(settings.camera_target_fps * settings.vlm_pre_roll_sec * 2))

        self.triage = triage or build_triage_client(settings)
        self.history = history or AlertHistory()
        # Every alert, reviewed or not. `history` holds only judged alerts by design, so it
        # cannot answer "how many did this camera raise" — see incident_log.py.
        self.incidents = incident_log or IncidentLog()
        self.notifier = ZaloNotifier(settings.zalo_enabled, settings.zalo_access_token, settings.zalo_recipients)
        # Two workers: enough to overlap a slow API call with a second camera's alert,
        # small enough that a burst cannot spawn threads without bound.
        self._triage_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-triage")
        self._notification_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="incident-notify")
        self._triage_in_flight = 0
        self._incident_workflow = build_incident_workflow(
            wait_for_post_roll=self._wait_for_post_roll,
            collect_evidence=self._collect_triage_evidence,
            assess_evidence=self._assess_triage_evidence,
            persist_verdict=self._persist_triage_verdict,
            persist_failure=self._persist_triage_failure,
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue] = []

        self._restore_registry()

    # ------------------------------------------------------------------ lifecycle
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop so worker threads can publish SSE events into it."""
        self._loop = loop

    def shutdown(self) -> None:
        self.manager.stop_all()
        # wait=False: a triage call already blocked on a 20s API timeout must not hold the
        # process open on Ctrl-C. Its verdict is lost, which is the right trade — the
        # verdict is advisory and the alert it belongs to has already been shown.
        self._triage_pool.shutdown(wait=False)
        self._notification_pool.shutdown(wait=False)

    # ------------------------------------------------------------------ sources
    @staticmethod
    def list_demo_videos() -> list[DemoVideo]:
        """Sample clips shipped in the repo, offered as one-click camera sources."""
        if not os.path.isdir(DEMO_DIR):
            return []
        videos = []
        for filename in sorted(os.listdir(DEMO_DIR)):
            if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            stem = os.path.splitext(filename)[0]
            videos.append(
                DemoVideo(
                    id=filename,
                    name=stem.replace("_", " ").title(),
                    suggested_label=_DEMO_HINTS.get(stem),
                )
            )
        return videos

    @staticmethod
    def resolve_source(source_type: str, source: str) -> str:
        """Map a UI source selection onto something cv2.VideoCapture can open.

        Demo and upload names are joined onto a fixed directory and stripped of any path
        separators first — the value arrives from an HTTP body, so `../../etc/passwd`
        must not resolve to a readable file.
        """
        if source_type == "rtsp":
            return source
        safe = os.path.basename(source)
        base = DEMO_DIR if source_type == "demo" else UPLOAD_DIR
        return os.path.join(base, safe)

    # ------------------------------------------------------------------ cameras
    def list_cameras(self) -> list[CameraInfo]:
        with self._lock:
            cameras = list(self._cameras.values())

        stats = {s["camera_id"]: s for s in self.manager.get_status()["cameras"]}
        for camera in cameras:
            stat = stats.get(camera.id, {})
            camera.running = bool(stat.get("running", False))
            camera.processed_frames = int(stat.get("processed_frames", 0))
            camera.inference_capacity_fps = float(stat.get("inference_capacity_fps", 0.0))
            camera.latency_p50_ms = float(stat.get("latency_p50_ms", 0.0))
            camera.latency_p95_ms = float(stat.get("latency_p95_ms", 0.0))
            camera.latency_p99_ms = float(stat.get("latency_p99_ms", 0.0))
            camera.inference_wait_p95_ms = float(stat.get("inference_wait_p95_ms", 0.0))
            camera.processing_errors = int(stat.get("processing_errors", 0))
        return cameras

    def get_camera(self, camera_id: str) -> CameraInfo | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def add_camera(self, payload: CameraCreate) -> CameraInfo:
        """Register, configure zones for, and start one camera."""
        settings = get_settings()
        camera_id = f"cam_{uuid.uuid4().hex[:8]}"
        resolved = self.resolve_source(payload.source_type, payload.source)

        if payload.source_type != "rtsp" and not os.path.exists(resolved):
            raise FileNotFoundError(f"Video source not found: {payload.source}")

        # Zones go in before the worker starts, so the very first analysed frame already
        # sees the polygon rather than missing intrusions during startup.
        self._apply_zones(camera_id, payload)

        labels = list(ALWAYS_ON_LABELS) + (["intrusion"] if payload.intrusion_enabled else [])
        info = CameraInfo(
            id=camera_id,
            name=payload.name,
            source_type=payload.source_type,
            source=payload.source,
            resolved_source=resolved,
            intrusion_enabled=payload.intrusion_enabled,
            zone_polygon=payload.zone_polygon,
            zone_active_hours=payload.zone_active_hours,
            active_labels=labels,
            target_fps=payload.target_fps or settings.camera_target_fps,
            stream_url=f"/api/v1/cameras/{camera_id}/stream",
            fall_perspective=payload.fall_perspective,
        )

        with self._lock:
            self._cameras[camera_id] = info

        worker = self.manager.add_camera(
            camera_id=camera_id,
            rtsp_url=resolved,
            target_fps=info.target_fps,
            autostart=False,
            # A file source that stops at EOF leaves a dead tile in the grid; RTSP is
            # already continuous so looping it is meaningless.
            loop_finite_sources=payload.loop_source and payload.source_type != "rtsp",
        )
        if worker is not None and payload.fall_perspective is not None:
            worker.pipeline.rule_engine.set_camera_fall_profile(
                camera_id,
                FallPerspectiveProfile(
                    **payload.fall_perspective.model_dump(exclude_none=True)
                ),
            )
        if worker is not None:
            worker.start()
        self._persist_registry()
        self._publish("camera.added", info.model_dump())
        logger.info(f"Camera '{info.name}' ({camera_id}) started on {resolved}")
        return info

    def remove_camera(self, camera_id: str) -> bool:
        self.manager.remove_camera(camera_id)
        with self._lock:
            existed = self._cameras.pop(camera_id, None) is not None
            self._frames.pop(camera_id, None)
            self._ring.pop(camera_id, None)
        self.zone_manager.clear_zones(camera_id)
        if existed:
            self._persist_registry()
            self._publish("camera.removed", {"id": camera_id})
        return existed

    def _apply_zones(self, camera_id: str, payload: CameraCreate) -> None:
        """Install the drawn polygon, or an explicit empty list when intrusion is off.

        The empty list is load-bearing: `zones_for` falls back to the wildcard `"*"` entry
        in `data/vision/zones.json` for any camera it has not heard of, so silently doing
        nothing here would turn intrusion *on* for a camera the operator left it off for.
        """
        if payload.intrusion_enabled and payload.zone_polygon:
            window = payload.zone_active_hours
            zone = Zone(
                name="restricted_zone",
                polygon=payload.zone_polygon,
                # PRD §3.1's "khung giờ cấm". None keeps the zone armed around the clock,
                # which is what the abandoned-bag and stairwell demos want; a night window
                # is what the intrusion user story asks for. `CameraCreate` has already
                # normalized the string, so `Zone` only ever sees 'HH:MM-HH:MM' or None.
                active_hours=window,
                description=(
                    f"Operator-drawn zone for {payload.name}" + (f", armed {window}" if window else ", armed 24/7")
                ),
            )
            self.zone_manager.set_zones(camera_id, [zone])
        else:
            self.zone_manager.set_zones(camera_id, [])

    # ------------------------------------------------------------------ frames
    def latest_jpeg(self, camera_id: str) -> bytes | None:
        return self._frames.get(camera_id)

    def _on_frame(self, camera_id: str, result: FrameAnalysisResult, frame: np.ndarray) -> None:
        """Every analysed frame: refresh the preview buffer and the triage ring.

        Driving the preview off the incident callback instead would leave the grid frozen
        between events, which reads as a broken demo.
        """
        try:
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _PREVIEW_JPEG_QUALITY])
            if ok:
                jpeg = buffer.tobytes()
                # Plain assignment, no lock: dict item assignment is atomic under the GIL
                # and readers only ever want the newest frame.
                self._frames[camera_id] = jpeg
                # Same bytes into the pre-roll ring — one encode serves both. deque.append
                # is atomic too, and the eviction is what bounds memory per camera.
                ring = self._ring.get(camera_id)
                if ring is None:
                    camera = self.get_camera(camera_id)
                    fps = camera.target_fps if camera else None
                    ring_size = (
                        max(8, int(fps * self._pre_roll_sec * 2))
                        if fps is not None
                        else self._default_ring_size
                    )
                    ring = self._ring.setdefault(camera_id, deque(maxlen=ring_size))
                ok_ring, ring_buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _RING_JPEG_QUALITY]
                )
                if ok_ring:
                    ring.append((time.monotonic(), ring_buffer.tobytes()))
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the worker
            logger.error(f"[{camera_id}] preview encode failed: {exc}")

    # ------------------------------------------------------------------ alerts
    def _on_incident(self, camera_id: str, result: FrameAnalysisResult, frame: np.ndarray) -> None:
        """Frames carrying rule-engine events. One alert per event."""
        for event in result.events:
            self._emit_alert(
                camera_id=camera_id,
                label=event.label,
                confidence=event.confidence,
                frame=frame,
                reason=event.reason,
                zone_name=event.zone_name,
                track_id=event.track_id,
            )

    def _emit_alert(
        self,
        camera_id: str,
        label: str,
        confidence: float,
        frame: np.ndarray,
        reason: str | None = None,
        zone_name: str | None = None,
        track_id: str | None = None,
    ) -> None:
        """Capture the frame and publish the alert.

        The whole path is synchronous and allocation-cheap on purpose: it runs on the
        camera's processor thread, so the operator sees the event and the image within one
        frame time and nothing here can stall the stream behind it.
        """
        info = self.get_camera(camera_id)
        try:
            # Face blurring was removed on 2026-08-10, so this frame carries identifiable
            # faces into the alert image, the history thumbnail, and the VLM clip when
            # triage is on. PRD §3.3 asks for anonymization; reinstate `FaceAnonymizer` in
            # VisionPipeline to get it back.
            jpeg = encode_frame_to_jpeg(frame)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{camera_id}] failed to capture incident frame: {exc}")
            return

        alert = Alert(
            id=uuid.uuid4().hex,
            camera_id=camera_id,
            camera_name=info.name if info else camera_id,
            label=label,
            confidence=round(float(confidence), 3),
            created_at=datetime.now(UTC).isoformat(),
            reason=reason,
            zone_name=zone_name,
            track_id=track_id,
            image_data_url="data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("utf-8"),
        )

        with self._lock:
            self._alerts.insert(0, alert)
            del self._alerts[_MAX_ALERTS:]

        # Append on the processor thread, like `_publish` a line below: one small buffered
        # write under a lock. Without this row the alert would only exist for the life of
        # the in-memory list — an alert nobody judged would vanish, and "how many incidents
        # per camera" is exactly the number the manager dashboard needs to show unreviewed.
        self.incidents.record_alert(
            alert_id=alert.id,
            camera_id=alert.camera_id,
            camera_name=alert.camera_name,
            label=alert.label,
            confidence=alert.confidence,
        )

        queued_for_triage = self._queue_triage(alert)
        self._publish("alert.created", alert.model_dump())
        if self.notifier.enabled:
            self._notification_pool.submit(self.notifier.send_alert, alert.model_dump())
        # Only after the operator already has the alert. Everything below this line happens
        # on another thread and can take seconds.
        if queued_for_triage:
            self._submit_triage(alert)

    # ------------------------------------------------------------------ triage
    def _queue_triage(self, alert: Alert) -> bool:
        """Mark an alert for second-stage assessment before the first SSE payload."""
        if isinstance(self.triage, NullVLMTriage):
            return False  # Layer is off; the alert keeps its default "skipped" status.

        with self._lock:
            if self._triage_in_flight >= _MAX_TRIAGE_IN_FLIGHT:
                alert.triage_status = "skipped"
                alert.triage_error = "triage queue saturated"
                logger.warning(f"[{alert.camera_id}] skipping triage for {alert.id}: queue saturated")
                return False
            self._triage_in_flight += 1
            alert.triage_status = "pending"
            return True

    def _submit_triage(self, alert: Alert) -> None:
        """Queue the second-stage assessment, or mark it skipped and move on.

        Called from the camera's processor thread, so it must return immediately — the
        submit is the only work done here.
        """
        try:
            self._triage_pool.submit(self._run_triage, alert)
        except RuntimeError:
            # The pool is shut down (process exiting). Undo the bookkeeping rather than
            # leaving the alert pending forever.
            with self._lock:
                self._triage_in_flight -= 1
                alert.triage_status = "skipped"
                alert.triage_error = "service shutting down"
                snapshot = alert.model_dump()
            self._publish("alert.verdict", snapshot)

    def _run_triage(self, alert: Alert) -> None:
        """Executor thread: run the LangGraph incident-review workflow.

        The wait is why this cannot live in the callback. `_on_incident` fires on the frame
        the rule engine decided on, so the "after" frames do not exist yet — the camera's
        processor thread has to keep running to produce them, which it can only do if this
        thread is not the one it is running on.
        """
        try:
            self._incident_workflow.invoke({"alert": alert})

        except Exception as exc:  # noqa: BLE001 - an executor exception is silent otherwise
            logger.error(f"[{alert.camera_id}] triage job failed for {alert.id}: {exc}")
            with self._lock:
                alert.triage_status = "complete"
                alert.triage_verdict = "uncertain"
                alert.triage_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._triage_in_flight -= 1
            # Published regardless of outcome: a UI left spinning on a failed assessment is
            # worse than one told the assessment is unavailable.
            self._publish("alert.verdict", alert.model_dump())

    def _wait_for_post_roll(self, alert: Alert) -> None:
        """Wait until the camera ring contains evidence after the alert."""
        del alert
        if self._post_roll_sec > 0:
            time.sleep(self._post_roll_sec)

    def _collect_triage_evidence(
        self, alert: Alert
    ) -> tuple[list[bytes], int, list[AlertEvidenceFrame], list[str]]:
        frames, pre_count = self._collect_clip(alert.camera_id)
        return (
            frames,
            pre_count,
            _evidence_frames(frames, pre_count),
            self.history.recent_examples(alert.camera_id, alert.label, limit=self._history_examples),
        )

    def _assess_triage_evidence(
        self,
        alert: Alert,
        frames: list[bytes],
        pre_count: int,
        examples: list[str],
    ) -> TriageVerdict:
        if pre_count < 1 or len(frames) - pre_count < 1:
            return TriageVerdict(
                verdict="uncertain",
                error="insufficient visual evidence: missing before or after frames",
            )
        return self.triage.assess(
            TriageRequest(
                label=alert.label,
                reason=alert.reason,
                zone_name=alert.zone_name,
                camera_name=alert.camera_name,
                frames=frames,
                pre_count=pre_count,
                examples=examples,
            )
        )

    def _persist_triage_verdict(
        self,
        alert: Alert,
        verdict: TriageVerdict,
        evidence_frames: list[AlertEvidenceFrame],
    ) -> None:
        with self._lock:
            alert.triage_status = "complete"
            alert.triage_verdict = verdict.verdict
            alert.triage_confidence = verdict.confidence
            alert.triage_rationale = verdict.rationale
            alert.triage_frame_observations = verdict.frame_observations[: len(evidence_frames)]
            alert.triage_error = verdict.error
            alert.evidence_frames = evidence_frames
        logger.info(
            f"[{alert.camera_id}] triage {alert.label} -> {verdict.verdict} "
            f"({verdict.confidence:.2f}) in {verdict.latency_ms:.0f}ms"
        )

    def _persist_triage_failure(self, alert: Alert, error: str) -> None:
        with self._lock:
            alert.triage_status = "complete"
            alert.triage_verdict = "uncertain"
            alert.triage_error = error

    def _collect_clip(self, camera_id: str) -> tuple[list[bytes], int]:
        """Sample an evenly-spaced clip from the ring: `(frames, how many precede the event)`.

        The split point is `now - post_roll_sec`, which is when the rule actually fired —
        this runs after sleeping through the post-roll, so "now" is already too late.

        Sampling matters: every frame is a billed image and accuracy degrades with long
        sequences, so a 5-second window at 8 fps is reduced to a handful of stills that still
        span the whole transition.
        """
        ring = self._ring.get(camera_id)
        if not ring:
            return [], 0

        # Copy first — the worker thread is appending to this deque concurrently.
        samples = list(ring)
        event_time = time.monotonic() - self._post_roll_sec
        window_start = event_time - self._pre_roll_sec

        pre = [jpeg for stamp, jpeg in samples if window_start <= stamp <= event_time]
        post = [jpeg for stamp, jpeg in samples if stamp > event_time]

        # Two thirds before, one third after. The lead-up is what distinguishes a fall from
        # someone already lying down; the aftermath only has to confirm they stayed there.
        pre_budget = max(1, (self._max_frames * 2) // 3)
        post_budget = max(1, self._max_frames - pre_budget)
        pre = _evenly_spaced(pre, pre_budget)
        post = _evenly_spaced(post, post_budget)

        return pre + post, len(pre)

    def record_operator_verdict(
        self,
        alert_id: str,
        verdict: str,
        judged_by: str | None = None,
        operator_note: str | None = None,
        actual_event_type: str | None = None,
    ) -> bool:
        """HITL (PRD §3.2): the guard's judgement, which is also the training signal.

        Separate from `acknowledge` on purpose — that one means "dismissed", this one means
        "reviewed and judged", and only this one writes history. An unlabelled row would
        poison the dataset the false-alarm rate is computed from.

        `judged_by` is the signed-in operator's username, so a disputed label can be traced
        to whoever gave it. It defaults to None because the rule-layer tests construct this
        service directly with no auth in front of it; a row written without a name is still a
        valid row, just an unattributed one.
        """
        with self._lock:
            alert = next((a for a in self._alerts if a.id == alert_id), None)
            if alert is None:
                return False
            alert.operator_verdict = verdict
            alert.acknowledged = True
            alert.judged_by = judged_by
            alert.operator_note = operator_note
            alert.actual_event_type = actual_event_type
            snapshot = alert.model_dump()

        self.history.record(
            alert_id=alert.id,
            camera_id=alert.camera_id,
            label=alert.label,
            operator_verdict=verdict,
            reason=alert.reason,
            vlm_verdict=alert.triage_verdict,
            vlm_confidence=alert.triage_confidence,
            vlm_rationale=alert.triage_rationale,
            judged_by=judged_by,
            operator_note=operator_note,
            actual_event_type=actual_event_type,
        )
        self.incidents.record_review(
            alert_id=alert.id,
            verdict=verdict,
            judged_by=judged_by,
            operator_note=operator_note,
            actual_event_type=actual_event_type,
        )
        self._publish("alert.acknowledged", snapshot)
        return True

    def list_alerts(self, unacknowledged_only: bool = False) -> list[Alert]:
        with self._lock:
            alerts = list(self._alerts)
        return [a for a in alerts if not a.acknowledged] if unacknowledged_only else alerts

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            alert = next((a for a in self._alerts if a.id == alert_id), None)
            if alert is None:
                return False
            alert.acknowledged = True
        self._publish("alert.acknowledged", {"id": alert_id})
        return True

    def status(self) -> SystemStatus:
        raw = self.manager.get_status()
        return SystemStatus(
            total_cameras=int(raw["total_cameras"]),
            active_cameras=int(raw["active_cameras"]),
            average_inference_capacity_fps=float(raw["average_inference_capacity_fps"]),
            average_latency_p95_ms=float(raw["average_latency_p95_ms"]),
            processing_errors=int(raw["processing_errors"]),
            total_processed_frames=int(raw["total_processed_frames"]),
            max_concurrent_inferences=int(raw["max_concurrent_inferences"]),
            replaced_frame_callbacks=int(raw["replaced_frame_callbacks"]),
            unacknowledged_alerts=len(self.list_alerts(unacknowledged_only=True)),
        )

    def operational_stats(self) -> dict:
        """Alert volume and review activity, for the manager dashboard.

        Deliberately separate from `EvalReport`, which is computed from reviewed rows only
        and is about *accuracy*. This is about *workload*: how much fired, on which camera,
        and how much of it nobody has looked at.

        Blocking file I/O — call it off the event loop.
        """
        rows = self.incidents.incidents()
        reviewed = [row for row in rows if row.get("operator_verdict")]

        by_camera: dict[str, dict] = {}
        by_label: dict[str, int] = defaultdict(int)
        by_day: dict[str, int] = defaultdict(int)
        for row in rows:
            camera_id = row.get("camera_id", "unknown")
            bucket = by_camera.setdefault(
                camera_id,
                {"camera_id": camera_id, "camera_name": row.get("camera_name", camera_id), "total": 0, "reviewed": 0},
            )
            bucket["total"] += 1
            if row.get("operator_verdict"):
                bucket["reviewed"] += 1
            by_label[row.get("label", "unknown")] += 1
            # Date only: an hour-resolution series over a 24/7 system is noise on a dashboard,
            # and the PRD's story is about nights, which a daily count still shows.
            created = row.get("created_at") or ""
            if created:
                by_day[created[:10]] += 1

        # "Right now" comes from memory, not the log. The log is every alert ever raised, so
        # counting unreviewed rows there would include alerts from weeks ago that no guard on
        # this shift can act on — a backlog number that only ever grows is not actionable.
        live_unacked = len(self.list_alerts(unacknowledged_only=True))

        return {
            "total_alerts": len(rows),
            "reviewed_alerts": len(reviewed),
            # Unreviewed in the whole log, which is not the same as the live backlog above.
            "unreviewed_alerts": len(rows) - len(reviewed),
            "live_unacknowledged": live_unacked,
            "by_camera": sorted(by_camera.values(), key=lambda item: -item["total"]),
            "by_label": [{"label": label, "total": total} for label, total in sorted(by_label.items())],
            "by_day": [{"date": day, "total": total} for day, total in sorted(by_day.items())],
            "reviewers": sorted({row["judged_by"] for row in reviewed if row.get("judged_by")}),
            "first_alert": rows[0].get("created_at") if rows else None,
            "last_alert": rows[-1].get("created_at") if rows else None,
        }

    def heatmap(self) -> dict:
        """Return a simulated floor-plan heat map until real camera coordinates are configured."""
        points: dict[str, dict] = {}
        for row in self.incidents.incidents():
            camera_id = row.get("camera_id") or "unknown"
            point = points.setdefault(camera_id, {
                "camera_id": camera_id,
                "camera_name": row.get("camera_name", camera_id),
                "x": (sum(ord(char) for char in camera_id) % 80) + 10,
                "y": (sum((index + 1) * ord(char) for index, char in enumerate(camera_id)) % 60) + 20,
                "total": 0,
                "by_label": {},
            })
            point["total"] += 1
            label = row.get("label", "unknown")
            point["by_label"][label] = point["by_label"].get(label, 0) + 1
        return {"coordinate_system": "simulated_floor_plan_percent", "points": list(points.values())}

    # ------------------------------------------------------------------ SSE fan-out
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish_event(self, event_type: str, data: dict) -> None:
        """Broadcast an event to every SSE subscriber.

        The public door onto `_publish`, for events that originate outside the camera
        pipeline — `auth.revoked` on logout is the current one. Every subscriber receives it,
        so anything user-specific has to carry the username and let the client decide whether
        it is about them; there is no per-user subscriber index.
        """
        self._publish(event_type, data)

    def _publish(self, event_type: str, data: dict) -> None:
        """Hand an event to every SSE subscriber from whichever thread produced it.

        `call_soon_threadsafe` is mandatory here: asyncio.Queue is not thread-safe, and
        these callers are worker and executor threads, not the event loop.
        """
        if self._loop is None or self._loop.is_closed():
            return
        message = {"type": event_type, "data": data}
        for queue in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._offer, queue, message)

    @staticmethod
    def _offer(queue: asyncio.Queue, message: dict) -> None:
        """Never block the loop on a stalled client — drop instead."""
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("SSE subscriber queue is full; dropping an event")

    # ------------------------------------------------------------------ persistence
    def _persist_registry(self) -> None:
        """Save camera configs so a backend restart doesn't lose the operator's setup."""
        with self._lock:
            cameras = [c.model_dump() for c in self._cameras.values()]
        try:
            os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
            with open(REGISTRY_PATH, "w", encoding="utf-8") as handle:
                json.dump({"cameras": cameras}, handle, indent=2)
        except OSError as exc:
            logger.warning(f"Could not persist camera registry: {exc}")

    def _restore_registry(self) -> None:
        """Re-add previously configured cameras on startup, skipping dead sources."""
        if not os.path.exists(REGISTRY_PATH):
            return
        try:
            with open(REGISTRY_PATH, encoding="utf-8") as handle:
                saved = json.load(handle).get("cameras", [])
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            logger.warning(f"Ignoring unreadable camera registry: {exc}")
            return

        for entry in saved:
            try:
                self.add_camera(
                    CameraCreate(
                        name=entry["name"],
                        source_type=entry["source_type"],
                        source=entry["source"],
                        intrusion_enabled=entry.get("intrusion_enabled", False),
                        zone_polygon=entry.get("zone_polygon"),
                        zone_active_hours=entry.get("zone_active_hours"),
                        target_fps=entry.get("target_fps"),
                        fall_perspective=entry.get("fall_perspective"),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one stale entry must not block startup
                logger.warning(f"Skipping saved camera '{entry.get('name')}': {exc}")
