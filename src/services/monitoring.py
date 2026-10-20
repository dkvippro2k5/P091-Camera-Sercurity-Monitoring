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
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import NamedTuple, TypeVar

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
from src.services.camera_registry import create_camera_registry
from src.services.embedding_store import create_embedding_store
from src.services.embeddings import build_embedder
from src.services.event_bus import build_event_relay
from src.services.evidence_store import EvidenceStore
from src.services.incident_log import IncidentLog
from src.services.notifications import NotificationWorker, TelegramNotifier, should_notify
from src.services.vlm import NullVLMTriage, TriageRequest, TriageVerdict, VLMTriage, build_triage_client
from src.vision.appearance import AppearanceEncoder
from src.vision.detector import IncidentDetector
from src.vision.fall_pose import FallPoseEstimator
from src.vision.pipeline import VisionPipeline
from src.vision.rule_engine import FallPerspectiveProfile, RuleEngine
from src.vision.schemas import FrameAnalysisResult, Zone
from src.vision.static_objects import StaticObjectDetector
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
_RING_JPEG_QUALITY = 80
# The ring feeds the VLM, which downsamples to 1024px on its longest edge anyway, so
# storing anything larger just burns CPU on every encode — and the ring encode now runs
# on the display loop, several times a second per camera. Encoding a native 4K frame is
# ~40ms; at 4 cameras that alone saturates the box. Cap it here.
_RING_MAX_EDGE = 1280
# The ring does not need display_fps; a clip is a handful of stills spanning a few seconds.
# Feeding it a few times a second is plenty and leaves the CPU for inference.
_RING_FEED_FPS = 4.0
_MAX_ALERTS = 50
# How much wider/taller than the raw person bbox a triage crop pads on each side. A cut
# tight to the box loses exactly the context a fall verdict needs most — the floor the
# person is on or near — and stitching bugs across pre/post-roll (track_id changes) are
# more forgiving with a wider box. Same padding fraction on both axes: bboxes vary in
# aspect ratio (a fallen person is wide, a standing one is tall) and a wider ratio in one
# axis than the other would arbitrarily favor one orientation.
_TRIAGE_CROP_PADDING = 0.4
# Below this, either dimension after padding, a crop is not worth sending: the model would
# be reasoning from a thumbnail instead of a clip. Falls back to the full frame instead of
# risking an uninformative crop that reads as evidence when it is not.
_TRIAGE_CROP_MIN_EDGE_PX = 64

# Triage runs on a small thread pool and refuses work beyond this many alerts in flight. A
# camera flickering between states would otherwise queue an unbounded backlog of paid API
# calls whose verdicts arrive long after the operator has moved on. RuleEngine's
# alert_cooldown_sec (30s per track and label, with fall alerts merged per incident) already
# caps the common case; this is the backstop for the uncommon one. Sized for several
# cameras alerting at once: an assess()
# call is a network round trip (I/O-bound, not CPU), so more concurrency here costs almost
# nothing and stops a real incident on camera 4 being dropped as "queue saturated" while
# cameras 1-3 are mid-call.
_TRIAGE_POOL_WORKERS = 4
_MAX_TRIAGE_IN_FLIGHT = 8

_DEMO_HINTS = {
    "fall": "fall_detection",
    "abandoned": "abandoned_object",
    "input_test": "intrusion",
}


def _detector_labels(csv: str) -> list[str] | None:
    """Parse the configured COCO allow-list; an empty value deliberately means no filter."""
    labels = [label.strip() for label in csv.split(",") if label.strip()]
    return labels or None


class RingFrame(NamedTuple):
    """One still in a camera's pre-roll ring.

    `tracks` maps `track_id` -> normalized (xmin, ymin, xmax, ymax) for every person YOLO
    was tracking at the moment this frame was captured — see `VisionPipeline.last_tracks`
    for why normalized. Empty before the first inference, or for a camera with no VLM
    triage in play; `_collect_clip` falls back to the untouched frame whenever a lookup
    into this dict comes up empty, so an empty `tracks` degrades to the pre-crop behavior
    rather than failing.
    """

    timestamp: float
    jpeg: bytes
    tracks: dict[str, tuple[float, float, float, float]]


def _crop_track(
    jpeg: bytes, bbox: tuple[float, float, float, float], padding: float = _TRIAGE_CROP_PADDING
) -> bytes | None:
    """Crop `jpeg` to `bbox` (normalized xmin/ymin/xmax/ymax) padded by `padding` per side.

    Returns None rather than raising when the crop is not usable — a bad JPEG, a
    degenerate box, or a result too small to carry information (`_TRIAGE_CROP_MIN_EDGE_PX`)
    — so the caller can fall back to the uncropped frame instead of losing the frame
    entirely over one bad box.
    """
    array = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    height, width = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    box_w = max(0.0, xmax - xmin)
    box_h = max(0.0, ymax - ymin)
    if box_w <= 0 or box_h <= 0:
        return None
    pad_x, pad_y = box_w * padding, box_h * padding
    x0 = max(0, int((xmin - pad_x) * width))
    y0 = max(0, int((ymin - pad_y) * height))
    x1 = min(width, int((xmax + pad_x) * width))
    y1 = min(height, int((ymax + pad_y) * height))
    if x1 - x0 < _TRIAGE_CROP_MIN_EDGE_PX or y1 - y0 < _TRIAGE_CROP_MIN_EDGE_PX:
        return None
    cropped = frame[y0:y1, x0:x1]
    ok, buffer = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), _RING_JPEG_QUALITY])
    return buffer.tobytes() if ok else None


_T = TypeVar("_T")


def _evenly_spaced(items: list[_T], budget: int) -> list[_T]:
    """Thin a list down to at most `budget` items, keeping the first, last, and even spread.

    Consecutive frames at 8 fps are near-duplicates, so taking the first N of a window would
    describe the first half-second and nothing else. Spacing them is what makes a handful of
    stills still cover the whole transition.

    Generic over the item type (JPEG bytes or a `RingFrame`) so `_collect_clip` can thin the
    ring's `RingFrame` entries — keeping each frame's `tracks` alongside its JPEG — before
    deciding per-frame whether to crop, rather than thinning post-encode bytes that have
    already lost that information.
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


def _evidence_frames(
    frames: list[bytes], pre_count: int, alert_id: str, store: EvidenceStore
) -> list[AlertEvidenceFrame]:
    """Persist the sampled VLM clip and describe it by URL rather than by value.

    Runs on a triage worker thread, never on the camera's processor thread, so the extra
    writes here cannot stall detection. A frame that fails to store is dropped from the strip
    rather than aborting the whole set: a partial evidence strip is still useful to the guard,
    and triage itself has already finished by the time this is called.
    """
    evidence: list[AlertEvidenceFrame] = []
    for index, jpeg in enumerate(frames):
        # Namespaced by alert so one alert's frames can never collide with another's, and so
        # they sort next to the alert's own key in the bucket / directory listing.
        evidence_id = f"{alert_id}-f{index + 1}"
        try:
            url = store.put(evidence_id, jpeg)
        except Exception:  # noqa: BLE001 - a storage failure must not lose the verdict itself
            logger.exception(f"Could not store evidence frame {evidence_id}")
            continue
        evidence.append(
            AlertEvidenceFrame(
                index=index + 1,
                phase="before" if index < pre_count else "after",
                image_url=url,
            )
        )
    return evidence


_ZONE_NAME_PATTERN = re.compile(r"restricted zone '[^']*'")


def embedding_text(alert: Alert) -> str:
    """The text compared by the similar-incident lookup, from every source that has one.

    Two problems with embedding `alert.reason` alone, which is what this replaced.

    The first is that the reason is a rule-engine string, and for intrusion it embeds
    `zone_name` — an operator-chosen label like 'Kho hàng chính'. In a sentence of fifteen
    words a proper noun that specific dominates the vector, so the lookup matched *same
    place* rather than *same behaviour*: two intrusions on one camera looked alike however
    differently they unfolded. Substituting a generic phrase keeps the sentence's shape and
    drops the landmark. Fall and abandoned-object reasons carry no zone name, so this is a
    no-op for them.

    The second is that the VLM's own reading of the clip — a sentence describing what
    happened, plus a note per frame — was computed, shown to the guard, and then left out of
    the comparison entirely. That was the richest description in the system. Including it is
    what lets the lookup agree or disagree with the assessment on screen.

    `operator_note` is deliberately absent. This same function builds both the stored vector
    and the query vector, and at query time the guard has not written their note yet — it is
    the field they are about to fill in. Adding it to the stored side alone would mean the
    two sides describe different things, which is worse than leaving it out of both.

    Sections are omitted rather than emptied when a source is missing, so a deployment with
    VLM disabled produces exactly the pre-existing text and nothing about its results changes.
    """
    sections = []
    if alert.reason:
        sections.append(f"[rule] {_ZONE_NAME_PATTERN.sub('a restricted zone', alert.reason)}")
    if alert.triage_rationale:
        sections.append(f"[ai] {alert.triage_rationale}")
    if alert.triage_frame_observations:
        sections.append(f"[frames] {' '.join(alert.triage_frame_observations)}")
    return "\n".join(sections)


class MonitoringService:
    """Owns every camera worker, the latest-frame buffers, and the alert log."""

    def __init__(
        self,
        alert_cooldown_sec: float | None = None,
        triage: VLMTriage | None = None,
        history: AlertHistory | None = None,
        incident_log: IncidentLog | None = None,
        embedding_store=None,
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
        # tracker state and dwell timers must not be shared across cameras. Retained on the
        # instance so a test (and the abandonment tunables above) can read back what every
        # worker's rule engine was built with.
        self._pipeline_template = VisionPipeline(
            detector=IncidentDetector(
                model_path=settings.model_weights_path,
                conf_threshold=settings.detector_confidence,
                target_labels=_detector_labels(settings.detector_target_labels),
            ),
            rule_engine=RuleEngine(
                zone_manager=self.zone_manager,
                alert_cooldown_sec=(settings.alert_cooldown_sec if alert_cooldown_sec is None else alert_cooldown_sec),
                abandon_sec=settings.abandon_sec,
                owner_radius_ratio=settings.owner_radius_ratio,
                stationary_tolerance_ratio=settings.stationary_tolerance_ratio,
                abandon_require_owner=settings.abandon_require_owner,
                owner_bind_sec=settings.owner_bind_sec,
                owner_match_similarity=settings.owner_match_similarity,
            ),
            draw_overlays=True,
            fall_pose_estimator=(
                FallPoseEstimator(settings.fall_pose_weights_path) if settings.fall_pose_weights_path else None
            ),
            appearance_encoder=AppearanceEncoder() if settings.owner_appearance_enabled else None,
            static_object_detector=StaticObjectDetector() if settings.abandon_background_enabled else None,
        )
        self.manager = MultiStreamManager(
            shared_pipeline=self._pipeline_template,
            global_on_frame_callback=self._on_frame,
            global_on_raw_frame_callback=self._on_raw_frame,
            global_on_incident_callback=self._on_incident,
            max_concurrent_inferences=(
                settings.camera_max_concurrent_inferences or None
            ),
            capture_max_height=settings.camera_capture_max_height,
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
        # Each entry also carries that moment's tracked person boxes (see `RingFrame`), so
        # `_collect_clip` can crop a clip down to the one track an alert is about instead of
        # always sending the full scene — see docs/plans/vlm-throughput-and-concurrent-events.md
        # stage 3 for why that matters when several people are in frame at once.
        self._ring: dict[str, deque[RingFrame]] = {}
        # Cameras registered but not yet decoding, waiting for their first viewer. Only ever
        # holds finite file sources — see `add_camera`. Guarded by `self._lock`, because two
        # viewers attaching at once must still start the worker exactly once.
        self._awaiting_viewer: set[str] = set()
        self._pre_roll_sec = settings.vlm_pre_roll_sec
        self._post_roll_sec = settings.vlm_post_roll_sec
        self._max_frames = settings.vlm_max_frames
        self._history_examples = settings.vlm_history_examples
        self._similar_examples = settings.vlm_similar_examples
        self._similar_min_similarity = settings.vlm_similar_min_similarity
        self._auto_suppress_confidence = settings.vlm_auto_suppress_confidence
        # Ring holds pre_roll + post_roll seconds of frames at the throttled feed rate, with
        # headroom so nothing is evicted before _collect_clip reads it.
        self._default_ring_size = max(
            16, int(_RING_FEED_FPS * (self._pre_roll_sec + self._post_roll_sec + 2.0) * 1.5)
        )
        # Per-camera timestamp of the last frame accepted into the ring, for the feed throttle.
        self._ring_last_fed: dict[str, float] = {}

        self.triage = triage or build_triage_client(settings)
        self.history = history or AlertHistory()
        # Every alert, reviewed or not. `history` holds only judged alerts by design, so it
        # cannot answer "how many did this camera raise" — see incident_log.py.
        self.incidents = incident_log or IncidentLog()
        # Optional: None unless EMBEDDINGS_ENABLED and DATABASE_URL is Postgres. Every call
        # site below checks for None rather than assuming a store, the same way self.triage
        # degrades to NullVLMTriage instead of assuming a real client.
        self.embeddings = embedding_store if embedding_store is not None else create_embedding_store()
        self._embed_text = build_embedder(settings)
        self.notifier = TelegramNotifier(settings.telegram_enabled, settings.telegram_bot_token, settings.telegram_chat_ids)
        self.evidence_store = EvidenceStore(
            backend=settings.evidence_storage_backend,
            local_dir=settings.evidence_storage_dir,
            bucket=settings.evidence_s3_bucket,
            endpoint_url=settings.evidence_s3_endpoint_url,
            region=settings.evidence_s3_region,
            public_base_url=settings.evidence_s3_public_base_url,
        )
        # Two workers: enough to overlap a slow API call with a second camera's alert,
        # small enough that a burst cannot spawn threads without bound.
        self._triage_pool = ThreadPoolExecutor(max_workers=_TRIAGE_POOL_WORKERS, thread_name_prefix="vlm-triage")
        # Same reasoning as _triage_pool: record_operator_verdict is what a guard's
        # "Confirm real"/"False alarm" click is waiting on, and embedding the reason text is
        # an annotation on top of that judgement, not part of it. Even with the 10s client
        # timeout, a guard should not feel an embeddings hiccup as a slow button -- storing
        # it here, off the response path, is what makes the timeout a background-retry
        # concern instead of a foreground wait.
        self._embedding_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embeddings")
        self._triage_in_flight = 0
        # Post-roll wait, done as a `threading.Timer` rather than inside a `_triage_pool`
        # worker. A worker that blocks on `time.sleep(post_roll_sec)` before doing any real
        # work is a worker not answering the API call it exists for — with a handful of
        # alerts firing close together, most of the pool could be asleep at once while the
        # in-flight count still climbs toward `_MAX_TRIAGE_IN_FLIGHT` and later alerts get
        # skipped as "queue saturated" even though no network call is actually running.
        # Tracked so `shutdown()` can cancel pending timers instead of leaking threads.
        self._pending_triage_timers: set[threading.Timer] = set()
        self._incident_workflow = build_incident_workflow(
            wait_for_post_roll=self._wait_for_post_roll,
            collect_evidence=self._collect_triage_evidence,
            assess_evidence=self._assess_triage_evidence,
            persist_verdict=self._persist_triage_verdict,
            persist_failure=self._persist_triage_failure,
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._event_relay = build_event_relay(settings.redis_url, settings.redis_event_channel, self._receive_remote_event)
        self._registry = create_camera_registry(REGISTRY_PATH)
        self._notification_worker = NotificationWorker(
            self.notifier,
            self._publish,
            max_retries=settings.notification_max_retries,
            backoff_sec=settings.notification_backoff_sec,
        )

        if settings.restore_cameras_on_start:
            self._restore_registry()
        else:
            logger.info("RESTORE_CAMERAS_ON_START is off; starting with no cameras running.")

    # ------------------------------------------------------------------ notifications
    def notification_status(self) -> dict:
        return {"enabled": bool(self.notifier.enabled), "configured": bool(self.notifier.chat_ids)}

    def set_notifications_enabled(self, enabled: bool) -> dict:
        self.notifier.enabled = bool(enabled and self.notifier.bot_token)
        return self.notification_status()

    # ------------------------------------------------------------------ lifecycle
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop so worker threads can publish SSE events into it."""
        self._loop = loop

    def shutdown(self) -> None:
        self.manager.stop_all()
        # Cancel post-roll timers that have not fired yet. Without this, a timer started
        # just before shutdown would fire seconds later and submit into an already-shutdown
        # `_triage_pool`, which `_enqueue_triage`'s `RuntimeError` handler already tolerates
        # — but there is no reason to let it fire at all on a process that is exiting.
        # `cancel()` on an already-fired timer is a harmless no-op.
        with self._lock:
            pending = list(self._pending_triage_timers)
            self._pending_triage_timers.clear()
        for timer in pending:
            timer.cancel()
        # wait=False: a triage call already blocked on a 20s API timeout must not hold the
        # process open on Ctrl-C. Its verdict is lost, which is the right trade — the
        # verdict is advisory and the alert it belongs to has already been shown.
        self._triage_pool.shutdown(wait=False)
        self._notification_worker.close()
        self._embedding_pool.shutdown(wait=False)
        self._event_relay.close()

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
            camera.stopped_reason = stat.get("stopped_reason")
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

    def add_camera(self, payload: CameraCreate, camera_id: str | None = None) -> CameraInfo:
        """Register, configure zones for, and start one camera."""
        settings = get_settings()
        camera_id = camera_id or f"cam_{uuid.uuid4().hex[:8]}"
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
            video_url=f"/api/v1/cameras/{camera_id}/video",
            fall_perspective=payload.fall_perspective,
        )

        with self._lock:
            self._cameras[camera_id] = info

        worker = self.manager.add_camera(
            camera_id=camera_id,
            rtsp_url=resolved,
            target_fps=info.target_fps,
            autostart=False,
            # File sources stop at EOF by default. The grid retains their final frame and
            # offers an explicit replay action; RTSP is already continuous.
            loop_finite_sources=payload.loop_source and payload.source_type != "rtsp",
        )
        if worker is not None and payload.fall_perspective is not None:
            worker.pipeline.rule_engine.set_camera_fall_profile(
                camera_id,
                FallPerspectiveProfile(
                    **payload.fall_perspective.model_dump(exclude_none=True)
                ),
            )
        # A finite file waits for its first viewer; a live camera does not.
        #
        # `worker.start()` begins decoding immediately, and `add_camera` returns ~15ms later —
        # long before the browser has re-rendered onto the grid and completed the video
        # socket's handshake. A file source is paced to its own frame rate, so those seconds
        # are not buffered anywhere: they are decoded, analysed and discarded, and the
        # operator who just uploaded a clip watches it from the middle. On a short clip that
        # is most of the evidence.
        #
        # RTSP is deliberately excluded. A live camera exists to be monitored whether or not
        # anyone has the grid open — deferring it would mean no alerts fire on an unattended
        # wall display, which inverts the product. Only a finite file, whose beginning is
        # reproducible and whose whole purpose here is to be watched, waits.
        if worker is not None:
            if self._defers_start(payload.source_type):
                with self._lock:
                    self._awaiting_viewer.add(camera_id)
                logger.info(f"Camera '{info.name}' ({camera_id}) registered on {resolved}; waiting for a viewer")
            else:
                worker.start()
                logger.info(f"Camera '{info.name}' ({camera_id}) started on {resolved}")
        self._persist_registry()
        self._publish("camera.added", info.model_dump())
        return info

    @staticmethod
    def _defers_start(source_type: str) -> bool:
        """Should this source wait for a viewer before it begins decoding?

        Files only. See the note in `add_camera` for why RTSP must never wait.
        """
        return source_type != "rtsp"

    def start_if_awaiting_viewer(self, camera_id: str) -> bool:
        """Begin decoding a deferred camera, if this is the first viewer to ask.

        Called from the video socket and the MJPEG stream as a viewer attaches. Idempotent
        and safe to call on any camera: a camera that was never deferred, or that a previous
        viewer already started, is left alone — the set membership check under the lock is
        what makes two viewers connecting at once start the worker exactly once.
        """
        with self._lock:
            if camera_id not in self._awaiting_viewer:
                return False
            self._awaiting_viewer.discard(camera_id)

        if not self.manager.start_camera(camera_id):
            return False
        logger.info(f"[{camera_id}] first viewer attached; started from the beginning")
        return True

    def replay_camera(self, camera_id: str) -> CameraInfo | None:
        """Restart an ended file source under the same configured camera id."""
        with self._lock:
            camera = self._cameras.get(camera_id)
        if camera is None:
            return None
        if camera.source_type == "rtsp":
            raise ValueError("RTSP cameras are continuous and cannot be replayed")

        stats = {s["camera_id"]: s for s in self.manager.get_status()["cameras"]}
        stat = stats.get(camera_id, {})
        if stat.get("running"):
            raise ValueError("Camera is still running")
        if stat.get("stopped_reason") != "end_of_source":
            raise ValueError("Camera can only be replayed after the video reaches its end")

        worker = self.manager.add_camera(
            camera_id=camera_id,
            rtsp_url=camera.resolved_source,
            target_fps=camera.target_fps,
            autostart=False,
            loop_finite_sources=False,
        )
        if camera.fall_perspective is not None:
            worker.pipeline.rule_engine.set_camera_fall_profile(
                camera_id,
                FallPerspectiveProfile(**camera.fall_perspective.model_dump(exclude_none=True)),
            )
        worker.start()
        camera.running = True
        camera.stopped_reason = None
        self._publish("camera.updated", camera.model_dump())
        logger.info(f"Camera '{camera.name}' ({camera_id}) replayed from the beginning")
        return camera

    def remove_camera(self, camera_id: str) -> bool:
        self.manager.remove_camera(camera_id)
        with self._lock:
            existed = self._cameras.pop(camera_id, None) is not None
            self._frames.pop(camera_id, None)
            self._ring.pop(camera_id, None)
            # Or a camera deleted before anyone ever watched it would leave its id behind,
            # and a later camera reusing that id would refuse to start on its first viewer.
            self._awaiting_viewer.discard(camera_id)
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
        # The display thread's high-cadence buffer (see CameraStreamWorker._display_loop) is
        # what a viewer should actually see; self._frames (populated by _on_frame below, at
        # target_fps) is only a fallback for the brief window before the display thread has
        # encoded its first frame.
        return self.manager.latest_display_jpeg(camera_id) or self._frames.get(camera_id)

    def h264_encoder(self, camera_id: str):
        """The camera's H.264/fMP4 encoder for the video WebSocket, or None.

        MJPEG (`latest_jpeg` above) remains the fallback: it is what the snapshot endpoint
        serves and what the grid uses on a browser without MediaSource, or when this returns
        None or an encoder whose `.available` is False.
        """
        return self.manager.h264_encoder(camera_id)

    def _ring_for(self, camera_id: str) -> deque[RingFrame]:
        """The camera's pre-roll ring, created on first use.

        Sized from `_RING_FEED_FPS` (the throttled feed rate below), not the camera's
        target_fps — the feed rate is what actually governs how many frames land in the
        window, and it is deliberately low so encoding the ring cannot compete with
        inference for CPU.
        """
        ring = self._ring.get(camera_id)
        if ring is None:
            ring = self._ring.setdefault(camera_id, deque(maxlen=self._default_ring_size))
        return ring

    def _on_raw_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Raw frame from a worker's display loop: the pre-roll ring feed.

        Decoupled from `_on_frame` (the inference callback) on purpose. On a CPU host with
        several cameras the inference callback can fall to ~1 fps, which left the ring too
        sparse for `_collect_clip` to build a VLM clip at all — every alert then surfaced as
        "insufficient visual evidence" without an API call ever being made.

        Two costs are controlled here so this cannot itself starve inference:
        - Throttled to `_RING_FEED_FPS`. The display loop calls this ~20x/s per camera; a
          clip only needs a handful of stills over a few seconds, so most of those are
          dropped before any work is done.
        - Downscaled to `_RING_MAX_EDGE` before the JPEG encode. A native 4K encode is
          ~40ms; at 4 cameras that alone saturates the box, and the VLM downsamples to
          1024px anyway.
        """
        now = time.monotonic()
        last = self._ring_last_fed.get(camera_id, 0.0)
        if now - last < 1.0 / _RING_FEED_FPS:
            return
        self._ring_last_fed[camera_id] = now
        try:
            # Read tracks before any resize: they are normalized to [0, 1] (see
            # VisionPipeline.last_tracks), so they stay correct regardless of what this
            # frame gets resized to below or what size the camera was capturing at when
            # the last inference ran. This is what lets _collect_clip crop a clip down to
            # the one person an alert's track_id names — see the ring's docstring above.
            tracks = self.manager.last_tracks(camera_id)
            height, width = frame.shape[:2]
            longest = max(height, width)
            if longest > _RING_MAX_EDGE:
                scale = _RING_MAX_EDGE / longest
                frame = cv2.resize(
                    frame, (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _RING_JPEG_QUALITY])
            if ok:
                self._ring_for(camera_id).append(RingFrame(now, buffer.tobytes(), tracks))
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the display loop
            logger.error(f"[{camera_id}] ring encode failed: {exc}")

    def _on_frame(self, camera_id: str, result: FrameAnalysisResult, frame: np.ndarray) -> None:
        """Every analysed frame: refresh the preview buffer.

        Driving the preview off the incident callback instead would leave the grid frozen
        between events, which reads as a broken demo. The pre-roll ring is fed separately by
        `_on_raw_frame`, so a slow inference cadence no longer starves it — and this path
        also seeds the ring (subject to the same throttle) so an alert in the first ~50ms,
        before the display loop has produced a frame, still gets a clip.
        """
        try:
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _PREVIEW_JPEG_QUALITY])
            if ok:
                # Plain assignment, no lock: dict item assignment is atomic under the GIL
                # and readers only ever want the newest frame.
                self._frames[camera_id] = buffer.tobytes()
            self._on_raw_frame(camera_id, frame)
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
        )

        # Stored before the alert is published so `evidence_url` is already on the very first
        # SSE payload and the operator's image resolves immediately. This is a local file
        # write (or one small S3 PUT) on the processor thread — measured in single-digit
        # milliseconds, against the alternative of publishing an alert whose image 404s until
        # a background thread happens to finish.
        try:
            alert.evidence_url = self.evidence_store.put(alert.id, jpeg)
        except Exception:  # noqa: BLE001 - evidence failure must not stop detection
            logger.exception("Could not persist evidence for alert %s", alert.id)

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
            evidence_url=alert.evidence_url,
        )

        queued_for_triage = self._queue_triage(alert)
        self._publish("alert.created", alert.model_dump())
        # Telegram is deliberately sent only after VLM triage confirms a real incident.
        # The dashboard still receives alert.created immediately, while false positives,
        # uncertain verdicts, and unavailable VLM assessments stay out of the notification
        # channel instead of diluting the operator's alert stream.
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
        """Schedule the second-stage assessment after the post-roll wait.

        Called from the camera's processor thread, so it must return immediately — the
        work here is scheduling a timer, never blocking. The wait itself used to be a
        `time.sleep(post_roll_sec)` inside a `_triage_pool` worker (see `_wait_for_post_roll`),
        which meant every alert occupied one of the pool's few worker threads for the whole
        post-roll window before it made a single network call. A `threading.Timer` does the
        same wait off the pool entirely, so all `_TRIAGE_POOL_WORKERS` stay free for actual
        API calls even when several alerts fire within the same post-roll window.
        """
        if self._post_roll_sec <= 0:
            self._enqueue_triage(alert, timer=None)
            return
        timer = threading.Timer(self._post_roll_sec, lambda: self._enqueue_triage(alert, timer=timer))
        timer.daemon = True
        with self._lock:
            self._pending_triage_timers.add(timer)
        timer.start()

    def _enqueue_triage(self, alert: Alert, timer: threading.Timer | None) -> None:
        """Submit the assessment to the triage pool once the post-roll wait is over.

        Runs on the `threading.Timer`'s own thread (see `_submit_triage`), not the camera's
        processor thread — submitting to a `ThreadPoolExecutor` is cheap but not guaranteed
        instant, and nothing here may stall frame processing.
        """
        if timer is not None:
            with self._lock:
                self._pending_triage_timers.discard(timer)
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
            snapshot = alert.model_dump()
            if should_notify(self.notifier, snapshot):
                self._notification_worker.submit(snapshot)
            self._publish("alert.verdict", snapshot)

    def _wait_for_post_roll(self, alert: Alert) -> None:
        """No-op: the post-roll wait already happened before this workflow was invoked.

        Kept as a graph node (rather than removed) so `build_incident_workflow`'s shape
        still matches its five documented steps one-to-one. The actual wait moved to a
        `threading.Timer` in `_submit_triage` / `_enqueue_triage`, off the `_triage_pool`
        entirely — by the time `_run_triage` calls into this graph, the ring already holds
        the post-roll frames `_collect_triage_evidence` is about to read.
        """
        del alert

    def _collect_triage_evidence(
        self, alert: Alert
    ) -> tuple[list[bytes], int, list[AlertEvidenceFrame], list[str]]:
        """Gather the clip, cropped to the alerting track when the alert has one.

        `alert.track_id` is the id RuleEngine assigned at the moment it fired — see
        `_emit_alert`. A frame in the ring may have re-detected the same person under a
        *different* track_id if YOLO's tracker dropped and RuleEngine's own stitching
        re-linked it (`RuleEngine._stitch_or_create`, `aliased_from`): that link lives on
        RuleEngine's internal track state, not on the `IncidentDetection` objects
        `VisionPipeline.last_tracks` reads, so it is not visible here. In that case
        `_collect_clip`'s per-frame fallback to the full scene is what actually degrades
        gracefully — a wider clip for the frames where the id changed, not a dropped frame
        or a crop of the wrong person.
        """
        frames, pre_count = self._collect_clip(alert.camera_id, alert.track_id)
        return (
            frames,
            pre_count,
            _evidence_frames(frames, pre_count, alert.id, self.evidence_store),
            self._triage_examples(alert),
        )

    def _triage_examples(self, alert: Alert) -> list[str]:
        """Prior-incident context for the triage prompt, most relevant last.

        Two sources, kept distinct because they answer different questions:
        - `recent_examples`: this camera's own recent operator judgements for this label —
          "what does *this* camera's noise look like".
        - `similar_incidents`: semantically nearest judged incidents anywhere (pgvector, on
          the alert's rule reason), fetched *here* rather than after the verdict so the model
          can see a contradicting precedent and lower its confidence. Empty unless
          EMBEDDINGS_ENABLED. The alert has no VLM rationale yet, so the query vector is the
          rule reason alone — which is the evidence the alert actually fired on.

        The similar rows lead and the same-camera history trails, so the newest, most
        on-point judgement is the last thing the model reads.
        """
        history = self.history.recent_examples(
            alert.camera_id, alert.label, limit=self._history_examples
        )
        if self._similar_examples <= 0 or self.embeddings is None or self._embed_text is None:
            return history

        try:
            rows = self.similar_incidents(
                alert.id,
                limit=self._similar_examples,
                min_similarity=self._similar_min_similarity,
            )
        except Exception:  # noqa: BLE001 - a lookup failure must not cost the alert its triage
            logger.exception(f"[{alert.camera_id}] similar-incident lookup failed for {alert.id}")
            return history

        rendered = [
            f'Sự cố tương tự (giống {round(row["similarity"] * 100)}%) tại '
            f'{row.get("camera_name") or row.get("camera_id") or "camera khác"} — '
            f'bảo vệ kết luận là {"SỰ CỐ THẬT" if row.get("operator_verdict") == "real" else "BÁO ĐỘNG GIẢ"}'
            + (f': {row["content"]}' if row.get("content") else "")
            for row in rows
        ]
        return rendered + history

    def _assess_triage_evidence(
        self,
        alert: Alert,
        frames: list[bytes],
        pre_count: int,
        examples: list[str],
    ) -> TriageVerdict:
        # What makes a clip unassessable is having nothing *after* the trigger, not having
        # nothing before it. The two were treated the same and they are not the same.
        #
        # Missing "after" means the camera died at the moment of the event: there is no
        # outcome to judge, and the call would be wasted. Missing "before" means the event
        # happened near the start of the ring — which is the common case, not an edge one.
        # The ring fills from empty when a camera starts, so on a 5fps camera with a 5s
        # pre-roll `pre` is 0 at t≈1s and only reaches 3 by t≈3s (measured). Every alert in
        # that window was refused an assessment and surfaced to the operator as "không thể
        # đánh giá" — a failure they can do nothing about, on exactly the alerts a freshly
        # started clip produces.
        #
        # A one-sided clip still carries most of the signal, and which side it has depends on
        # the failure: a clip with only aftermath (camera stalled at the event) still shows
        # whether the person stayed down; one with only lead-up (event at the very end of the
        # ring, common on a slow CPU host) still shows the fall itself. `pre_count` and the
        # split line in the prompt tell the model what it is missing, so it can answer
        # `uncertain` itself — a better judgement than this branch refusing to call the API.
        #
        # The one genuinely unassessable case is too few frames to show any transition. Two
        # is the floor: one frame cannot show a change.
        if len(frames) < 2:
            return TriageVerdict(
                verdict="uncertain",
                error=(
                    f"insufficient visual evidence: only {len(frames)} frame(s) in the pre-roll "
                    "window — the camera's effective FPS is too low for this clip length"
                ),
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
                # Mirrors the condition _collect_triage_evidence used to decide whether to
                # crop (`alert.track_id is not None`), not a guarantee every frame in
                # `frames` actually got cropped — a per-frame tracking gap falls back to
                # the full scene for that one frame (see `_crop_for_triage`). True here
                # tells the model to expect a mostly-cropped clip, which is the honest
                # signal either way: a model warned about crops it did not get is unaffected,
                # one not warned about crops it did get can misread empty space as absence.
                cropped=alert.track_id is not None,
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
            frame_rows = [frame.model_dump() for frame in evidence_frames]
        self.incidents.record_evidence_frames(alert.id, frame_rows)
        logger.info(
            f"[{alert.camera_id}] triage {alert.label} -> {verdict.verdict} "
            f"({verdict.confidence:.2f}) in {verdict.latency_ms:.0f}ms"
        )

    def _persist_triage_failure(self, alert: Alert, error: str) -> None:
        with self._lock:
            alert.triage_status = "complete"
            alert.triage_verdict = "uncertain"
            alert.triage_error = error

    def _collect_clip(self, camera_id: str, track_id: str | None = None) -> tuple[list[bytes], int]:
        """Sample an evenly-spaced clip from the ring: `(frames, how many precede the event)`.

        The split point is `now - post_roll_sec`, which is when the rule actually fired —
        this runs after the post-roll timer has fired (see `_submit_triage`), so "now" is
        already past it.

        Sampling matters: every frame is a billed image and accuracy degrades with long
        sequences, so a 5-second window at 8 fps is reduced to a handful of stills that still
        span the whole transition.

        `track_id`, when given, crops every sampled frame down to that one person (padded —
        see `_crop_track`) instead of sending the full scene. This is what makes several
        alerts firing on the same camera at once separable: without it, N simultaneous
        alerts on N different people all asked the model to judge the same full-scene clip,
        which has no well-defined answer to "is this alert real" once more than one person
        is in frame. A frame where the track cannot be found, or the crop comes out too
        small to be useful, falls back to that one frame's full scene rather than dropping
        it — a slightly wider clip beats a clip with the transition missing.
        """
        ring = self._ring.get(camera_id)
        if not ring:
            return [], 0

        # Copy first — the worker thread is appending to this deque concurrently.
        samples = list(ring)
        event_time = time.monotonic() - self._post_roll_sec
        window_start = event_time - self._pre_roll_sec

        pre = [frame for frame in samples if window_start <= frame.timestamp <= event_time]
        post = [frame for frame in samples if frame.timestamp > event_time]

        # Two thirds before, one third after. The lead-up is what distinguishes a fall from
        # someone already lying down; the aftermath only has to confirm they stayed there.
        pre_budget = max(1, (self._max_frames * 2) // 3)
        post_budget = max(1, self._max_frames - pre_budget)
        pre = _evenly_spaced(pre, pre_budget)
        post = _evenly_spaced(post, post_budget)

        selected = pre + post
        jpegs = [self._crop_for_triage(frame, track_id) for frame in selected]
        return jpegs, len(pre)

    @staticmethod
    def _crop_for_triage(frame: RingFrame, track_id: str | None) -> bytes:
        """This one sampled frame's JPEG, cropped to `track_id` when possible.

        Falls back to the untouched frame — never raises, never drops the frame — whenever
        there is nothing to crop to: no `track_id` (triage layer called without one, or the
        alert never had a track), the track missing from this particular frame (tracking
        gap, or stitching changed the id — see `_collect_triage_evidence`'s docstring), or
        `_crop_track` deciding the result would be too small to be useful.
        """
        if not track_id:
            return frame.jpeg
        bbox = frame.tracks.get(track_id)
        if bbox is None:
            return frame.jpeg
        cropped = _crop_track(frame.jpeg, bbox)
        return cropped if cropped is not None else frame.jpeg

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
            vlm_error=alert.triage_error,
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
        # Stores this judged incident for the "similar past incidents" lookup (see
        # docs/database/schema.sql's PLANNED section). Only ever embeds a *judged* alert --
        # the comparison library must be guard-confirmed cases only, never a raw unjudged
        # one -- and only if both the store and the embedder are actually configured.
        # Submitted to a pool rather than called inline: this method is what a guard's
        # "Confirm real"/"False alarm" click is waiting on (see submit_verdict's
        # asyncio.to_thread call), and an embeddings API call has no business making that
        # click feel slow. The 10s client timeout in embeddings.py bounds it either way, but
        # off the response path a slow embeddings call costs nothing the guard can feel.
        # Composed from the rule reason *and* the VLM's reading of the clip -- see
        # `embedding_text`. Built here on the caller's thread rather than inside the worker
        # because `alert` is shared mutable state guarded by `self._lock`; the worker gets an
        # immutable string.
        content = embedding_text(alert)
        if self.embeddings is not None and self._embed_text is not None and content:
            try:
                self._embedding_pool.submit(self._store_embedding, alert.id, alert.camera_id, content)
            except RuntimeError:
                pass  # pool is shut down (process exiting); losing one embedding is fine
        self._publish("alert.acknowledged", snapshot)
        return True

    def _store_embedding(self, alert_id: str, camera_id: str, content: str) -> None:
        """Runs on `_embedding_pool`, off the HITL response path -- see the call site.

        Broad except: a flaky embeddings call or a Postgres blip here must not raise on a
        pool worker thread, where nothing would catch it, and must never turn into a failed
        HITL verdict -- that response was already sent before this ever runs.
        """
        try:
            embedded = self._embed_text(content)
            if embedded is not None:
                vector, model = embedded
                self.embeddings.upsert(alert_id, content, vector, model)
        except Exception:  # noqa: BLE001
            logger.exception(f"[{camera_id}] could not store incident embedding for {alert_id}")

    def similar_incidents(
        self, alert_id: str, limit: int = 5, min_similarity: float = 0.0
    ) -> list[dict]:
        """Past judged incidents that read similarly to this one -- see `embedding_text`.

        Works on *any* alert, judged or not -- a guard staring at a brand-new alert is
        exactly who this is for, so the alert being looked up does not need a verdict yet.
        Its embedding is computed here and thrown away rather than stored (see
        `record_operator_verdict`: only judged alerts join the comparison library).

        Restricted to the same label. A fall and an abandoned bag are not comparable evidence
        however closely their text happens to embed, and showing one while judging the other
        invites exactly the wrong inference.

        `limit` is a ceiling and `min_similarity` a floor, both chosen by the caller from the
        VLM's confidence: a strong assessment wants a few close neighbours, a weak one wants
        more context. Neither shapes the real/false mix that comes back -- that stays a
        measurement of what the library actually holds, which is the only reason it is worth
        showing a guard at all.
        """
        if self.embeddings is None or self._embed_text is None:
            return []
        with self._lock:
            alert = next((a for a in self._alerts if a.id == alert_id), None)
        if alert is None:
            return []
        content = embedding_text(alert)
        if not content:
            return []
        embedded = self._embed_text(content)
        if embedded is None:
            return []
        vector, _ = embedded
        try:
            return self.embeddings.search(
                vector,
                limit=limit,
                exclude_alert_id=alert_id,
                label=alert.label,
                min_similarity=min_similarity,
            )
        except Exception:  # noqa: BLE001 - a lookup failure should read as "no similar incidents", not 500
            logger.exception(f"[{alert.camera_id}] could not search for similar incidents to {alert_id}")
            return []

    def incident_frames(self, alert_id: str) -> list[AlertEvidenceFrame]:
        """Rebuild a past incident's evidence strip from storage.

        The guard judging a live alert gets a before/after sequence, because a fall is only
        legible as a sequence -- a single still of someone mid-crouch and one of someone
        mid-collapse are the same picture. A past incident offered as comparable evidence is
        being read for the same judgement, so it needs the same sequence; the one thumbnail
        on the neighbour row is the detection-moment frame, which is precisely the most
        ambiguous one.

        Reconstructed rather than looked up because `AlertEvidenceFrame` is never persisted:
        it lives on the in-memory Alert only. What *is* durable is the frames themselves,
        under the `{alert_id}-f{n}` key that `_evidence_frames` derives. So the keys are
        probed in order and the run that exists is returned.

        Stops at the first gap. `_evidence_frames` writes from index 1 upward and drops a
        frame only when its own write failed, so a gap means the strip ended there -- and
        stopping bounds the probe at `vlm_max_frames` calls even when nothing is stored.
        Phase is recomputed from the same two-thirds split `_triage_frame_budget` uses, since
        which frames were "before" is not recoverable from the keys alone.

        No lock and no exists/serve race to worry about: nothing in the app ever deletes an
        evidence frame -- there is no retention job, no `EvidenceStore.delete` -- so the set
        this probes is append-only for the life of the deployment. If a frame did go missing
        between this call and the browser's fetch, `/evidence/{id}` already 404s cleanly and
        the panel is best-effort by design (a lookup failure reads as "nothing similar").
        """
        settings = get_settings()
        max_frames = settings.vlm_max_frames
        stored = 0
        for index in range(1, max_frames + 1):
            if not self.evidence_store.exists(f"{alert_id}-f{index}"):
                break
            stored = index
        if stored == 0:
            return []

        pre_count = max(1, (stored * 2) // 3)
        return [
            AlertEvidenceFrame(
                index=index,
                phase="before" if index <= pre_count else "after",
                image_url=f"/api/v1/evidence/{alert_id}-f{index}",
            )
            for index in range(1, stored + 1)
        ]

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
            auto_suppress_confidence=self._auto_suppress_confidence,
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

    def incident_history(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: str | None = None,
        label: str | None = None,
        verdict: str | None = None,
    ) -> dict:
        """A page of the durable event log, newest first, for the manager's history screen.

        Reads `IncidentLog`, not the `_alerts` ring buffer. The buffer is capped and dies with
        the process — a "history" built on it would quietly show a few recent alerts and hide
        everything from before the last restart, which is the opposite of what the screen is
        for. `operational_stats` above reads the same log for the same reason.

        Newest first, unlike the log's own oldest-first order: a history screen is read from
        the top, and an operator looking for last night's incident should not have to page to
        the end to find it.

        `verdict` accepts the two recorded values plus `"unreviewed"`, which is not a verdict
        but the absence of one. It has to be expressible: "what has nobody looked at yet" is
        the question a manager actually opens this screen to answer, and it cannot be asked by
        filtering on a value that was never written.

        Paginated because this list grows without bound. Every alert the system has ever
        raised is in here, and serving all of it would put an unbounded response on a route
        the dashboard polls.

        Blocking file I/O — call it off the event loop.
        """
        rows = self.incidents.incidents()

        # Built before filtering: these populate the screen's filter dropdowns, and deriving
        # them from the filtered rows would make each option disappear as soon as it was
        # chosen — the operator could never widen a filter again without reloading.
        cameras = sorted({row.get("camera_name") or row.get("camera_id") or "unknown" for row in rows})
        labels = sorted({row.get("label") or "unknown" for row in rows})

        if camera_id:
            rows = [row for row in rows if row.get("camera_id") == camera_id]
        if label:
            rows = [row for row in rows if row.get("label") == label]
        if verdict == "unreviewed":
            rows = [row for row in rows if not row.get("operator_verdict")]
        elif verdict:
            rows = [row for row in rows if row.get("operator_verdict") == verdict]

        total = len(rows)
        # Reversed rather than re-sorted: `incidents()` already guarantees oldest-first, so
        # this is the same ordering read the other way and costs nothing extra.
        page = list(reversed(rows))[offset : offset + limit]
        for row in page:
            frames = row.get("evidence_frames")
            if isinstance(frames, str):
                try:
                    frames = json.loads(frames)
                except (TypeError, json.JSONDecodeError):
                    frames = []
            row["evidence_frames"] = frames if isinstance(frames, list) else []

        return {
            "items": page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "cameras": cameras,
            "labels": labels,
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
        self._event_relay.publish(message)
        for queue in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._offer, queue, message)

    def _receive_remote_event(self, message: dict) -> None:
        """Fan out an event published by another API or vision-worker process."""
        if not isinstance(message, dict) or not message.get("type"):
            return
        if self._loop is None or self._loop.is_closed():
            return
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
            self._registry.save(cameras)
        except Exception as exc:  # noqa: BLE001 - registry failure must not stop monitoring
            logger.warning(f"Could not persist camera registry: {exc}")

    def _restore_registry(self) -> None:
        """Re-add previously configured cameras on startup, skipping dead sources."""
        try:
            saved = self._registry.load()
        except Exception as exc:  # noqa: BLE001
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
                    ),
                    camera_id=entry.get("id"),
                )
            except Exception as exc:  # noqa: BLE001 - one stale entry must not block startup
                logger.warning(f"Skipping saved camera '{entry.get('name')}': {exc}")
