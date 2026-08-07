"""Runtime that turns the vision pipeline into a multi-camera web service.

This is the missing seam the repo never had: `src/vision/` could analyse a stream and
`src/api/` could answer HTTP, but nothing connected them. `MonitoringService` owns the
`MultiStreamManager`, keeps the latest anonymized JPEG per camera for the grid to poll,
and turns rule-engine events into operator alerts.

Threading is the thing to get right here. Worker callbacks run on the per-camera processor
thread, never on the event loop:

    worker thread ──▶ _on_frame     ──▶ JPEG buffer          (grid, every frame)
                  └─▶ _on_incident  ──▶ alert + SSE publish  (modal, events only)

So three rules hold throughout: nothing in a callback may block (a slow call here stalls
frame processing for that camera outright), nothing in a callback may touch an asyncio
primitive directly (only `loop.call_soon_threadsafe`), and the frame handed to a callback
may alias the reader thread's buffer, so it is encoded immediately rather than stored.

An alert therefore carries only what the vision module itself produced: the anonymized
frame and the rule engine's `reason`. Generating prose about the image is a separate
workstream; when it lands it must go through an executor rather than inline, for the
blocking reason above.
"""

import asyncio
import base64
import json
import logging
import os
import threading
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np

from src.config import get_settings
from src.models.camera_schemas import ALWAYS_ON_LABELS, Alert, CameraCreate, CameraInfo, DemoVideo, SystemStatus
from src.vision.detector import IncidentDetector
from src.vision.pipeline import VisionPipeline
from src.vision.rule_engine import RuleEngine
from src.vision.schemas import FrameAnalysisResult, Zone
from src.vision.stream_manager import MultiStreamManager
from src.vision.zone_config import ZoneManager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.join(_PROJECT_ROOT, "data", "vision", "test")
UPLOAD_DIR = os.path.join(_PROJECT_ROOT, "data", "vision", "uploads")
REGISTRY_PATH = os.path.join(_PROJECT_ROOT, "data", "cameras.json")

# Grid tiles are small and there may be a dozen of them; 65 is visually indistinguishable
# from 95 at tile size and roughly a third of the bytes over the wire.
_PREVIEW_JPEG_QUALITY = 65
_MAX_ALERTS = 50

_DEMO_HINTS = {
    "fall": "fall_detection",
    "abandoned": "abandoned_object",
    "input_test": "intrusion",
}


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


class MonitoringService:
    """Owns every camera worker, the latest-frame buffers, and the alert log."""

    def __init__(self, alert_cooldown_sec: float | None = None) -> None:
        settings = get_settings()
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # One ZoneManager shared by every pipeline. Zones are immutable config with no
        # per-stream state, so sharing is safe — and it is what lets a polygon drawn in
        # the UI reach workers that already exist (see ZoneManager.set_zones).
        self.zone_manager = ZoneManager()

        # Template pipeline: MultiStreamManager clones its *config* per worker, because
        # tracker state and dwell timers must not be shared across cameras.
        template = VisionPipeline(
            detector=IncidentDetector(),
            rule_engine=RuleEngine(
                zone_manager=self.zone_manager,
                alert_cooldown_sec=(
                    settings.alert_cooldown_sec if alert_cooldown_sec is None else alert_cooldown_sec
                ),
            ),
            draw_overlays=True,
        )
        self.manager = MultiStreamManager(
            shared_pipeline=template,
            global_on_frame_callback=self._on_frame,
            global_on_incident_callback=self._on_incident,
        )

        self._cameras: dict[str, CameraInfo] = {}
        self._frames: dict[str, bytes] = {}
        self._alerts: list[Alert] = []
        self._lock = threading.RLock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue] = []

        self._restore_registry()

    # ------------------------------------------------------------------ lifecycle
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop so worker threads can publish SSE events into it."""
        self._loop = loop

    def shutdown(self) -> None:
        self.manager.stop_all()

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
            active_labels=labels,
            target_fps=payload.target_fps or settings.camera_target_fps,
            stream_url=f"/api/v1/cameras/{camera_id}/stream",
        )

        with self._lock:
            self._cameras[camera_id] = info

        self.manager.add_camera(
            camera_id=camera_id,
            rtsp_url=resolved,
            target_fps=info.target_fps,
            autostart=True,
            # A file source that stops at EOF leaves a dead tile in the grid; RTSP is
            # already continuous so looping it is meaningless.
            loop_finite_sources=payload.loop_source and payload.source_type != "rtsp",
        )
        self._persist_registry()
        self._publish("camera.added", info.model_dump())
        logger.info(f"Camera '{info.name}' ({camera_id}) started on {resolved}")
        return info

    def remove_camera(self, camera_id: str) -> bool:
        self.manager.remove_camera(camera_id)
        with self._lock:
            existed = self._cameras.pop(camera_id, None) is not None
            self._frames.pop(camera_id, None)
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
            zone = Zone(
                name="restricted_zone",
                polygon=payload.zone_polygon,
                description=f"Operator-drawn zone for {payload.name}",
            )
            self.zone_manager.set_zones(camera_id, [zone])
        else:
            self.zone_manager.set_zones(camera_id, [])

    # ------------------------------------------------------------------ frames
    def latest_jpeg(self, camera_id: str) -> bytes | None:
        return self._frames.get(camera_id)

    def _on_frame(self, camera_id: str, result: FrameAnalysisResult, frame: np.ndarray) -> None:
        """Every analysed frame: refresh the preview buffer.

        Driving the preview off the incident callback instead would leave the grid frozen
        between events, which reads as a broken demo.
        """
        try:
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _PREVIEW_JPEG_QUALITY])
            if ok:
                # Plain assignment, no lock: dict item assignment is atomic under the GIL
                # and readers only ever want the newest frame.
                self._frames[camera_id] = buffer.tobytes()
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
            # The frame here is the pipeline's *anonymized* output, so faces are already
            # blurred before anything is stored or sent off-box (PRD §3.3).
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

        self._publish("alert.created", alert.model_dump())

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
            total_processed_frames=int(raw["total_processed_frames"]),
            unacknowledged_alerts=len(self.list_alerts(unacknowledged_only=True)),
        )

    # ------------------------------------------------------------------ SSE fan-out
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

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
                        target_fps=entry.get("target_fps"),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one stale entry must not block startup
                logger.warning(f"Skipping saved camera '{entry.get('name')}': {exc}")
