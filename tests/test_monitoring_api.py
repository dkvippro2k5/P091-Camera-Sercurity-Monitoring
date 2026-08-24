"""Tests for the monitoring API layer: zones, alerts, and route wiring.

None of these load YOLO weights. `IncidentDetector(model_path=None)` is the supported null
detector, so a MonitoringService can be built and exercised in milliseconds — the alert
machinery is independent of whether anything was actually detected.
"""

import os

import numpy as np
import pytest

from src.models.camera_schemas import CameraCreate
from src.services import monitoring as monitoring_module
from src.services.monitoring import MonitoringService
from src.vision.schemas import Zone
from src.vision.zone_config import WILDCARD_CAMERA, ZoneManager


@pytest.fixture
def service(isolated_data):
    """A MonitoringService with a null detector and every file it writes under `tmp_path`.

    `isolated_data` (conftest) is what keeps `_emit_alert` from appending to the real
    `data/incidents.jsonl` — `test_alert_ring_buffer_is_bounded` below fires 210 alerts.
    """
    instance = MonitoringService()
    yield instance
    instance.shutdown()


def _frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


# ------------------------------------------------------------------ zones
def test_intrusion_off_does_not_inherit_the_wildcard_zone(service):
    """The regression this design exists to prevent.

    `data/vision/zones.json` ships a `"*"` entry that ZoneManager applies to any camera it
    has never heard of. A camera whose operator left intrusion *off* must not silently get
    that zone, so the service writes an explicit empty list rather than writing nothing.
    """
    service.zone_manager.set_zones(WILDCARD_CAMERA, [Zone(name="everywhere", polygon=[(0, 0), (1, 0), (1, 1)])])

    payload = CameraCreate(name="No zone", source_type="rtsp", source="rtsp://x", intrusion_enabled=False)
    service._apply_zones("cam_off", payload)

    assert service.zone_manager.zones_for("cam_off") == []
    # A camera that was never configured still inherits the wildcard — proving the empty
    # list above is what suppressed it, not an empty ZoneManager.
    assert len(service.zone_manager.zones_for("cam_unknown")) == 1


def test_drawn_polygon_reaches_the_zone_manager(service):
    polygon = [(0.1, 0.5), (0.6, 0.5), (0.6, 0.9), (0.1, 0.9)]
    payload = CameraCreate(
        name="Lobby",
        source_type="rtsp",
        source="rtsp://x",
        intrusion_enabled=True,
        zone_polygon=polygon,
    )
    service._apply_zones("cam_on", payload)

    zones = service.zone_manager.zones_for("cam_on")
    assert len(zones) == 1
    assert zones[0].polygon == polygon


def test_set_zones_is_visible_to_an_already_built_pipeline():
    """Zones drawn after a worker starts must reach it — the manager instance is shared."""
    manager = ZoneManager()
    manager.set_zones("cam_a", [Zone(name="z", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])])
    assert manager.zones_for("cam_a")[0].name == "z"

    manager.clear_zones("cam_a")
    assert manager.zones_for("cam_a") == []


# ------------------------------------------------------------------ validation
def test_intrusion_requires_a_polygon():
    with pytest.raises(ValueError, match="at least 3 points"):
        CameraCreate(name="x", source_type="demo", source="a.mp4", intrusion_enabled=True)


def test_polygon_must_be_normalized():
    """Pixel coordinates are the likely mistake, and the error has to say so."""
    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        CameraCreate(
            name="x",
            source_type="demo",
            source="a.mp4",
            intrusion_enabled=True,
            zone_polygon=[(10.0, 20.0), (300.0, 20.0), (300.0, 400.0)],
        )


def test_source_resolution_rejects_path_traversal():
    """`source` arrives in an HTTP body, so it must not escape the media directories."""
    resolved = MonitoringService.resolve_source("demo", "../../../../etc/passwd")
    assert os.path.dirname(resolved) == monitoring_module.DEMO_DIR
    assert resolved.endswith("passwd")

    # RTSP is passed through untouched — there is no directory to escape.
    assert MonitoringService.resolve_source("rtsp", "rtsp://cam/1") == "rtsp://cam/1"


def test_file_sources_do_not_loop_by_default():
    payload = CameraCreate(name="Clip", source_type="upload", source="clip.mp4")
    assert payload.loop_source is False


def test_replay_replaces_an_ended_worker_without_removing_the_camera(service, monkeypatch):
    """Replay keeps the configured tile/id and starts a fresh finite-source worker."""
    camera = monitoring_module.CameraInfo(
        id="cam_clip",
        name="Clip",
        source_type="upload",
        source="clip.mp4",
        resolved_source="data/uploads/clip.mp4",
        intrusion_enabled=False,
        active_labels=["fall_detection", "abandoned_object"],
        target_fps=12.0,
    )
    service._cameras[camera.id] = camera
    monkeypatch.setattr(
        service.manager,
        "get_status",
        lambda: {"cameras": [{"camera_id": camera.id, "running": False, "stopped_reason": "end_of_source"}]},
    )

    class Worker:
        def __init__(self):
            self.started = False
            self.pipeline = type("Pipeline", (), {})()

        def start(self):
            self.started = True

    worker = Worker()
    calls = []

    def add_camera(**kwargs):
        calls.append(kwargs)
        return worker

    monkeypatch.setattr(service.manager, "add_camera", add_camera)
    replayed = service.replay_camera(camera.id)

    assert replayed is camera
    assert service.get_camera(camera.id) is camera
    assert worker.started is True
    assert calls == [{
        "camera_id": camera.id,
        "rtsp_url": camera.resolved_source,
        "target_fps": 12.0,
        "autostart": False,
        "loop_finite_sources": False,
    }]
    assert camera.running is True
    assert camera.stopped_reason is None


# ------------------------------------------------------------------ alerts
def test_alert_carries_the_frame_and_the_rule_reason(service):
    """An event must publish immediately, carrying its own evidence.

    `reason` is the rule engine's explanation and the only prose an alert has — there is
    no captioning layer behind it, so nothing here is filled in asynchronously.
    """
    service._emit_alert(
        camera_id="cam_1",
        label="fall_detection",
        confidence=0.9,
        frame=_frame(),
        reason="person_3 collapsed for 3.1s",
    )

    alerts = service.list_alerts()
    assert len(alerts) == 1
    assert alerts[0].label == "fall_detection"
    assert alerts[0].image_data_url.startswith("data:image/jpeg;base64,")
    assert alerts[0].reason == "person_3 collapsed for 3.1s"


def test_acknowledge_marks_the_alert(service):
    service._emit_alert(camera_id="cam_1", label="intrusion", confidence=0.5, frame=_frame())

    alert_id = service.list_alerts()[0].id
    assert service.acknowledge(alert_id) is True
    assert service.list_alerts(unacknowledged_only=True) == []
    assert service.acknowledge("nope") is False


def test_alert_ring_buffer_is_bounded(service):
    """Alerts accumulate forever on a 24/7 system unless something caps them."""
    for _ in range(monitoring_module._MAX_ALERTS + 10):
        service._emit_alert(camera_id="cam_1", label="intrusion", confidence=0.5, frame=_frame())

    assert len(service.list_alerts()) == monitoring_module._MAX_ALERTS


def test_frame_is_downscaled_before_upload():
    """A 1080p JPEG is ~10x the bytes of a 640px one, and the grid renders neither at size."""
    import cv2

    from src.services.monitoring import encode_frame_to_jpeg

    large = np.zeros((1080, 1920, 3), dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(encode_frame_to_jpeg(large, max_edge=640), np.uint8), cv2.IMREAD_COLOR)

    assert max(decoded.shape[:2]) == 640
    # Aspect ratio preserved — a squashed person reads as a fallen one.
    assert decoded.shape[1] / decoded.shape[0] == pytest.approx(1920 / 1080, rel=0.01)


# ------------------------------------------------------------------ routes
# Every route below is behind auth now, so these use the signed-in fixtures from
# `conftest.py` rather than a bare TestClient. Which fixture a test takes is itself an
# assertion: every route in this file is the guard's — the live grid, the alerts, and camera
# setup — so `guard_client` throughout, and `manager_client` appearing here would mean the
# split had drifted. The manager's only surface is `/statistics`, tested in `test_auth.py`
# alongside the 401/403 boundary itself.
def test_health_and_demo_videos(guard_client):
    # /health stays public — it is the Docker healthcheck, which carries no cookie.
    assert guard_client.get("/health").json()["status"] == "ok"

    videos = guard_client.get("/api/v1/demo-videos").json()
    assert all("id" in v and "name" in v for v in videos)


def test_unknown_camera_is_404(guard_client):
    assert guard_client.get("/api/v1/cameras/nope/stream").status_code == 404
    assert guard_client.delete("/api/v1/cameras/nope").status_code == 404


def test_snapshot_returns_a_placeholder_before_the_first_frame(guard_client):
    """An empty body renders a broken-image icon, which reads as a crash."""
    response = guard_client.get("/api/v1/cameras/nope/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")  # JPEG SOI


def test_creating_a_camera_with_a_missing_file_is_a_400(guard_client):
    response = guard_client.post(
        "/api/v1/cameras",
        json={"name": "Ghost", "source_type": "demo", "source": "does_not_exist.mp4"},
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_upload_rejects_a_non_video(guard_client):
    response = guard_client.post("/api/v1/uploads", files={"file": ("evil.sh", b"#!/bin/sh", "text/plain")})
    assert response.status_code == 400
