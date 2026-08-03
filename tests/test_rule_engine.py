"""Tests for the temporal rule engine, zone configuration, and pipeline wiring."""

from datetime import datetime

import numpy as np

from src.vision import (
    BoundingBox,
    CameraZoneConfig,
    IncidentDetection,
    IncidentDetector,
    RuleEngine,
    VisionPipeline,
    Zone,
    ZoneManager,
    is_zone_active,
    point_in_polygon,
)

WIDTH, HEIGHT = 640, 480


def make_det(label: str, track_id: str, cx: float, cy: float, bw: float, bh: float) -> IncidentDetection:
    """Detection with a bbox built from normalized center/size, like the mock does."""
    return IncidentDetection(
        label=label,
        track_id=track_id,
        confidence=0.9,
        timestamp=1.0,
        bbox=BoundingBox(
            xmin=int((cx - bw / 2) * WIDTH),
            ymin=int((cy - bh / 2) * HEIGHT),
            xmax=int((cx + bw / 2) * WIDTH),
            ymax=int((cy + bh / 2) * HEIGHT),
        ),
    )


def make_engine(*, zone: Zone | None = None, **kwargs) -> RuleEngine:
    configs = [CameraZoneConfig(camera_id="cam_test", zones=[zone])] if zone else []
    return RuleEngine(zone_manager=ZoneManager(configs), **kwargs)


def make_zone(name: str = "restricted") -> Zone:
    return Zone(
        name=name,
        polygon=[(0.4, 0.2), (0.6, 0.2), (0.6, 0.8), (0.4, 0.8)],
        active_hours=None,
    )


# ---------------------------------------------------------------------------
# Intrusion
# ---------------------------------------------------------------------------


def test_intrusion_fires_once_then_rearms_after_leaving():
    engine = make_engine(zone=make_zone())
    now = datetime.now()

    inside = make_det("person", "person_1", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    outside = make_det("person", "person_1", cx=0.1, cy=0.5, bw=0.08, bh=0.35)

    events = engine.evaluate([inside], WIDTH, HEIGHT, timestamp=1.0, now=now, camera_id="cam_test")
    assert len(events) == 1
    assert events[0].label == "intrusion"
    assert events[0].zone_name == "restricted"

    # Same track still inside: no duplicate alert.
    assert engine.evaluate([inside], WIDTH, HEIGHT, timestamp=1.1, now=now, camera_id="cam_test") == []

    # Leave (re-arms), then come back: fires again.
    assert engine.evaluate([outside], WIDTH, HEIGHT, timestamp=1.2, now=now, camera_id="cam_test") == []
    events = engine.evaluate([inside], WIDTH, HEIGHT, timestamp=1.3, now=now, camera_id="cam_test")
    assert len(events) == 1
    assert events[0].zone_name == "restricted"


def test_intrusion_ignores_idle_zones_outside_active_hours():
    night_zone = Zone(
        name="night_lobby",
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        active_hours="22:00-05:00",
    )
    engine = make_engine(zone=night_zone)
    person = make_det("person", "person_1", cx=0.5, cy=0.5, bw=0.08, bh=0.35)

    daytime = datetime(2026, 8, 3, 12, 0)
    assert engine.evaluate([person], WIDTH, HEIGHT, timestamp=1.0, now=daytime, camera_id="cam_test") == []

    night = datetime(2026, 8, 3, 23, 0)
    events = engine.evaluate([person], WIDTH, HEIGHT, timestamp=1.1, now=night, camera_id="cam_test")
    assert len(events) == 1
    assert events[0].zone_name == "night_lobby"


# ---------------------------------------------------------------------------
# Fall detection
# ---------------------------------------------------------------------------


def test_fall_fires_after_dwell_and_resets_when_upright():
    engine = make_engine(fall_dwell_sec=3.0)
    now = datetime.now()

    standing = make_det("person", "person_2", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    collapsed = make_det("person", "person_2", cx=0.5, cy=0.75, bw=0.30, bh=0.09)

    # Upright -> no fall state.
    assert engine.evaluate([standing], WIDTH, HEIGHT, timestamp=1.0, now=now) == []

    # Collapsed, but not past the 3s dwell.
    assert engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=1.5, now=now) == []

    # Past the dwell -> fires exactly once.
    events = engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=5.0, now=now)
    assert len(events) == 1
    assert events[0].label == "fall_detection"
    assert events[0].track_id == "person_2"
    assert engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=5.1, now=now) == []

    # Stands back up -> timer clears; a later collapse fires again.
    assert engine.evaluate([standing], WIDTH, HEIGHT, timestamp=6.0, now=now) == []
    assert engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=9.0, now=now) == []
    events = engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=12.5, now=now)
    assert len(events) == 1


def test_fall_ignores_brief_bends():
    engine = make_engine(fall_dwell_sec=3.0)
    now = datetime.now()

    standing = make_det("person", "person_2", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    collapsed = make_det("person", "person_2", cx=0.5, cy=0.75, bw=0.30, bh=0.09)

    assert engine.evaluate([standing], WIDTH, HEIGHT, timestamp=1.0, now=now) == []
    # Bends down, gets back up inside the dwell window.
    assert engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=1.5, now=now) == []
    assert engine.evaluate([standing], WIDTH, HEIGHT, timestamp=3.0, now=now) == []
    # A second bend later starts a fresh timer instead of resuming the old one.
    assert engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=4.0, now=now) == []
    events = engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=7.5, now=now)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Abandoned object
# ---------------------------------------------------------------------------


def test_abandoned_fires_after_owner_leaves():
    engine = make_engine(abandon_sec=10.0)
    now = datetime.now()

    bag = make_det("backpack", "bag_1", cx=0.2, cy=0.3, bw=0.06, bh=0.08)
    owner = make_det("person", "person_3", cx=0.22, cy=0.32, bw=0.08, bh=0.35)

    # Owner stands nearby: the unattended clock never runs.
    assert engine.evaluate([bag, owner], WIDTH, HEIGHT, timestamp=1.0, now=now) == []
    assert engine.evaluate([bag, owner], WIDTH, HEIGHT, timestamp=5.0, now=now) == []

    # Owner leaves; 10s later the bag is flagged.
    assert engine.evaluate([bag], WIDTH, HEIGHT, timestamp=6.0, now=now) == []
    assert engine.evaluate([bag], WIDTH, HEIGHT, timestamp=14.0, now=now) == []
    events = engine.evaluate([bag], WIDTH, HEIGHT, timestamp=15.0, now=now)
    assert len(events) == 1
    assert events[0].label == "abandoned_object"
    assert events[0].track_id == "bag_1"
    assert engine.evaluate([bag], WIDTH, HEIGHT, timestamp=16.0, now=now) == []


def test_abandoned_movement_restarts_the_clock():
    engine = make_engine(abandon_sec=10.0)
    now = datetime.now()

    bag = make_det("backpack", "bag_1", cx=0.2, cy=0.3, bw=0.06, bh=0.08)
    carried = make_det("backpack", "bag_1", cx=0.5, cy=0.3, bw=0.06, bh=0.08)

    engine.evaluate([bag], WIDTH, HEIGHT, timestamp=1.0, now=now)
    # Settled at t=1, the bag would be abandoned at t=11. Being carried at t=5
    # resets the anchor, so the clock restarts there instead.
    engine.evaluate([carried], WIDTH, HEIGHT, timestamp=5.0, now=now)
    assert engine.evaluate([carried], WIDTH, HEIGHT, timestamp=14.0, now=now) == []
    events = engine.evaluate([carried], WIDTH, HEIGHT, timestamp=15.0, now=now)
    assert len(events) == 1
    assert events[0].label == "abandoned_object"


# ---------------------------------------------------------------------------
# Zone configuration
# ---------------------------------------------------------------------------


def test_point_in_polygon():
    square = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    assert point_in_polygon(0.5, 0.5, square)
    assert not point_in_polygon(0.1, 0.5, square)
    assert not point_in_polygon(0.5, 0.9, square)

    # Concave U-shape: the notch is outside even though its bounding box is inside.
    concave = [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0),
        (0.6, 0.4), (0.4, 0.4), (0.4, 1.0), (0.0, 1.0),
    ]
    assert not point_in_polygon(0.5, 0.7, concave)
    assert point_in_polygon(0.5, 0.2, concave)


def test_active_hours_and_midnight_wrap():
    day_zone = Zone(name="day", polygon=[(0, 0), (1, 0), (1, 1)], active_hours="08:00-18:00")
    assert is_zone_active(day_zone, datetime(2026, 8, 3, 12, 0))
    assert not is_zone_active(day_zone, datetime(2026, 8, 3, 20, 0))

    night_zone = Zone(name="night", polygon=[(0, 0), (1, 0), (1, 1)], active_hours="22:00-05:00")
    assert is_zone_active(night_zone, datetime(2026, 8, 3, 23, 30))
    assert is_zone_active(night_zone, datetime(2026, 8, 4, 3, 0))
    assert not is_zone_active(night_zone, datetime(2026, 8, 3, 12, 0))

    always_zone = Zone(name="always", polygon=[(0, 0), (1, 0), (1, 1)], active_hours=None)
    assert is_zone_active(always_zone, datetime(2026, 8, 3, 4, 0))


def test_zone_manager_wildcard_fallback_and_file_loading(tmp_path):
    config = CameraZoneConfig(
        camera_id="*", zones=[Zone(name="wild", polygon=[(0, 0), (1, 0), (1, 1)])]
    )
    manager = ZoneManager([config])
    assert manager.zone_count == 1
    # Unconfigured camera falls back to the wildcard entry.
    assert len(manager.zones_for("cam_unknown")) == 1

    zone_file = tmp_path / "zones.json"
    zone_file.write_text(
        '{"cameras": [{"camera_id": "cam_a", "zones": [{"name": "z1", "polygon": [[0, 0], [1, 0], [1, 1]]}]}]}',
        encoding="utf-8",
    )
    loaded = ZoneManager.from_file(str(zone_file))
    assert loaded.zone_count == 1
    assert len(loaded.zones_for("cam_a")) == 1

    # Missing or corrupt file yields an empty manager instead of crashing.
    assert ZoneManager.from_file(str(tmp_path / "missing.json")).zone_count == 0
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not json", encoding="utf-8")
    assert ZoneManager.from_file(str(bad_file)).zone_count == 0


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_pipeline_triggers_all_three_events_on_mock_timeline():
    """Drive the mock scenario for 16s of footage; every rule must fire at least once."""
    zone = Zone(
        name="restricted_corridor",
        polygon=[(0.45, 0.35), (0.72, 0.35), (0.72, 0.95), (0.45, 0.95)],
        active_hours=None,
    )
    engine = RuleEngine(
        zone_manager=ZoneManager([CameraZoneConfig(camera_id="*", zones=[zone])])
    )
    # Pinned to mock: the scripted timeline below is what drives all three rules. The
    # default detector would load real weights and see nothing in a blank frame.
    pipeline = VisionPipeline(
        detector=IncidentDetector(model_path=None), rule_engine=engine
    )
    dummy_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    labels = set()
    first_fired: dict[str, int] = {}
    for frame_idx in range(480):  # 16 seconds at 30 fps
        _, result = pipeline.process_frame(
            dummy_frame, frame_id=frame_idx, timestamp=frame_idx / 30.0
        )
        for event in result.events:
            labels.add(event.label)
            first_fired.setdefault(event.label, frame_idx)

    assert {"intrusion", "fall_detection", "abandoned_object"} <= labels
    # Mock timeline: intrusion ~8s, fall ~8s, abandoned ~16s into the stream.
    assert first_fired["intrusion"] / 30.0 < 12.0
    assert first_fired["fall_detection"] / 30.0 < 12.0
    assert first_fired["abandoned_object"] / 30.0 < 20.0
