"""Tests for the temporal rule engine, zone configuration, and pipeline wiring."""

from datetime import datetime

import pytest

from src.vision import (
    BoundingBox,
    CameraZoneConfig,
    FallPerspectiveProfile,
    IncidentDetection,
    RuleEngine,
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


def test_intrusion_accepts_partial_lower_body_overlap_when_feet_are_outside():
    zone = Zone(
        name="restricted",
        polygon=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.7), (0.4, 0.7)],
        active_hours=None,
    )
    engine = make_engine(zone=zone)
    # The feet are below the zone, but the lower body overlaps about half of it.
    person = make_det("person", "person_1", cx=0.5, cy=0.5, bw=0.2, bh=0.8)

    events = engine.evaluate([person], WIDTH, HEIGHT, timestamp=1.0, now=datetime.now(), camera_id="cam_test")

    assert len(events) == 1
    assert events[0].label == "intrusion"


def test_intrusion_fallback_never_samples_below_the_person_box():
    zone = Zone(
        name="below_feet",
        polygon=[(0.35, 0.705), (0.65, 0.705), (0.65, 0.72), (0.35, 0.72)],
        active_hours=None,
    )
    engine = make_engine(zone=zone)
    person = make_det("person", "person_1", cx=0.5, cy=0.5, bw=0.2, bh=0.4)

    assert engine.evaluate(
        [person], WIDTH, HEIGHT, timestamp=1.0, now=datetime.now(), camera_id="cam_test"
    ) == []


def test_cooldown_is_scoped_to_each_incident_track():
    engine = make_engine(zone=make_zone(), alert_cooldown_sec=30.0)
    now = datetime.now()
    first = make_det("person", "person_1", cx=0.48, cy=0.5, bw=0.08, bh=0.35)
    second = make_det("person", "person_2", cx=0.52, cy=0.5, bw=0.08, bh=0.35)

    events = engine.evaluate([first, second], WIDTH, HEIGHT, 1.0, now, "cam_test")

    assert {event.track_id for event in events} == {"person_1", "person_2"}


# ---------------------------------------------------------------------------
# Fall detection
# ---------------------------------------------------------------------------


def test_fall_fires_after_dwell_and_resets_when_upright():
    """Frames are fed ~0.2s apart because a fall is now a *transition*, not a shape.

    All three indicators measure change across a 0.5s window, so stepping the clock by
    whole seconds between frames leaves one sample per window and measures nothing. A live
    camera delivers ~5 fps per 720p stream on this hardware, so the test does too.
    """
    engine = make_engine(fall_dwell_sec=3.0)
    now = datetime.now()

    standing = make_det("person", "person_2", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    collapsed = make_det("person", "person_2", cx=0.5, cy=0.75, bw=0.30, bh=0.09)

    def step(t: float, det: IncidentDetection) -> list[IncidentDetection]:
        return engine.evaluate([det], WIDTH, HEIGHT, timestamp=t, now=now)

    # Upright -> nothing suspected.
    assert step(1.0, standing) == []
    # Drops inside one window: suspicion armed, but the 3s dwell is nowhere near met.
    assert step(1.2, collapsed) == []

    events = []
    for i in range(1, 17):  # t = 1.4 .. 4.4
        events.extend(step(1.2 + i * 0.2, collapsed))
    assert len(events) == 1, "fall should fire exactly once, 3s after the collapse"
    assert events[0].label == "fall_detection"
    assert events[0].track_id == "person_2"

    # Stands back up -> suspicion cleared; a later collapse arms and fires on its own.
    assert step(5.0, standing) == []
    assert step(5.2, standing) == []
    assert step(5.4, collapsed) == []

    events = []
    for i in range(1, 17):  # t = 5.6 .. 8.6
        events.extend(step(5.4 + i * 0.2, collapsed))
    assert len(events) == 1


def test_fall_ignores_brief_bends():
    engine = make_engine(fall_dwell_sec=3.0)
    now = datetime.now()

    standing = make_det("person", "person_2", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    collapsed = make_det("person", "person_2", cx=0.5, cy=0.75, bw=0.30, bh=0.09)

    def step(t: float, det: IncidentDetection) -> list[IncidentDetection]:
        return engine.evaluate([det], WIDTH, HEIGHT, timestamp=t, now=now)

    assert step(1.0, standing) == []
    assert step(1.2, collapsed) == []  # suspicion armed
    # Back on their feet well inside the dwell, and staying up long past the 3s mark:
    # the suspicion is dropped rather than quietly ripening into an alert.
    for i in range(1, 20):
        assert step(1.2 + i * 0.2, standing) == []

    # A genuine collapse later still arms and fires on its own merits.
    assert step(6.0, standing) == []
    assert step(6.2, collapsed) == []
    events = []
    for i in range(1, 17):
        events.extend(step(6.2 + i * 0.2, collapsed))
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


def test_abandoned_clock_restarts_after_object_moves():
    engine = make_engine(abandon_sec=10.0)
    now = datetime.now()

    bag = make_det("backpack", "bag_1", cx=0.2, cy=0.3, bw=0.06, bh=0.08)
    carried = make_det("backpack", "bag_1", cx=0.5, cy=0.3, bw=0.06, bh=0.08)

    engine.evaluate([bag], WIDTH, HEIGHT, timestamp=1.0, now=now)
    engine.evaluate([carried], WIDTH, HEIGHT, timestamp=5.0, now=now)
    assert engine.evaluate([carried], WIDTH, HEIGHT, timestamp=11.0, now=now) == []
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
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.6, 1.0),
        (0.6, 0.4),
        (0.4, 0.4),
        (0.4, 1.0),
        (0.0, 1.0),
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
    config = CameraZoneConfig(camera_id="*", zones=[Zone(name="wild", polygon=[(0, 0), (1, 0), (1, 1)])])
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


def test_abandoned_timer_counts_from_owner_seen_at_time_zero():
    """An owner last seen at t=0.0 must still start the unattended clock.

    process_video passes video-relative time, so the first frame of every video is
    exactly 0.0 — a falsy value. Selecting the timer with `or` silently discarded it and
    restarted the countdown from the stationary anchor, delaying the alert.
    """
    engine = make_engine(abandon_sec=10.0)
    now = datetime.now()

    # t=0.0: bag and its owner together, so last_owner_seen is set to exactly 0.0.
    engine.evaluate(
        [make_det("backpack", "bag_1", 0.2, 0.3, 0.06, 0.08), make_det("person", "owner_1", 0.21, 0.32, 0.08, 0.35)],
        WIDTH,
        HEIGHT,
        0.0,
        now,
        "cam_test",
    )
    state = engine._tracks[("cam_test", "bag_1")]
    assert state.last_owner_seen == 0.0

    # Owner leaves. The bag is then moved once, resetting its stationary anchor at t=5,
    # which is the value the buggy fallback used instead of the owner timestamp.
    engine.evaluate(
        [make_det("backpack", "bag_1", 0.6, 0.3, 0.06, 0.08)],
        WIDTH,
        HEIGHT,
        5.0,
        now,
        "cam_test",
    )

    fired_at = None
    for step in range(11, 41):
        t = step * 0.5
        events = engine.evaluate(
            [make_det("backpack", "bag_1", 0.6, 0.3, 0.06, 0.08)],
            WIDTH,
            HEIGHT,
            t,
            now,
            "cam_test",
        )
        if events:
            fired_at = t
            break

    # The bag moved at t=5, so it must remain stationary for ten seconds after that.
    assert fired_at is not None, "abandoned_object never fired"
    assert fired_at == pytest.approx(15.0, abs=0.6), f"fired at t={fired_at}, expected ~15.0"


def test_track_stitching_across_occlusion():
    engine = make_engine(track_ttl_sec=1.0, track_stitch_window_sec=2.0)
    now = datetime.now()

    person_v1 = make_det("person", "person_10", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    person_v2 = make_det("person", "person_20", cx=0.51, cy=0.51, bw=0.08, bh=0.35)

    # First track seen
    engine.evaluate([person_v1], WIDTH, HEIGHT, timestamp=1.0, now=now, camera_id="cam_test")
    assert ("cam_test", "person_10") in engine._tracks

    # Evict track after ttl (1.0s)
    engine.evaluate([], WIDTH, HEIGHT, timestamp=2.5, now=now, camera_id="cam_test")
    assert ("cam_test", "person_10") not in engine._tracks
    assert ("cam_test", "person_10") in engine._expired_tracks

    # New track appears near old position within stitch window (2.0s) -> stitched
    engine.evaluate([person_v2], WIDTH, HEIGHT, timestamp=3.0, now=now, camera_id="cam_test")
    assert ("cam_test", "person_20") in engine._tracks
    state = engine._tracks[("cam_test", "person_20")]
    assert state.aliased_from == "person_10"


def test_default_stitching_handles_an_id_change_before_ttl_expiry():
    engine = make_engine(zone=make_zone())
    now = datetime.now()
    first_id = make_det("person", "person_10", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    second_id = make_det("person", "person_20", cx=0.505, cy=0.5, bw=0.08, bh=0.35)

    assert len(engine.evaluate([first_id], WIDTH, HEIGHT, 1.0, now, "cam_test")) == 1
    # The old ID is absent but has not reached the one-second TTL yet.
    assert engine.evaluate([second_id], WIDTH, HEIGHT, 1.2, now, "cam_test") == []

    state = engine._tracks[("cam_test", "person_20")]
    assert state.aliased_from == "person_10"
    assert ("cam_test", "person_10") not in engine._tracks


def test_fall_dwell_survives_track_id_change():
    engine = make_engine(fall_dwell_sec=1.5, track_ttl_sec=0.3, track_stitch_window_sec=2.0)
    now = datetime.now()
    standing = make_det("person", "person_10", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    collapsed_v1 = make_det("person", "person_10", cx=0.5, cy=0.75, bw=0.30, bh=0.09)
    collapsed_v2 = make_det("person", "person_20", cx=0.5, cy=0.75, bw=0.30, bh=0.09)

    engine.evaluate([standing], WIDTH, HEIGHT, 0.0, now, "cam_test")
    engine.evaluate([standing], WIDTH, HEIGHT, 0.2, now, "cam_test")
    engine.evaluate([collapsed_v1], WIDTH, HEIGHT, 0.4, now, "cam_test")
    engine.evaluate([collapsed_v1], WIDTH, HEIGHT, 0.6, now, "cam_test")
    engine.evaluate([], WIDTH, HEIGHT, 1.0, now, "cam_test")

    assert engine.evaluate([collapsed_v2], WIDTH, HEIGHT, 1.2, now, "cam_test") == []
    events = engine.evaluate([collapsed_v2], WIDTH, HEIGHT, 2.0, now, "cam_test")

    assert len(events) == 1
    assert events[0].track_id == "person_20"
    state = engine._tracks[("cam_test", "person_20")]
    assert state.aliased_from == "person_10"
    assert state.fallen_since == pytest.approx(0.4)


def test_persistent_fall_does_not_refire_after_track_id_change():
    engine = make_engine(fall_dwell_sec=0.4, track_ttl_sec=0.3, track_stitch_window_sec=2.0)
    now = datetime.now()
    standing = make_det("person", "person_10", cx=0.5, cy=0.5, bw=0.08, bh=0.35)
    collapsed_v1 = make_det("person", "person_10", cx=0.5, cy=0.75, bw=0.30, bh=0.09)
    collapsed_v2 = make_det("person", "person_20", cx=0.5, cy=0.75, bw=0.30, bh=0.09)

    engine.evaluate([standing], WIDTH, HEIGHT, 0.0, now, "cam_test")
    engine.evaluate([standing], WIDTH, HEIGHT, 0.2, now, "cam_test")
    engine.evaluate([collapsed_v1], WIDTH, HEIGHT, 0.4, now, "cam_test")
    first = engine.evaluate([collapsed_v1], WIDTH, HEIGHT, 0.8, now, "cam_test")
    assert len(first) == 1

    engine.evaluate([], WIDTH, HEIGHT, 1.2, now, "cam_test")
    assert engine.evaluate([collapsed_v2], WIDTH, HEIGHT, 1.4, now, "cam_test") == []
    assert engine._tracks[("cam_test", "person_20")].fall_phase == "persistent"


def test_pos_history_records_top_edge_and_height():
    """The fall indicators read (timestamp, y_top, height) — not a centroid.

    Storing the wrong triple here fails silently: every indicator still computes, just off
    the wrong numbers, so the layout is worth pinning.
    """
    engine = make_engine(fall_dwell_sec=3.0)
    now = datetime.now()

    standing = make_det("person", "person_fall", cx=0.5, cy=0.2, bw=0.08, bh=0.35)
    collapsed = make_det("person", "person_fall", cx=0.5, cy=0.8, bw=0.30, bh=0.09)

    engine.evaluate([standing], WIDTH, HEIGHT, timestamp=1.0, now=now)
    # Rapid drop downward, inside the 0.5s window so both samples survive pruning.
    engine.evaluate([collapsed], WIDTH, HEIGHT, timestamp=1.5, now=now)

    history = engine._tracks[(None, "person_fall")].pos_history
    assert len(history) == 2
    for (timestamp, y_top, height, _center_y, _area), det in zip(history, (standing, collapsed), strict=True):
        assert y_top == float(det.bbox.ymin)
        assert height == float(det.bbox.ymax - det.bbox.ymin)
    assert [sample[0] for sample in history] == [1.0, 1.5]


def test_abandoned_original_owner_tracking():
    engine = make_engine(abandon_sec=5.0)
    now = datetime.now()

    bag = make_det("backpack", "bag_owner_test", cx=0.3, cy=0.3, bw=0.06, bh=0.08)
    owner = make_det("person", "owner_orig", cx=0.31, cy=0.31, bw=0.08, bh=0.35)
    bystander = make_det("person", "bystander_stranger", cx=0.45, cy=0.32, bw=0.08, bh=0.35)

    # Initial frame: owner near bag -> owner_track_id set to owner_orig
    engine.evaluate([bag, owner], WIDTH, HEIGHT, timestamp=1.0, now=now)
    state = engine._tracks[(None, "bag_owner_test")]
    assert state.owner_track_id == "owner_orig"

    # Owner leaves, but stranger stands near bag
    engine.evaluate([bag, bystander], WIDTH, HEIGHT, timestamp=2.0, now=now)
    # Because original owner left, timer accumulates despite bystander
    events = engine.evaluate([bag, bystander], WIDTH, HEIGHT, timestamp=7.0, now=now)
    assert len(events) == 1
    assert events[0].label == "abandoned_object"


def test_abandoned_binds_the_nearest_owner_not_detection_order():
    engine = make_engine(abandon_sec=5.0)
    now = datetime.now()
    bag = make_det("backpack", "bag_1", cx=0.3, cy=0.3, bw=0.06, bh=0.08)
    farther = make_det("person", "farther", cx=0.48, cy=0.3, bw=0.08, bh=0.35)
    nearest = make_det("person", "nearest", cx=0.31, cy=0.3, bw=0.08, bh=0.35)

    engine.evaluate([bag, farther, nearest], WIDTH, HEIGHT, 1.0, now)

    assert engine._tracks[(None, "bag_1")].owner_track_id == "nearest"


def test_owner_association_survives_person_track_id_change():
    engine = make_engine(abandon_sec=1.0)
    now = datetime.now()
    bag = make_det("backpack", "bag_1", cx=0.3, cy=0.3, bw=0.06, bh=0.08)
    owner_v1 = make_det("person", "owner_1", cx=0.31, cy=0.3, bw=0.08, bh=0.35)
    owner_v2 = make_det("person", "owner_2", cx=0.315, cy=0.3, bw=0.08, bh=0.35)

    engine.evaluate([bag, owner_v1], WIDTH, HEIGHT, 0.0, now)
    engine.evaluate([bag, owner_v2], WIDTH, HEIGHT, 0.2, now)
    state = engine._tracks[(None, "bag_1")]

    assert state.owner_track_id == "owner_2"
    assert engine.evaluate([bag, owner_v2], WIDTH, HEIGHT, 2.0, now) == []


def test_camera_fall_profile_overrides_only_the_selected_camera():
    engine = make_engine()
    engine.set_camera_fall_profile(
        "cam_wide",
        FallPerspectiveProfile(area_growth_ratio=0.25, confirmation_sec=0.0),
    )

    assert engine._fall_profile("cam_wide").area_growth_ratio == 0.25
    assert engine._fall_profile("cam_other").area_growth_ratio == engine.fall_area_growth_ratio
