"""Intrusion benchmark: ground truth derived from labelled foot points, and alert matching."""

import pytest

from src.benchmark.run_intrusion_benchmark import (
    Alert,
    ClipRun,
    ClipTruth,
    derive_entries,
    match_clip,
    score,
)

ZONE = [(100, 0), (200, 0), (200, 100), (100, 100)]
FPS = 25.0


def _track(points_by_second: dict[float, tuple[float, float]], until: float) -> dict[int, tuple[float, float]]:
    """Per-frame foot points, holding each given point until the next one."""
    keys = sorted(points_by_second)
    track = {}
    for frame in range(int(until * FPS)):
        t = frame / FPS
        current = max((k for k in keys if k <= t), default=keys[0])
        track[frame] = points_by_second[current]
    return track


INSIDE, OUTSIDE, NEAR = (150, 50), (150, 300), (150, 110)


def test_a_stay_inside_is_one_entry_and_a_brief_crossing_is_none():
    feet = {
        "stays": _track({0: OUTSIDE, 2: INSIDE, 5: OUTSIDE}, 8),
        "blip": _track({0: OUTSIDE, 2: INSIDE, 2.2: OUTSIDE}, 8),  # 0.2 s < MIN_INSIDE_SEC
    }
    entries, walkers = derive_entries(feet, ZONE, FPS)
    assert [(e.object_id, e.start_sec) for e in entries] == [("stays", 2.0)]
    assert walkers == ["blip"]  # never entered for long enough, but came right up to the edge


def test_leaving_and_coming_back_after_the_gap_is_a_second_entry():
    feet = {"p": _track({0: INSIDE, 2: OUTSIDE, 4: INSIDE}, 6)}
    entries, _ = derive_entries(feet, ZONE, FPS)
    assert [e.start_sec for e in entries] == [0.0, 4.0]


def test_a_walker_along_the_edge_is_a_boundary_walker_and_a_far_one_is_not():
    feet = {"near": _track({0: NEAR}, 3), "far": _track({0: OUTSIDE}, 3)}
    entries, walkers = derive_entries(feet, ZONE, FPS)
    assert entries == []
    assert walkers == ["near"]


def _truth() -> ClipTruth:
    feet = {"in": _track({0: OUTSIDE, 2: INSIDE}, 10), "edge": _track({0: NEAR}, 10)}
    entries, walkers = derive_entries(feet, ZONE, FPS)
    return ClipTruth("clip", "clip.mpg", 10.0, entries, feet, walkers)


def test_matching_counts_hits_duplicates_and_boundary_false_alarms():
    run = ClipRun(
        "clip",
        10.0,
        50,
        alerts=[
            Alert(2.2, (152, 55), "track_1"),  # the entry
            Alert(4.0, (150, 50), "track_7"),  # same person, new track ID inside the zone
            Alert(5.0, (148, 105), "track_3"),  # the walker along the edge
        ],
    )
    outcome = match_clip(_truth(), run)
    assert (outcome.detected, outcome.duplicates, outcome.false_alarms) == (1, 1, 1)
    assert outcome.boundary_walkers_alarmed == 1
    assert outcome.latencies_sec == [pytest.approx(0.2)]


def test_an_alert_on_someone_else_does_not_count_as_the_entry():
    run = ClipRun("clip", 10.0, 50, alerts=[Alert(2.2, (20, 20), "track_9")])
    outcome = match_clip(_truth(), run)
    assert (outcome.detected, outcome.false_alarms) == (0, 1)
    assert outcome.missed_entries == ["objin@2.0s"]


def test_score_aggregates_recall_and_boundary_rate():
    truth = _truth()
    s = score([truth], [ClipRun("clip", 3600.0, 50, alerts=[Alert(2.2, (152, 55), "t1")])])
    assert s["recall"] == 1.0
    assert s["precision"] == 1.0
    assert s["boundary_false_alarm_rate"] == 0.0
    assert s["false_alarms_per_hour"] == 0.0


def test_the_shop_zone_leaves_out_the_mannequin_but_keeps_the_floor_around_it():
    from datetime import datetime

    from src.benchmark.run_intrusion_benchmark import SHOP_ZONE_PX, _point_in_polygon
    from src.vision.schemas import Zone
    from src.vision.zone_config import ZoneManager

    zones = ZoneManager()
    zones.set_zones("front", [Zone(name="shop", polygon=[(x / 384, y / 288) for x, y in SHOP_ZONE_PX])])
    noon = datetime(2026, 1, 1, 12, 0)

    def inside(x, y):
        # The benchmark labels and the rule must agree on the hole.
        labelled = _point_in_polygon(x, y, SHOP_ZONE_PX)
        assert labelled == bool(zones.zones_containing("front", x, y, 384, 288, noon))
        return labelled

    mannequin = [(300, 127), (296, 114), (304, 123)]  # feet and lower-body samples
    floor = [(250, 100), (320, 126), (285, 134), (300, 140), (299, 140), (300, 100)]
    assert not any(inside(x, y) for x, y in mannequin)
    assert all(inside(x, y) for x, y in floor)


def test_mot_ground_truth_becomes_zero_based_foot_points(tmp_path):
    from src.benchmark.run_intrusion_benchmark import parse_mot_feet

    path = tmp_path / "gt.txt"
    path.write_text("1,7,100,50,20,80,1,-1,-1,-1\n2,7,102,50,20,80,1,-1,-1,-1\n")
    assert parse_mot_feet(path) == {"7": {0: (110.0, 130.0), 1: (112.0, 130.0)}}
