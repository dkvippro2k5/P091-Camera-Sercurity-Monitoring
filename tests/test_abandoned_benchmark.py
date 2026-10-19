"""Abandoned-object benchmark: matching alerts to hand-labelled abandonments."""

from src.benchmark.run_abandoned_benchmark import Alert, ClipRun, ClipTruth, ObjectLabel, match_clip, score

BAG = ObjectLabel((100, 100, 40, 40), placed=10.0, unattended=20.0, removed=50.0)
ON_BAG = (120.0, 120.0)


def _run(*alerts: tuple[float, tuple[float, float]], footage: float = 60.0) -> ClipRun:
    return ClipRun("clip", footage, 300, alerts=[Alert(t, c, "backpack") for t, c in alerts])


def test_first_alert_after_the_owner_leaves_is_a_hit_and_later_ones_duplicates():
    outcome = match_clip(ClipTruth("clip", "x", [BAG]), _run((22.0, ON_BAG), (30.0, ON_BAG)))
    assert (outcome.detected, outcome.duplicates, outcome.false_alarms) == (1, 1, 0)
    assert outcome.latencies_sec == [2.0]


def test_an_alert_while_the_owner_is_still_beside_it_is_an_early_false_alarm():
    outcome = match_clip(ClipTruth("clip", "x", [BAG]), _run((12.0, ON_BAG)))
    assert (outcome.detected, outcome.false_alarms) == (0, 1)
    assert "early" in outcome.false_alarm_details[0]


def test_alerts_after_removal_or_elsewhere_are_false_alarms():
    outcome = match_clip(ClipTruth("clip", "x", [BAG]), _run((55.0, ON_BAG), (30.0, (500.0, 400.0))))
    assert outcome.false_alarms == 2
    assert {d.split()[1] for d in outcome.false_alarm_details} == {"after", "unlabelled"}


def test_an_alert_on_attended_luggage_is_a_false_alarm():
    attended = ObjectLabel((100, 100, 40, 40), placed=10.0, unattended=None)
    outcome = match_clip(ClipTruth("clip", "x", [attended]), _run((40.0, ON_BAG)))
    assert (outcome.abandonments, outcome.false_alarms) == (0, 1)
    assert "attended" in outcome.false_alarm_details[0]


def test_score_counts_events_not_clips():
    truth = ClipTruth("clip", "x", [BAG, ObjectLabel((400, 300, 40, 40), 5.0, 8.0)])
    s = score([truth], [_run((22.0, ON_BAG), footage=3600.0)])
    assert (s["abandonments"], s["detected"], s["recall"]) == (2, 1, 0.5)
    assert s["false_alarms_per_hour"] == 0.0
