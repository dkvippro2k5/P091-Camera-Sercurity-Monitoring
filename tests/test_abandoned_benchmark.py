"""Abandoned-object benchmark: matching alerts to hand-labelled abandonments."""

from src.benchmark.run_abandoned_benchmark import (
    Alert,
    ClipRun,
    ClipTruth,
    ObjectLabel,
    caviar_left_objects,
    match_clip,
    parse_viper_objects,
    score,
)

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


VIPER = """<viper><config><descriptor name="Information" type="FILE"></attribute></descriptor></config>
<data><sourcefile filename="x.mpg">
<object framespan="2000:4727" id="0" name="PutObject">
<attribute name="BoundingBox"><data:bbox height="155" width="88" x="59" y="242"/></attribute></object>
<object framespan="2870:4727 " id="0" name="AbandonedObject">
<attribute name="BoundingBox"><data:bbox height="155" width="88" x="59" y="242"/></attribute></object>
</sourcefile></data></viper>"""


def test_viper_ground_truth_is_read_even_when_not_well_formed(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(VIPER)
    events = parse_viper_objects(path)
    assert events["PutObject"] == ((59, 242, 88, 155), 2000, 4727)
    assert events["AbandonedObject"][1:] == (2870, 4727)


def _caviar_frame(number: int, objects: list[tuple[str, str, int, int]]) -> str:
    body = "".join(
        f'<object id="{oid}"><box xc="{x}" yc="{y}" w="10" h="10"/>'
        f"<hypothesislist><hypothesis><role>{role}</role></hypothesis></hypothesislist></object>"
        for oid, role, x, y in objects
    )
    return f'<frame number="{number}"><objectlist>{body}</objectlist></frame>'


def test_caviar_owner_is_gone_once_they_stay_far_for_a_second(tmp_path):
    frames = []
    for n in range(200):
        objects = []
        if n >= 50:
            objects.append(("bag", "leaving object", 100, 100))
        owner_x = 110 if n < 100 else 300  # beside the bag, then far away from frame 100
        objects.append(("owner", "walker", owner_x, 100))
        frames.append(_caviar_frame(n, objects))
    path = tmp_path / "LeftBag.xml"
    path.write_text("<dataset>" + "".join(frames) + "</dataset>")

    [label] = caviar_left_objects(path, fps=25.0, frame_width=384)
    assert label.placed == 2.0
    assert label.unattended == 4.0
    assert label.removed is None  # still there at the end
