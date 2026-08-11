"""Tests for zone active hours (PRD §3.1's "khung giờ cấm") and the PRD §4 eval report.

Offline and weight-free — no detector, no network. Two subjects that share a file because
they are the two halves of the same gap: the night window is the intrusion user story's
own condition, and the report is what turns operator labels into a number that story can
be judged by.

What these pin, in order of how expensive the bug would be:

  - a wrapping window (22:00-05:00) is active at 23:00 and at 03:00, and not at noon
  - a bad window is *rejected* at configuration time rather than silently ignored
  - the window survives a backend restart (it lives in the persisted registry)
  - the report never invents a recall figure
  - the report counts real incidents a gating VLM would have hidden
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.camera_schemas import CameraCreate
from src.services import monitoring as monitoring_module
from src.services.alert_history import AlertHistory
from src.services.eval_report import build_report
from src.services.monitoring import MonitoringService
from src.vision.detector import IncidentDetector
from src.vision.schemas import Zone
from src.vision.zone_config import ZoneManager, is_zone_active, validate_active_hours

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def service(isolated_data):
    """A MonitoringService with a null detector and every file it writes under `tmp_path`."""
    instance = MonitoringService()
    yield instance
    instance.shutdown()


def _zone(active_hours: str | None) -> Zone:
    return Zone(name="restricted_zone", polygon=SQUARE, active_hours=active_hours)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 9, hour, minute)


# ------------------------------------------------------------------ the window itself
def test_a_window_that_wraps_midnight_covers_both_sides_of_it():
    """The PRD's own example is 22:00-05:00, which is the case naive comparison gets wrong.

    `start <= now <= end` is false at 23:00 *and* at 03:00 for this window, so a zone armed
    for the night would be asleep all night — the exact hours it exists for.
    """
    night = _zone("22:00-05:00")

    assert is_zone_active(night, _at(23, 30)) is True
    assert is_zone_active(night, _at(3)) is True
    assert is_zone_active(night, _at(22)) is True, "inclusive at the start"
    assert is_zone_active(night, _at(5)) is True, "inclusive at the end"


def test_a_wrapping_window_is_inactive_during_the_day():
    assert is_zone_active(_zone("22:00-05:00"), _at(12)) is False
    assert is_zone_active(_zone("22:00-05:00"), _at(21, 59)) is False
    assert is_zone_active(_zone("22:00-05:00"), _at(5, 1)) is False


def test_a_same_day_window_behaves_normally():
    office = _zone("09:00-17:00")

    assert is_zone_active(office, _at(12)) is True
    assert is_zone_active(office, _at(8)) is False
    assert is_zone_active(office, _at(18)) is False


def test_no_window_means_always_armed():
    """The default. A zone with no hours must never be treated as a zone with empty hours."""
    assert is_zone_active(_zone(None), _at(3)) is True
    assert is_zone_active(_zone(None), _at(15)) is True


def test_an_unparseable_window_degrades_to_always_armed_at_runtime():
    """Fail loud at configuration, fail *safe* at runtime — see validate_active_hours.

    A malformed window reaching the frame loop must not disarm the zone: an operator who
    wanted a guarded area gets one that over-alerts, not one that is silently off.
    """
    assert is_zone_active(_zone("22-05"), _at(12)) is True
    assert is_zone_active(_zone("not a window"), _at(12)) is True


# ------------------------------------------------------------------ configuration edge
def test_a_malformed_window_is_rejected_when_it_is_configured():
    """The counterpart to the test above, and the reason there are two parsers.

    At runtime a typo degrades to always-active. At configuration time that same silent
    degradation would hand the operator a zone alerting around the clock while the form
    said 22:00-05:00, so here it raises.
    """
    for bad in ("22-05", "25:00-05:00", "22:00", "", "22:00-05:61"):
        with pytest.raises(ValueError):
            validate_active_hours(bad)


def test_a_zero_length_window_is_rejected():
    """'22:00-22:00' reads as a one-minute window — configured, but alerting essentially never."""
    with pytest.raises(ValueError, match="same time"):
        validate_active_hours("22:00-22:00")


def test_a_valid_window_is_normalized():
    assert validate_active_hours(" 9:5-17:00 ") == "09:05-17:00"
    assert validate_active_hours("22:00-05:00") == "22:00-05:00"


def test_camera_create_rejects_a_malformed_window():
    with pytest.raises(ValidationError):
        CameraCreate(
            name="Lobby",
            source_type="demo",
            source="clip.mp4",
            intrusion_enabled=True,
            zone_polygon=SQUARE,
            zone_active_hours="10pm to 5am",
        )


def test_an_empty_window_string_means_always_armed():
    """A cleared form field posts "", which must mean "no window", not a parse error."""
    payload = CameraCreate(
        name="Lobby",
        source_type="demo",
        source="clip.mp4",
        intrusion_enabled=True,
        zone_polygon=SQUARE,
        zone_active_hours="",
    )
    assert payload.zone_active_hours is None


# ------------------------------------------------------------------ zone manager wiring
def test_a_dormant_zone_reports_no_hits():
    """`zones_containing` is what the intrusion rule calls, so the window has to bite here."""
    manager = ZoneManager()
    manager.set_zones("cam_1", [_zone("22:00-05:00")])

    inside_at_night = manager.zones_containing("cam_1", 50, 50, 100, 100, now=_at(23))
    inside_at_noon = manager.zones_containing("cam_1", 50, 50, 100, 100, now=_at(12))

    assert [z.name for z in inside_at_night] == ["restricted_zone"]
    assert inside_at_noon == []


def test_active_only_false_still_returns_a_dormant_zone():
    """Drawing the polygon on the preview must not depend on the time of day."""
    manager = ZoneManager()
    manager.set_zones("cam_1", [_zone("22:00-05:00")])

    hits = manager.zones_containing("cam_1", 50, 50, 100, 100, now=_at(12), active_only=False)

    assert [z.name for z in hits] == ["restricted_zone"]


# ------------------------------------------------------------------ end-to-end wiring
def test_the_window_reaches_the_zone_manager_from_a_camera_payload(service):
    """The gap this closes: `parse_active_hours` and `Zone.active_hours` both already
    existed, but nothing carried a window from the form to the rule engine, so intrusion
    fired around the clock while the PRD's story is explicitly about night."""
    payload = CameraCreate(
        name="Yard",
        source_type="rtsp",
        source="rtsp://x",
        intrusion_enabled=True,
        zone_polygon=SQUARE,
        zone_active_hours="22:00-05:00",
    )
    service._apply_zones("cam_yard", payload)

    assert service.zone_manager.zones_for("cam_yard")[0].active_hours == "22:00-05:00"


def test_the_window_survives_a_restart(service, monkeypatch, tmp_path):
    """The window lives in the persisted registry, so a restart must not silently arm a
    night-only zone around the clock.

    The worker start is stubbed out: this is about what `_restore_registry` reconstructs,
    and letting it dial a bogus RTSP URL would add a thread and a timeout to the assertion.
    """
    monkeypatch.setattr(service.manager, "add_camera", lambda **kwargs: None)
    service.add_camera(
        CameraCreate(
            name="Yard",
            source_type="rtsp",
            source="rtsp://x",
            intrusion_enabled=True,
            zone_polygon=SQUARE,
            zone_active_hours="22:00-05:00",
        )
    )

    monkeypatch.setattr(monitoring_module, "IncidentDetector", lambda: IncidentDetector(model_path=None))
    monkeypatch.setattr(monitoring_module.MultiStreamManager, "add_camera", lambda self, **kwargs: None)
    revived = MonitoringService()
    try:
        restored = next(iter(revived._cameras.values()))
        assert restored.zone_active_hours == "22:00-05:00"
        assert revived.zone_manager.zones_for(restored.id)[0].active_hours == "22:00-05:00"
    finally:
        revived.shutdown()


def test_clearing_a_camera_forgets_its_window_too(service):
    """`clear_zones` is what actually forgets a camera; a stale window would outlive it."""
    payload = CameraCreate(
        name="Yard",
        source_type="rtsp",
        source="rtsp://x",
        intrusion_enabled=True,
        zone_polygon=SQUARE,
        zone_active_hours="22:00-05:00",
    )
    service._apply_zones("cam_yard", payload)

    off = CameraCreate(name="Yard", source_type="rtsp", source="rtsp://x", intrusion_enabled=False)
    service._apply_zones("cam_yard", off)

    assert service.zone_manager.zones_for("cam_yard") == []


# ------------------------------------------------------------------ eval report
def _history(tmp_path, rows: list[dict]) -> AlertHistory:
    """A history file built from explicit rows, so every figure below is hand-checkable."""
    history = AlertHistory(path=str(tmp_path / "history.jsonl"))
    for row in rows:
        history.record(
            alert_id=row.get("alert_id", "a1"),
            camera_id=row.get("camera_id", "cam_1"),
            label=row.get("label", "fall_detection"),
            operator_verdict=row["operator_verdict"],
            reason=row.get("reason"),
            vlm_verdict=row.get("vlm_verdict"),
            vlm_confidence=row.get("vlm_confidence"),
        )
    return history


def test_an_empty_history_reports_nothing_rather_than_zero_precision(tmp_path):
    """0% precision would read as "every alert was wrong" on a system nobody has used yet."""
    report = build_report(_history(tmp_path, []))

    assert report.reviewed_rows == 0
    assert report.rule_precision is None
    assert report.false_alarm_rate is None
    assert any("No reviewed alerts yet" in w for w in report.warnings)


def test_precision_and_false_alarm_rate_come_from_operator_labels(tmp_path):
    rows = [{"operator_verdict": "real"}] * 3 + [{"operator_verdict": "false"}]
    report = build_report(_history(tmp_path, rows))

    assert report.reviewed_rows == 4
    assert report.real_incidents == 3
    assert report.rule_precision == 0.75
    assert report.false_alarm_rate == 0.25


def test_unreviewed_rows_are_excluded_from_every_figure(tmp_path):
    """An alert nobody judged is not evidence — counting it either way invents data."""
    history = _history(tmp_path, [{"operator_verdict": "real"}, {"operator_verdict": "false"}])
    # A row with no verdict, as `acknowledge` deliberately never writes one.
    history.record(alert_id="a3", camera_id="cam_1", label="fall_detection", operator_verdict="")

    report = build_report(history)

    assert report.total_rows == 3
    assert report.reviewed_rows == 2
    assert report.rule_precision == 0.5


def test_the_report_never_produces_a_recall_number(tmp_path):
    """Recall is not computable from alerts that fired, so no field may carry one.

    A missed incident never became an alert, so any "recall" from this dataset would divide
    by the alerts that fired and land near 1.0 — a fabricated number, not a weak one.
    """
    report = build_report(_history(tmp_path, [{"operator_verdict": "real"}] * 5))

    assert "recall" not in report.model_dump()
    assert "not computable" in report.recall_note


def test_per_label_precision_is_reported_separately(tmp_path):
    """A pooled number hides one bad label behind two good ones."""
    rows = [
        {"label": "fall_detection", "operator_verdict": "real"},
        {"label": "fall_detection", "operator_verdict": "real"},
        {"label": "abandoned_object", "operator_verdict": "false"},
        {"label": "abandoned_object", "operator_verdict": "false"},
    ]
    report = build_report(_history(tmp_path, rows))
    by_label = {row.label: row for row in report.by_label}

    assert by_label["fall_detection"].precision == 1.0
    assert by_label["abandoned_object"].precision == 0.0
    assert by_label["fall_detection"].low_sample is True


def test_the_comparison_counts_real_incidents_a_gating_vlm_would_have_hidden(tmp_path):
    """The figure that decides whether triage should ever suppress. See src/services/vlm.py.

    Only knowable because the alerts were shown anyway: the model called a confirmed-real
    fall a false positive, and the guard's label is on record to prove it.
    """
    rows = [
        {"operator_verdict": "real", "vlm_verdict": "false_positive"},
        {"operator_verdict": "false", "vlm_verdict": "false_positive"},
        {"operator_verdict": "false", "vlm_verdict": "false_positive"},
        {"operator_verdict": "real", "vlm_verdict": "real"},
    ]
    report = build_report(_history(tmp_path, rows))
    kept = next(c for c in report.comparisons if c.policy == "uncertain_kept")

    assert kept.baseline_precision == 0.5
    assert kept.alerts_shown == 1
    assert kept.precision == 1.0
    assert kept.false_alarms_suppressed == 2
    assert kept.real_incidents_suppressed == 1
    assert any("would have hidden up to 1" in w for w in report.warnings)


def test_the_uncertain_policy_changes_what_gets_hidden(tmp_path):
    """Both policies are reported because the choice between them is the actual decision."""
    rows = [
        {"operator_verdict": "real", "vlm_verdict": "uncertain"},
        {"operator_verdict": "false", "vlm_verdict": "uncertain"},
        {"operator_verdict": "false", "vlm_verdict": "false_positive"},
    ]
    report = build_report(_history(tmp_path, rows))
    kept = next(c for c in report.comparisons if c.policy == "uncertain_kept")
    dropped = next(c for c in report.comparisons if c.policy == "uncertain_dropped")

    assert (kept.false_alarms_suppressed, kept.real_incidents_suppressed) == (1, 0)
    assert (dropped.false_alarms_suppressed, dropped.real_incidents_suppressed) == (2, 1)


def test_the_comparison_baseline_is_the_triaged_subset_only(tmp_path):
    """Comparing a triaged subset against a whole-file baseline would price two populations.

    Here the untriaged rows are all false alarms; if they leaked into the baseline its
    precision would read 0.25 instead of the triaged subset's own 0.5.
    """
    rows = [
        {"operator_verdict": "real", "vlm_verdict": "real"},
        {"operator_verdict": "false", "vlm_verdict": "false_positive"},
        {"operator_verdict": "false"},
        {"operator_verdict": "false"},
    ]
    report = build_report(_history(tmp_path, rows))
    kept = next(c for c in report.comparisons if c.policy == "uncertain_kept")

    assert report.triage_coverage == 2
    assert kept.triaged_rows == 2
    assert kept.baseline_precision == 0.5
    assert any("covers that subset" in w for w in report.warnings)


def test_no_triaged_rows_means_no_comparison_at_all(tmp_path):
    """Rather than a table of zeroes that reads as "the VLM filtered nothing"."""
    report = build_report(_history(tmp_path, [{"operator_verdict": "real"}] * 3))

    assert report.comparisons == []
    assert report.triage_agreement is None
    assert any("No alert carries a VLM verdict" in w for w in report.warnings)


def test_uncertain_counts_as_disagreement(tmp_path):
    """Folding abstentions into "agreed" would flatter a model that mostly abstains."""
    rows = [
        {"operator_verdict": "real", "vlm_verdict": "real"},
        {"operator_verdict": "real", "vlm_verdict": "uncertain"},
    ]
    report = build_report(_history(tmp_path, rows))

    assert report.triage_agreement == 0.5


def test_filtering_by_camera_isolates_one_camera(tmp_path):
    """Cameras watch different scenes; a pooled rate can hide one bad camera."""
    rows = [
        {"camera_id": "cam_1", "operator_verdict": "real"},
        {"camera_id": "cam_2", "operator_verdict": "false"},
        {"camera_id": "cam_2", "operator_verdict": "false"},
    ]
    history = _history(tmp_path, rows)

    # 0.3333, not 1/3 — `_precision` rounds to 4dp, since trailing float noise in a
    # percentage printed to one decimal place is not information.
    assert build_report(history).rule_precision == 0.3333
    assert build_report(history, camera_id="cam_1").rule_precision == 1.0
    assert build_report(history, camera_id="cam_2").rule_precision == 0.0


def test_a_missing_history_file_reports_cleanly(tmp_path):
    """The report runs on a fresh clone, before anyone has labelled anything."""
    report = build_report(AlertHistory(path=str(tmp_path / "nope.jsonl")))

    assert report.total_rows == 0
    assert report.rule_precision is None
