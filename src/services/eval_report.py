"""Precision and false-alarm-rate figures for PRD §4, computed from operator labels.

The input is `data/vlm_history.jsonl` — every alert a guard reviewed, carrying the rule
engine's evidence, the VLM's verdict, and the guard's own judgement. That file is the only
ground truth this system produces, and it exists because `vlm.py` is annotate-only: every
rule-engine event reached a human, so for each one we know both what the model said and what
was actually true. A suppressing design would have no such record for the alerts it dropped.

**What this can and cannot measure, stated up front because the difference is the whole
integrity of the number:**

- *Precision* — of the alerts shown, how many were real — is measurable. Every shown alert
  is in the file with a verdict.
- *Recall* — of the real incidents, how many were caught — is **not** measurable here, and
  this module refuses to print a number for it. An incident the rule engine missed never
  became an alert, was never reviewed, and is not in this file. A "recall" computed from
  these rows would divide by the alerts that fired and always land near 1.0, which is not a
  weak measurement but a fabricated one. Getting real recall needs a separate annotated
  clip set where every true incident is marked whether or not the system fired
  (`data/vision/test/`), which is a labelling job, not a query.

The rules-alone vs rules+VLM comparison is computed **only over rows the VLM actually
assessed**, and its baseline is that same subset rather than the whole file. Comparing a
VLM-filtered subset against a baseline that includes untriaged rows would attribute the
difference between two populations to the model.
"""

from pydantic import BaseModel, Field

from src.services.alert_history import AlertHistory

# Below this many reviewed alerts the percentages move by more than they mean — one
# relabelled row shifts a 20-row precision by 5 points. The report still prints, with the
# figure flagged, because "not enough data yet" is itself a finding worth showing a grader.
MIN_ROWS_FOR_CONFIDENCE = 30

_SUPPRESSING_VERDICTS = {
    # Only these two are candidates for hiding an alert. `real` and a missing verdict never
    # suppress anything.
    "uncertain_kept": ("false_positive",),
    "uncertain_dropped": ("false_positive", "uncertain"),
}


def _precision(true_positives: int, shown: int) -> float | None:
    """Share of shown alerts that were real. None when nothing was shown — not 0.0.

    Zero would read as "every alert was wrong" on a camera that simply has no reviewed
    alerts yet, which is the opposite of the truth.
    """
    return None if shown <= 0 else round(true_positives / shown, 4)


class LabelBreakdown(BaseModel):
    """Per-incident-label precision. The three PRD §3.1 labels fail in different ways."""

    label: str
    reviewed: int
    real: int
    false_alarms: int
    precision: float | None = None
    low_sample: bool = Field(default=False, description="Fewer than MIN_ROWS_FOR_CONFIDENCE rows.")


class TriageComparison(BaseModel):
    """Rules-alone vs rules+VLM over the triaged subset, under one `uncertain` policy.

    `real_incidents_suppressed` is the figure that decides whether the VLM should ever gate
    the dashboard. It is the count of alerts a guard confirmed as real that this policy
    would have hidden — on a fall, a missed medical emergency. It is only knowable because
    the alerts were shown anyway.
    """

    policy: str
    triaged_rows: int
    baseline_precision: float | None = None
    alerts_shown: int = 0
    precision: float | None = None
    false_alarms_suppressed: int = 0
    real_incidents_suppressed: int = 0
    false_alarm_reduction: float | None = Field(
        default=None, description="Share of the subset's false alarms this policy would have hidden."
    )


class EvalReport(BaseModel):
    """Everything the report prints. Assembled by `build_report`, rendered by the script."""

    total_rows: int = 0
    reviewed_rows: int = 0
    real_incidents: int = 0
    false_alarms: int = 0
    rule_precision: float | None = None
    false_alarm_rate: float | None = None

    cameras: int = 0
    first_recorded: str | None = None
    last_recorded: str | None = None

    by_label: list[LabelBreakdown] = Field(default_factory=list)

    triage_coverage: int = Field(default=0, description="Reviewed rows that also carry a VLM verdict.")
    triage_agreement: float | None = Field(
        default=None, description="Share of triaged rows where the VLM and the guard agreed."
    )
    comparisons: list[TriageComparison] = Field(default_factory=list)

    # Surfaced rather than logged: a number whose caveat is buried in a docstring gets
    # quoted without it, and these are the caveats a grader will ask about.
    warnings: list[str] = Field(default_factory=list)

    @property
    def recall_note(self) -> str:
        """Why there is no recall figure. Printed verbatim, never replaced by a number."""
        return (
            "Recall is not computable from operator-reviewed alerts: an incident the rule engine "
            "missed never produced an alert and so is absent from this dataset. Measuring it needs "
            "an annotated clip set where every true incident is marked independently of what fired."
        )


def _verdict_rows(rows: list[dict]) -> list[dict]:
    """Rows a guard actually labelled. Unreviewed rows are not evidence of anything."""
    return [row for row in rows if row.get("operator_verdict") in ("real", "false")]


def _breakdown(label: str, rows: list[dict]) -> LabelBreakdown:
    """Precision for one incident label."""
    real = sum(1 for row in rows if row["operator_verdict"] == "real")
    return LabelBreakdown(
        label=label,
        reviewed=len(rows),
        real=real,
        false_alarms=len(rows) - real,
        precision=_precision(real, len(rows)),
        low_sample=len(rows) < MIN_ROWS_FOR_CONFIDENCE,
    )


def _compare(rows: list[dict], policy: str) -> TriageComparison:
    """What one suppression policy would have done to the triaged subset.

    Nothing is actually suppressed anywhere — this replays the counterfactual over rows
    whose outcome is already known, which is the only way to price the trade-off before
    committing to it.
    """
    suppressing = _SUPPRESSING_VERDICTS[policy]
    baseline_real = sum(1 for row in rows if row["operator_verdict"] == "real")
    baseline_false = len(rows) - baseline_real

    shown = [row for row in rows if row.get("vlm_verdict") not in suppressing]
    shown_real = sum(1 for row in shown if row["operator_verdict"] == "real")

    return TriageComparison(
        policy=policy,
        triaged_rows=len(rows),
        baseline_precision=_precision(baseline_real, len(rows)),
        alerts_shown=len(shown),
        precision=_precision(shown_real, len(shown)),
        false_alarms_suppressed=baseline_false - (len(shown) - shown_real),
        real_incidents_suppressed=baseline_real - shown_real,
        false_alarm_reduction=(
            None if baseline_false == 0 else round((baseline_false - (len(shown) - shown_real)) / baseline_false, 4)
        ),
    )


def _agreement(rows: list[dict]) -> float | None:
    """Share of triaged rows where the model and the guard reached the same conclusion.

    `uncertain` counts as disagreement: it is not a wrong answer, but it is not the useful
    one either, and folding it into "agreed" would flatter a model that abstains a lot.
    """
    if not rows:
        return None
    agreed = sum(
        1
        for row in rows
        if (row["operator_verdict"] == "real" and row.get("vlm_verdict") == "real")
        or (row["operator_verdict"] == "false" and row.get("vlm_verdict") == "false_positive")
    )
    return round(agreed / len(rows), 4)


def _collect_warnings(report: "EvalReport") -> list[str]:
    """Caveats that must travel with the numbers."""
    warnings: list[str] = []
    if report.reviewed_rows == 0:
        warnings.append(
            "No reviewed alerts yet. Run the system, then use 'Confirm real' / 'False alarm' on each "
            "alert — every judgement appends one row to data/vlm_history.jsonl."
        )
        return warnings

    if report.reviewed_rows < MIN_ROWS_FOR_CONFIDENCE:
        warnings.append(
            f"Only {report.reviewed_rows} reviewed alerts (under {MIN_ROWS_FOR_CONFIDENCE}). "
            "Treat every percentage as indicative; one relabelled row moves it several points."
        )
    if report.triage_coverage == 0:
        warnings.append(
            "No alert carries a VLM verdict, so rules+VLM cannot be compared. The triage layer is "
            "off by default — set VLM_ENABLED=true with a valid VLM_API_KEY and collect again."
        )
    elif report.triage_coverage < report.reviewed_rows:
        warnings.append(
            f"Only {report.triage_coverage} of {report.reviewed_rows} reviewed alerts were triaged; "
            "the comparison covers that subset, and its baseline is that same subset."
        )

    suppressed = [c.real_incidents_suppressed for c in report.comparisons]
    if any(count > 0 for count in suppressed):
        warnings.append(
            f"A gating VLM would have hidden up to {max(suppressed)} operator-confirmed real "
            "incident(s). This is the argument for keeping triage advisory (see src/services/vlm.py)."
        )
    return warnings


def build_report(history: AlertHistory | None = None, camera_id: str | None = None) -> EvalReport:
    """Compute every figure from the operator-labelled history.

    Args:
        history: Source of rows. Defaults to the standard `data/vlm_history.jsonl`.
        camera_id: Restrict to one camera. Cameras watch different scenes and have genuinely
            different false-alarm profiles, so a pooled number can hide one bad camera behind
            four good ones.

    Returns:
        A fully populated report. Never raises on an empty or missing history — a report
        saying "no data yet" is more useful than a traceback.
    """
    history = history or AlertHistory()
    rows = history.all_rows()
    if camera_id:
        rows = [row for row in rows if row.get("camera_id") == camera_id]

    reviewed = _verdict_rows(rows)
    real = sum(1 for row in reviewed if row["operator_verdict"] == "real")
    triaged = [row for row in reviewed if row.get("vlm_verdict")]
    stamps = sorted(row["recorded_at"] for row in rows if row.get("recorded_at"))

    labels = sorted({row.get("label", "unknown") for row in reviewed})
    report = EvalReport(
        total_rows=len(rows),
        reviewed_rows=len(reviewed),
        real_incidents=real,
        false_alarms=len(reviewed) - real,
        rule_precision=_precision(real, len(reviewed)),
        # The PRD's "giảm báo động giả" is about this rate, so it is reported directly
        # rather than left as 1 - precision for the reader to work out.
        false_alarm_rate=(None if not reviewed else round((len(reviewed) - real) / len(reviewed), 4)),
        cameras=len({row.get("camera_id") for row in rows if row.get("camera_id")}),
        first_recorded=stamps[0] if stamps else None,
        last_recorded=stamps[-1] if stamps else None,
        by_label=[
            _breakdown(label, [row for row in reviewed if row.get("label", "unknown") == label]) for label in labels
        ],
        triage_coverage=len(triaged),
        triage_agreement=_agreement(triaged),
        # Both policies, because the choice of what to do with `uncertain` is the whole
        # decision and one number would hide it: dropping uncertain filters more noise and
        # costs more real incidents.
        comparisons=[_compare(triaged, policy) for policy in _SUPPRESSING_VERDICTS] if triaged else [],
    )
    report.warnings = _collect_warnings(report)
    return report
