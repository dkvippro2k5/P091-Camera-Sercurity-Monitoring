"""Print the PRD §4 evaluation report from operator-labelled alert history.

    python scripts/report_eval.py                      # text to stdout
    python scripts/report_eval.py --camera cam_ab12cd   # one camera
    python scripts/report_eval.py --json               # machine-readable
    python scripts/report_eval.py --markdown docs/gate1/eval.md

The numbers come from `data/vlm_history.jsonl`, which is appended to every time a guard
presses "Confirm real" or "False alarm". No model is loaded and no network call is made, so
this is safe to run against a live system.

Note what is deliberately absent: a recall figure. See `src/services/eval_report.py` — a
recall computed from alerts that fired would be fabricated, so the report prints the reason
instead of a number.
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Windows console encoding for the box-drawing and ▲ characters below
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.alert_history import AlertHistory  # noqa: E402
from src.services.eval_report import EvalReport, build_report  # noqa: E402


def _pct(value: float | None) -> str:
    """Percentages with an explicit dash for 'not computable', never a misleading 0%."""
    return "—" if value is None else f"{value * 100:.1f}%"


def _render_text(report: EvalReport) -> str:
    lines = [
        "=" * 72,
        "  ALERT QUALITY REPORT — PRD §4",
        "=" * 72,
        "",
        f"  Reviewed alerts     {report.reviewed_rows} of {report.total_rows} recorded"
        f"   ·   {report.cameras} camera(s)",
    ]
    if report.first_recorded:
        lines.append(f"  Collected           {report.first_recorded[:19]} → {report.last_recorded[:19]}")

    lines += [
        "",
        "-" * 72,
        "  RULE ENGINE ALONE",
        "-" * 72,
        f"  Real incidents      {report.real_incidents}",
        f"  False alarms        {report.false_alarms}",
        f"  Precision           {_pct(report.rule_precision)}   (of alerts shown, how many were real)",
        f"  False-alarm rate    {_pct(report.false_alarm_rate)}",
        "",
        f"  Recall              not measured — {report.recall_note}",
    ]

    if report.by_label:
        lines += ["", "-" * 72, "  BY INCIDENT LABEL", "-" * 72]
        lines.append(f"  {'label':<20}{'reviewed':>10}{'real':>8}{'false':>8}{'precision':>12}")
        for row in report.by_label:
            flag = "  (low sample)" if row.low_sample else ""
            lines.append(
                f"  {row.label:<20}{row.reviewed:>10}{row.real:>8}{row.false_alarms:>8}{_pct(row.precision):>12}{flag}"
            )

    return "\n".join(lines + _render_comparison(report) + _render_warnings(report))


def _render_comparison(report: EvalReport) -> list[str]:
    """The rules-alone vs rules+VLM section, or why it is missing."""
    if not report.comparisons:
        return []

    lines = [
        "",
        "-" * 72,
        "  RULES + VLM TRIAGE  (counterfactual — nothing was actually suppressed)",
        "-" * 72,
        f"  Triaged             {report.triage_coverage} of {report.reviewed_rows} reviewed alerts",
        f"  Agreement w/ guard  {_pct(report.triage_agreement)}   ('uncertain' counts as disagreement)",
    ]
    for comparison in report.comparisons:
        policy = "uncertain → keep the alert" if comparison.policy == "uncertain_kept" else "uncertain → suppress"
        lines += [
            "",
            f"  If the VLM gated the dashboard, {policy}:",
            f"    precision           {_pct(comparison.baseline_precision)} → {_pct(comparison.precision)}",
            f"    false alarms hidden  {comparison.false_alarms_suppressed}"
            f"   ({_pct(comparison.false_alarm_reduction)} of them)",
            f"    REAL incidents hidden {comparison.real_incidents_suppressed}",
        ]
    return lines


def _render_warnings(report: EvalReport) -> list[str]:
    if not report.warnings:
        return [""]
    lines = ["", "-" * 72, "  READ THIS WITH THE NUMBERS", "-" * 72]
    for warning in report.warnings:
        lines.append(f"  ▲ {warning}")
    return lines + [""]


def _render_markdown(report: EvalReport) -> str:
    """Markdown for `docs/gate1/`, so the figures can be pasted into the report."""
    lines = [
        "# Alert quality — PRD §4",
        "",
        f"Computed from {report.reviewed_rows} operator-reviewed alerts across {report.cameras} camera(s).",
        "",
        "## Rule engine alone",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Real incidents | {report.real_incidents} |",
        f"| False alarms | {report.false_alarms} |",
        f"| Precision | {_pct(report.rule_precision)} |",
        f"| False-alarm rate | {_pct(report.false_alarm_rate)} |",
        "| Recall | not measured |",
        "",
        f"> {report.recall_note}",
        "",
    ]
    if report.by_label:
        lines += ["## By label", "", "| Label | Reviewed | Real | False | Precision |", "|---|---|---|---|---|"]
        lines += [
            f"| {r.label} | {r.reviewed} | {r.real} | {r.false_alarms} | {_pct(r.precision)} |" for r in report.by_label
        ]
        lines.append("")

    if report.comparisons:
        lines += [
            "## Rules + VLM triage (counterfactual)",
            "",
            f"Triaged {report.triage_coverage} of {report.reviewed_rows} reviewed alerts. "
            f"Agreement with the guard: {_pct(report.triage_agreement)}.",
            "",
            "| `uncertain` policy | Precision | False alarms hidden | Real incidents hidden |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {c.policy.replace('_', ' ')} | {_pct(c.baseline_precision)} → {_pct(c.precision)} "
            f"| {c.false_alarms_suppressed} ({_pct(c.false_alarm_reduction)}) | {c.real_incidents_suppressed} |"
            for c in report.comparisons
        ]
        lines.append("")

    if report.warnings:
        lines += ["## Caveats", ""] + [f"- {w}" for w in report.warnings] + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Alert precision / false-alarm report from operator labels.")
    parser.add_argument("--camera", help="Restrict to one camera id.")
    parser.add_argument("--history", help="Path to a history JSONL (defaults to data/vlm_history.jsonl).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--markdown", metavar="PATH", help="Also write a Markdown version to PATH.")
    args = parser.parse_args()

    history = AlertHistory(args.history) if args.history else AlertHistory()
    report = build_report(history, camera_id=args.camera)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(_render_text(report))

    if args.markdown:
        target = Path(args.markdown)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render_markdown(report), encoding="utf-8")
        print(f"Markdown written to {target}")

    # Exit 1 when there is nothing to report, so CI or a Make target can tell "no labels
    # collected yet" from "here are the numbers" without parsing stdout.
    return 0 if report.reviewed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
