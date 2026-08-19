"""Opt-in regression runner for the OpenAI fall-triage path.

Listing and validating cases is the default. Network calls only occur with ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.services.vlm import OpenAIVLMTriage, TriageRequest  # noqa: E402

DEFAULT_CASES = Path(__file__).with_name("vlm_cases.json")
VALID_VERDICTS = {"real", "false_positive", "uncertain"}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("case manifest must be a non-empty JSON array")

    required = {"id", "label", "source", "frame_times_sec", "pre_count", "expected_verdicts"}
    seen: set[str] = set()
    for case in payload:
        missing = required - set(case)
        if missing:
            raise ValueError(f"case is missing required fields: {sorted(missing)}")
        case_id = str(case["id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)

        times = case["frame_times_sec"]
        pre_count = case["pre_count"]
        expected = set(case["expected_verdicts"])
        if not isinstance(times, list) or len(times) < 2 or any(float(t) < 0 for t in times):
            raise ValueError(f"{case_id}: frame_times_sec must contain at least two non-negative values")
        if not isinstance(pre_count, int) or not 0 < pre_count < len(times):
            raise ValueError(f"{case_id}: pre_count must split the frame sequence")
        if not expected or not expected <= VALID_VERDICTS:
            raise ValueError(f"{case_id}: expected_verdicts contains an invalid verdict")
    return payload


def _read_jpeg_frames(source: Path, times_sec: list[float], jpeg_quality: int = 88) -> list[bytes]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not decode video: {source}")

    frames: list[bytes] = []
    try:
        for timestamp in times_sec:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"could not read {timestamp:.2f}s from {source}")
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not encoded:
                raise ValueError(f"could not JPEG-encode {timestamp:.2f}s from {source}")
            frames.append(buffer.tobytes())
    finally:
        capture.release()
    return frames


def _select_cases(cases: list[dict[str, Any]], selected_ids: list[str]) -> list[dict[str, Any]]:
    if not selected_ids:
        return cases
    selected = [case for case in cases if case["id"] in selected_ids]
    missing = set(selected_ids) - {case["id"] for case in selected}
    if missing:
        raise ValueError(f"unknown case id(s): {', '.join(sorted(missing))}")
    return selected


def _print_case_summary(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        source = ROOT / case["source"]
        state = "ready" if source.is_file() else "MISSING"
        expected = ", ".join(case["expected_verdicts"])
        print(
            f"{case['id']}: {len(case['frame_times_sec'])} frames, "
            f"expected [{expected}], source {state}: {case['source']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="JSON case manifest")
    parser.add_argument("--case", action="append", default=[], help="run only this case id")
    parser.add_argument("--execute", action="store_true", help="make billed OpenAI API calls")
    parser.add_argument("--model", help="override VLM_MODEL for this run")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()

    try:
        cases = _select_cases(_load_cases(args.cases.resolve()), args.case)
        _print_case_summary(cases)
        missing_sources = [case["source"] for case in cases if not (ROOT / case["source"]).is_file()]
        if missing_sources:
            raise FileNotFoundError("missing source video(s): " + ", ".join(missing_sources))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"case validation failed: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        print("Dry run only. Add --execute to make billed OpenAI API calls.")
        return 0

    settings = Settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is required with --execute.", file=sys.stderr)
        return 2

    model = args.model or settings.vlm_model
    client = OpenAIVLMTriage(
        api_key=settings.openai_api_key,
        model=model,
        timeout_sec=settings.vlm_timeout_sec,
    )
    results: list[dict[str, Any]] = []
    for case in cases:
        frames = _read_jpeg_frames(
            ROOT / case["source"], [float(value) for value in case["frame_times_sec"]]
        )
        verdict = client.assess(
            TriageRequest(
                label=case["label"],
                reason=case.get("reason"),
                camera_name=case.get("camera_name", case["id"]),
                frames=frames,
                pre_count=case["pre_count"],
            )
        )
        passed = verdict.error is None and verdict.verdict in case["expected_verdicts"]
        result = {
            "id": case["id"],
            "expected_verdicts": case["expected_verdicts"],
            "passed": passed,
            **verdict.model_dump(),
        }
        results.append(result)
        print(
            f"{case['id']}: {verdict.verdict} ({verdict.confidence:.2f}), "
            f"{verdict.latency_ms:.0f} ms - {'PASS' if passed else 'FAIL'}"
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "passed": sum(1 for result in results if result["passed"]),
        "total": len(results),
        "cases": results,
    }
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
