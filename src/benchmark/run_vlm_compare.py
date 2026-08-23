"""Run the TriageRequest set through Qwen 3.7 Flash and report accuracy/latency.

Reuses `CompatibleVLMTriage` unmodified (see `src/services/vlm.py`) pointed at QwenCloud's
DashScope-compatible endpoint. No new client, no new schema.

2026-08-14: renamed from OpenAIVLMTriage to CompatibleVLMTriage upstream (develop commit
0156f99) when a teammate landed the same Qwen migration independently -- config.py/vlm.py
now default to Qwen directly (llm_api_key/vlm_api_key, no more OPENAI_API_KEY fallback).

2026-08-14: this used to compare several OpenRouter models side by side (Gemini, Nemotron,
gpt-5.6-luna, Qwen2.5-VL). Narrowed to Qwen 3.7 Flash only, by request — VLM_MODEL/
VLM_BASE_URL in .env now standardize on it for the real system too, so this script measures
that one model against the case set rather than shopping between vendors.

Reads `tests/fixtures/benchmark/cases.jsonl`. A case is either one still (`image_file`,
pre_count=0 — no BEFORE/AFTER split) or a chronological sequence (`image_files`, with
`pre_count` marking how many lead the trigger moment) built the same way MonitoringService
frames a real alert. See the case file's `note` field for per-case caveats; one case is
flagged as carrying a baked-in annotation and is kept only to smoke-test the pipeline, not
to score accuracy.

Usage:
    VLM_API_KEY=sk-... python -m src.benchmark.run_vlm_compare
    VLM_API_KEY=sk-... python -m src.benchmark.run_vlm_compare fall_detection  # filter by label
"""

import json
import os
import sys
from pathlib import Path

from src.services.vlm import CompatibleVLMTriage, TriageRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_FILE = REPO_ROOT / "tests" / "fixtures" / "benchmark" / "cases.jsonl"
IMAGES_DIR = REPO_ROOT / "tests" / "fixtures" / "benchmark" / "images"

# Single entry by design -- see module docstring. (display_name, model_id, base_url, api_key_env).
MODEL_SPECS = [
    (
        "qwen3.7-flash",
        "qwen3.7-flash",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "VLM_API_KEY",
    ),
]


def load_cases() -> list[dict]:
    cases = []
    with open(CASES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_request(case: dict) -> TriageRequest:
    # image_files (list, chronological) covers multi-frame before/after cases; image_file
    # (single, legacy) covers the original one-still abandoned_object cases.
    image_files = case.get("image_files") or [case["image_file"]]
    frames = [(IMAGES_DIR / name).read_bytes() for name in image_files]
    return TriageRequest(
        label=case["label"],
        reason=case.get("reason"),
        zone_name=case.get("zone_name"),
        camera_name=case.get("camera_name", ""),
        frames=frames,
        pre_count=case.get("pre_count", 0),
    )


def main() -> None:
    # Optional: python -m src.benchmark.run_vlm_compare fall_detection
    # Filters cases by `label` so a single incident type can be re-run without touching
    # cases.jsonl -- useful while a specific model/case pairing is under investigation and
    # the other cases would just spend rate-limit budget for no new information.
    label_filter = sys.argv[1] if len(sys.argv) > 1 else None

    clients: dict[str, CompatibleVLMTriage] = {}
    missing_keys: list[str] = []
    for display_name, model_id, base_url, key_env in MODEL_SPECS:
        api_key = os.environ.get(key_env, "")
        if not api_key:
            missing_keys.append(f"{key_env} (needed for {display_name})")
            continue
        clients[display_name] = CompatibleVLMTriage(api_key=api_key, model=model_id, base_url=base_url)

    if missing_keys:
        print("Skipping models with no key set:", file=sys.stderr)
        for line in missing_keys:
            print(f"  - {line}", file=sys.stderr)
    if not clients:
        print("No API keys set at all. Export at least one and re-run.", file=sys.stderr)
        sys.exit(1)

    cases = load_cases()
    if label_filter:
        cases = [c for c in cases if c["label"] == label_filter]
        if not cases:
            print(f"No cases with label={label_filter!r}.", file=sys.stderr)
            sys.exit(1)
    models = list(clients.keys())

    rows = []
    for case in cases:
        request = build_request(case)
        row = {"case_id": case["case_id"], "expected": case["expected_verdict"]}
        for model in models:
            verdict = clients[model].assess(request)
            row[model] = {
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "latency_ms": verdict.latency_ms,
                "error": verdict.error,
                "match": verdict.verdict == case["expected_verdict"],
            }
        rows.append(row)

    _print_report(rows, models)


def _print_report(rows: list[dict], models: list[str]) -> None:
    print(f"\n{'case_id':<32} {'expected':<16} " + " ".join(f"{m.split('/')[-1]:<40}" for m in models))
    print("-" * (32 + 16 + 41 * len(models)))
    for row in rows:
        line = f"{row['case_id']:<32} {row['expected']:<16} "
        for model in models:
            r = row[model]
            mark = "OK " if r["match"] else "X  "
            cell = f"{mark}{r['verdict']:<14} conf={r['confidence']:.2f} {r['latency_ms']:.0f}ms"
            if r["error"]:
                cell = f"{mark}ERROR: {r['error'][:40]}"
            line += f"{cell:<41}"
        print(line)

    print()
    for model in models:
        total = len(rows)
        matched = sum(1 for r in rows if r[model]["match"])
        errors = sum(1 for r in rows if r[model]["error"])
        avg_latency = sum(r[model]["latency_ms"] for r in rows) / total if total else 0.0
        print(f"{model}: {matched}/{total} matched expected, {errors} error(s), avg latency {avg_latency:.0f}ms")

    # Full error text, un-truncated -- the table above cuts each cell to 40 chars for
    # alignment, which hides the part of the message that actually explains the failure
    # (OpenRouter puts the real reason in the JSON body, past where the table cuts off).
    for row in rows:
        for model in models:
            err = row[model]["error"]
            if err:
                print(f"\n[{row['case_id']} / {model}]\n{err}")


if __name__ == "__main__":
    main()
