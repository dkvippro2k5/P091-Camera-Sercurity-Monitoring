"""Compare "VLM only" vs "VLM + LLM" (CascadingTriage) on the same case set.

See docs/plans/llm-reasoning-node.md for the full design. This measures whether the optional
LLM reasoning stage (src/services/llm_reasoner.py) improves accuracy enough to justify its
added latency/cost over the VLM-only baseline that ships when LLM_REASONING_ENABLED=false.

**VLM is called exactly once per case, not twice.** A naive version of this script would
build a separate `CompatibleVLMTriage` for arm A and a separate `CascadingTriage` (with its
own internal VLM client) for arm B, calling the vision model twice per case. That doubles the
per-case image-call cost for nothing, and — worse — the two arms could see different VLM
verdicts for the same case purely from model non-determinism (observed earlier with other
providers in this project's own benchmarking), which would confound "did the LLM help?" with
"did the VLM answer differently this time?". Instead: call the VLM once, use that verdict
directly as arm A's result, and feed the exact same verdict into `CascadingTriage` (via a tiny
wrapper that returns it without a second call) to get arm B's result — reusing the real
`CascadingTriage.assess()` merge/fallback logic unmodified, not a reimplementation of it here.

Usage:
    VLM_API_KEY=sk-... LLM_API_KEY=sk-... python -m src.benchmark.run_llm_reasoning_compare
    VLM_API_KEY=sk-... LLM_API_KEY=sk-... python -m src.benchmark.run_llm_reasoning_compare fall_detection
"""

import json
import os
import sys
from pathlib import Path

from src.services.llm_reasoner import CompatibleLLMReasoner
from src.services.vlm import CascadingTriage, CompatibleVLMTriage, TriageRequest, TriageVerdict

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_FILE = REPO_ROOT / "tests" / "fixtures" / "benchmark" / "cases.jsonl"
IMAGES_DIR = REPO_ROOT / "tests" / "fixtures" / "benchmark" / "images"

QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.7-flash"

# UNVERIFIED — no confirmed official per-token price was found for qwen3.7-flash at the time
# this was written. Left None on purpose rather than printing a fabricated $ figure; fill in
# from the QwenCloud console (home.qwencloud.com) if you want cost_usd instead of just token
# counts. total_tokens alone is still the honest, always-available cost proxy — see
# docs/benchmark/pipeline-benchmark.md's own "dùng cost_usd, không dùng total_tokens" note,
# which assumes a verified price is available; here one is not, so tokens are what's reported.
QWEN_PRICE_PER_1M_TOKENS_USD: float | None = None


class _FixedVLM:
    """Stands in for a VLM client that already ran -- lets CascadingTriage reuse its verdict
    without a second network call. See module docstring for why this matters."""

    def __init__(self, verdict: TriageVerdict) -> None:
        self._verdict = verdict

    def assess(self, request: TriageRequest) -> TriageVerdict:
        return self._verdict


def load_cases() -> list[dict]:
    cases = []
    with open(CASES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_request(case: dict) -> TriageRequest:
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
    label_filter = sys.argv[1] if len(sys.argv) > 1 else None

    vlm_key = os.environ.get("VLM_API_KEY", "")
    llm_key = os.environ.get("LLM_API_KEY", "")
    if not vlm_key:
        print("VLM_API_KEY not set. Export it and re-run.", file=sys.stderr)
        sys.exit(1)
    if not llm_key:
        print("LLM_API_KEY not set. Export it and re-run.", file=sys.stderr)
        sys.exit(1)

    vlm_client = CompatibleVLMTriage(api_key=vlm_key, base_url=QWEN_BASE_URL, model=QWEN_MODEL)
    reasoner = CompatibleLLMReasoner(api_key=llm_key, base_url=QWEN_BASE_URL, model=QWEN_MODEL)

    cases = load_cases()
    if label_filter:
        cases = [c for c in cases if c["label"] == label_filter]
        if not cases:
            print(f"No cases with label={label_filter!r}.", file=sys.stderr)
            sys.exit(1)

    rows = []
    for case in cases:
        request = build_request(case)

        vlm_verdict = vlm_client.assess(request)  # the one and only VLM call for this case
        cascading = CascadingTriage(vlm=_FixedVLM(vlm_verdict), reasoner=reasoner)
        combined_verdict = cascading.assess(request)  # reuses real CascadingTriage merge logic

        rows.append(
            {
                "case_id": case["case_id"],
                "expected": case["expected_verdict"],
                "vlm_only": vlm_verdict,
                "vlm_plus_llm": combined_verdict,
            }
        )

    _print_report(rows)


def _print_report(rows: list[dict]) -> None:
    print(f"\n{'case_id':<32} {'expected':<16} {'VLM only':<48} {'VLM + LLM':<48}")
    print("-" * (32 + 16 + 48 + 48))
    for row in rows:
        vlm = row["vlm_only"]
        combined = row["vlm_plus_llm"]
        expected = row["expected"]
        line = f"{row['case_id']:<32} {expected:<16} "
        line += _cell(vlm, expected) + " " + _cell(combined, expected)
        print(line)

    print()
    for arm_key, arm_label in [("vlm_only", "VLM only"), ("vlm_plus_llm", "VLM + LLM")]:
        total = len(rows)
        matched = sum(1 for r in rows if r[arm_key].verdict == r["expected"])
        errors = sum(1 for r in rows if r[arm_key].error)
        avg_latency = sum(r[arm_key].latency_ms for r in rows) / total if total else 0.0
        avg_tokens = sum(r[arm_key].total_tokens for r in rows) / total if total else 0.0
        print(
            f"{arm_label}: {matched}/{total} matched expected, {errors} error(s), "
            f"avg latency {avg_latency:.0f}ms, avg tokens {avg_tokens:.0f}{_cost_note(avg_tokens)}"
        )

    if rows:
        added_latency = sum(r["vlm_plus_llm"].latency_ms - r["vlm_only"].latency_ms for r in rows) / len(rows)
        added_tokens = sum(r["vlm_plus_llm"].total_tokens - r["vlm_only"].total_tokens for r in rows) / len(rows)
        print(
            f"\nLLM stage added: avg +{added_latency:.0f}ms latency, "
            f"+{added_tokens:.0f} tokens per case{_cost_note(added_tokens)}"
        )

    for row in rows:
        for arm_key, arm_label in [("vlm_only", "VLM only"), ("vlm_plus_llm", "VLM + LLM")]:
            err = row[arm_key].error
            if err:
                print(f"\n[{row['case_id']} / {arm_label}]\n{err}")


def _cell(verdict: TriageVerdict, expected: str) -> str:
    mark = "OK " if verdict.verdict == expected else "X  "
    if verdict.error:
        return f"{mark}ERROR: {verdict.error[:38]:<38}"
    return f"{mark}{verdict.verdict:<14} conf={verdict.confidence:.2f} {verdict.latency_ms:.0f}ms tok={verdict.total_tokens:<38}"


def _cost_note(avg_tokens: float) -> str:
    """", ~$0.00012" when a price is configured, otherwise "" (see QWEN_PRICE_PER_1M_TOKENS_USD)."""
    if QWEN_PRICE_PER_1M_TOKENS_USD is None:
        return ""
    cost = avg_tokens / 1_000_000 * QWEN_PRICE_PER_1M_TOKENS_USD
    return f", ~${cost:.5f}"


if __name__ == "__main__":
    main()
