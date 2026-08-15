"""Third-stage triage: a text-only LLM giving a second opinion on the VLM's own verdict.

This is the optional Vision -> VLM -> LLM extension (docs/plans/llm-reasoning-node.md). It
plugs into `CascadingTriage` (src/services/vlm.py), which runs `CompatibleVLMTriage.assess()`
first and hands this module the resulting `TriageVerdict` alongside the original
`TriageRequest` — this module never sees the images.

**Text-only by design, not by omission.** A single-frame-worth of pixels means nothing to a
text LLM without a way to correlate coordinates against an image it cannot see — the rule
engine already does that translation (bounding boxes -> AR/V_y/dH indicators) for exactly
this reason. Re-sending raw frames here would double the per-alert image-call cost for no
reasoning benefit; see the module docstring in `vlm.py` and the plan doc for the fuller
argument.

**Known limitation, accepted for the first cut, not hidden:** this stage receives the VLM's
`rationale`, which `vlm.py` deliberately keeps to one sentence for a guard skimming a screen,
not for feeding a second model. If this stage turns out not to add measurable value in the
benchmark (`src/benchmark/run_llm_reasoning_compare.py`), a thin `rationale` is the first
suspect — not a broken architecture. Likewise, `rule_evidence` carries a full three-indicator
breakdown for fall_detection but is comparatively bare for intrusion/abandoned_object (see
`src/vision/rule_engine.py`'s `_check_intrusion`/`_check_abandoned`), so this stage should be
expected to help fall_detection more than the other two labels until that gap is closed
separately.

Same shape as `vlm.py`: a protocol, a null client, and a real client that never raises.
"""

import json
import logging
import time
from typing import Protocol

from pydantic import BaseModel, Field

from src.services.vlm import TriageRequest, TriageVerdict, Verdict

logger = logging.getLogger(__name__)

# CÔ ĐỊNH — cùng nội dung với _SYSTEM_PROMPT trong vlm.py (định nghĩa 3 nhãn), cộng thêm phần
# chống echo chamber: rule_evidence là bằng chứng chính, vlm_assessment chỉ là một giả thuyết
# cần kiểm chứng, không phải kết luận có sẵn để xác nhận lại. Xem docs/plans/llm-reasoning-node.md
# mục "Chống echo chamber" cho lý do đầy đủ.
_SYSTEM_PROMPT = """You give a second opinion on an alert a vision-language model already assessed.

You are not given images — you reason only from the structured evidence in the user message.
Weigh the two sources correctly:

- `rule_evidence` is MEASURED evidence from a motion/geometry rule (velocity, aspect ratio,
  height drop, zone containment, stationary duration). Treat it as the primary, higher-weight
  evidence — it is a number, not an interpretation.
- `vlm_assessment` is a HYPOTHESIS from a vision model that looked at the actual frames. Weigh
  it seriously, but it is something to check against `rule_evidence` and `camera_history`, not
  a conclusion to rubber-stamp. You have no image to independently verify it against, so your
  job is consistency-checking, not independent visual review.
- If `rule_evidence` and `vlm_assessment` point in different directions, say so explicitly in
  `rationale` rather than silently picking one or splitting the difference.

The three incident types:
- fall_detection: a person has fallen and may need medical help. Crouching, sitting down
  deliberately, bending to pick something up, and lying down on purpose are NOT falls.
- intrusion: a person entered a restricted zone. Someone merely passing nearby is not.
- abandoned_object: a bag or item left unattended with no owner nearby. An item beside a
  seated or standing person is not abandoned.

The alert has already reached the on-duty guard regardless of your answer — your verdict only
annotates it, it can never suppress or hide it. Answer only with the JSON object described by
the schema. Keep `rationale` to one sentence. Use "uncertain" honestly when the evidence does
not settle it — an unsure verdict is more useful than a confident wrong one."""

_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "llm_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["real", "false_positive", "uncertain"]},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["verdict", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
}


class LLMVerdict(BaseModel):
    """One LLM judgement on one alert, given the VLM's own verdict as input.

    Same shape as `TriageVerdict` plus nothing extra — `CascadingTriage` (vlm.py) is what
    merges this with the VLM's own latency/token counts into the final `TriageVerdict`.
    """

    verdict: Verdict = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    model: str = ""
    latency_ms: float = 0.0
    total_tokens: int = 0
    error: str | None = None


class LLMReasoner(Protocol):
    """What `CascadingTriage` depends on. Keeps the real client out of the tests."""

    def assess(self, vlm_verdict: TriageVerdict, request: TriageRequest) -> LLMVerdict: ...


class NullLLMReasoner:
    """No-op reasoner: used when the feature is off, no key is configured, or in tests."""

    def __init__(self, reason: str = "LLM reasoning disabled") -> None:
        self._reason = reason

    def assess(self, vlm_verdict: TriageVerdict, request: TriageRequest) -> LLMVerdict:
        return LLMVerdict(verdict="uncertain", error=self._reason)


class CompatibleLLMReasoner:
    """Text-only second opinion over an OpenAI-compatible chat completions API.

    Uses the OpenAI SDK directly, same as `CompatibleVLMTriage` — this is one structured
    text-only call, not a multi-step agent, so LangChain/LangGraph add nothing here.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3.7-flash",
        timeout_sec: float = 15.0,
        max_retries: int = 1,
    ) -> None:
        from openai import OpenAI  # imported lazily so the package is optional at runtime

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec, max_retries=max_retries)

    def assess(self, vlm_verdict: TriageVerdict, request: TriageRequest) -> LLMVerdict:
        """Judge one alert given the VLM's own verdict. Never raises.

        Same failure philosophy as `CompatibleVLMTriage.assess`: every failure becomes
        `uncertain` plus an `error` string, never an exception — `CascadingTriage` falls back
        to the VLM's own verdict when this happens, so a bad minute from this API never costs
        an alert its assessment entirely.
        """
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_message(vlm_verdict, request)},
                ],
                response_format=_RESPONSE_SCHEMA,
                temperature=0.0,
                max_tokens=200,
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            return LLMVerdict(
                verdict=payload.get("verdict", "uncertain"),
                confidence=min(1.0, max(0.0, float(payload.get("confidence", 0.0)))),
                rationale=str(payload.get("rationale", "")).strip(),
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            )
        except Exception as exc:  # noqa: BLE001 - see docstring; no failure may reach the alert
            logger.warning(f"LLM reasoning failed: {type(exc).__name__}: {exc}")
            return LLMVerdict(
                verdict="uncertain",
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _build_user_message(vlm_verdict: TriageVerdict, request: TriageRequest) -> str:
        """A single structured JSON block — see docs/plans/llm-reasoning-node.md mục 2.

        Not free text like `CompatibleVLMTriage._build_prompt`: the model should read named
        fields, not re-parse a hand-written sentence.
        """
        payload = {
            "label": request.label,
            "zone_name": request.zone_name,
            "camera_name": request.camera_name,
            "rule_evidence": request.reason,
            "frame_summary": {
                "before_count": request.pre_count,
                "after_count": max(0, len(request.frames) - request.pre_count),
            },
            "vlm_assessment": {
                "verdict": vlm_verdict.verdict,
                "confidence": vlm_verdict.confidence,
                "rationale": vlm_verdict.rationale,
            },
            "camera_history": request.examples,
        }
        return json.dumps(payload, ensure_ascii=False)
