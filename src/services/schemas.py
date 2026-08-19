"""Shared triage data models, owned by neither the VLM nor the LLM stage.

`vlm.py` and `llm_reasoner.py` both need these: the VLM stage produces a `TriageVerdict`
from a `TriageRequest`, and the LLM stage reads both back as its own input. Defining them
here rather than in `vlm.py` is what keeps the two modules from importing each other.
"""

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["real", "false_positive", "uncertain"]


class TriageVerdict(BaseModel):
    """One model judgement on one alert.

    `error` is set on every failure path, and the verdict is `uncertain` whenever it is —
    see `CompatibleVLMTriage.assess` for why this never raises.
    """

    verdict: Verdict = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    model: str = ""
    latency_ms: float = 0.0
    # Prompt + completion tokens for this one call, read from the API response's own usage
    # block rather than estimated — needed to compute cost_usd in the benchmark harness.
    # 0 on any path that didn't make a billable call (Null client, or a failed request).
    total_tokens: int = 0
    error: str | None = None


class TriageRequest(BaseModel):
    """Everything the model is shown about one alert.

    `reason` is the rule engine's own evidence string, passed verbatim — it carries the
    measured indicator values, which is the whole point of routing it here.
    """

    label: str
    reason: str | None = None
    zone_name: str | None = None
    camera_name: str = ""
    # JPEG bytes, chronological. `pre_count` marks the split: the first `pre_count` frames
    # precede the moment the rule fired, the rest follow it.
    frames: list[bytes] = Field(default_factory=list)
    pre_count: int = 0
    # Rendered lines from past operator-labelled incidents on this camera (alert_history).
    examples: list[str] = Field(default_factory=list)
