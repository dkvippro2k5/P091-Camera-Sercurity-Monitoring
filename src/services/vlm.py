"""Second-stage triage: a vision model judging whether a rule-engine event is real.

PRD §1.2 describes two stages — computer vision detects events, then an LLM reasons about
context to filter false alarms. `rule_engine.py` is the first stage and was deliberately
built as a cheap bbox filter whose job is to *trigger* this one, not to adjudicate: its
`reason` string is already formatted as measured evidence
(`2/3 indicators [+V_y=1.62H/s(>=1.2) -AR=0.83(>=1.2) +dH=65%(>=50%)]`) precisely so a model
could read it. This module is what finally reads it.

**The verdict is advisory. It never suppresses an alert.** Every rule-engine event still
reaches the operator; this layer only annotates. Two reasons: a small model silently
swallowing a real fall is unrecoverable for the person on the floor, and keeping every alert
visible is what makes a rules-alone vs rules-plus-VLM precision/recall comparison possible
at all (PRD §4, "Tỷ lệ lọc nhờ LLM"). Suppression becomes a config flip once the measured
numbers justify it — not before.

Three implementations behind one protocol:

    VLMTriage        - the protocol MonitoringService depends on
    OpenAIVLMTriage  - the real client
    NullVLMTriage    - returns `unavailable` without a network call

The null client is what runs when the feature is off or no API key is set, and it is what the
test suite uses — the suite stays offline and fast, and a fresh clone behaves exactly as it
did before this layer existed.
"""

import base64
import json
import logging
import time
from typing import Literal, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Verdict = Literal["real", "false_positive", "uncertain"]

# Ordered oldest-first and labelled in the prompt, because "did they fall or stand up" is
# answerable only from the sequence. A single frame cannot separate a fall from lying down.
_SYSTEM_PROMPT = """You review automated alerts from a residential security camera system.

A cheap motion/geometry rule has already fired on a video clip and produced measured evidence.
Your job is to judge whether the alert describes a REAL incident of the stated type, or a false
positive the rule could not rule out on its own.

The three incident types:
- fall_detection: a person has fallen and may need medical help. Crouching, sitting down
  deliberately, bending to pick something up, and lying down on purpose are NOT falls.
- intrusion: a person entered a restricted zone. Someone merely passing nearby is not.
- abandoned_object: a bag or item left unattended with no owner nearby. An item beside a
  seated or standing person is not abandoned.

You are given frames in chronological order, labelled BEFORE (leading up to the moment the rule
fired) and AFTER. Judge the transition between them, not any single frame.

Answer only with the JSON object described by the schema. Keep `rationale` to one sentence an
on-duty guard can read at a glance. Use "uncertain" honestly when the frames do not settle it —
an unsure verdict is more useful than a confident wrong one."""

_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triage_verdict",
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


class TriageVerdict(BaseModel):
    """One model judgement on one alert.

    `error` is set on every failure path, and the verdict is `uncertain` whenever it is —
    see `OpenAIVLMTriage.assess` for why this never raises.
    """

    verdict: Verdict = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    model: str = ""
    latency_ms: float = 0.0
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


class VLMTriage(Protocol):
    """What MonitoringService depends on. Keeps the real client out of the tests."""

    def assess(self, request: TriageRequest) -> TriageVerdict: ...


class NullVLMTriage:
    """No-op triage: used when the feature is off, no key is configured, or in tests."""

    def __init__(self, reason: str = "VLM triage disabled") -> None:
        self._reason = reason

    def assess(self, request: TriageRequest) -> TriageVerdict:
        return TriageVerdict(verdict="uncertain", error=self._reason)


class OpenAIVLMTriage:
    """Vision-model triage over the OpenAI chat completions API.

    Uses the OpenAI SDK directly rather than LangChain: this is one structured call with
    images, and the template's LangGraph scaffold belongs to a different product surface.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_sec: float = 20.0,
        max_retries: int = 1,
    ) -> None:
        from openai import OpenAI  # imported lazily so the package is optional at runtime

        self.model = model
        # The SDK's own retry handles the transient 429/5xx case; a hand-rolled loop around
        # a 20s timeout would multiply worst-case latency instead of bounding it.
        self._client = OpenAI(api_key=api_key, timeout=timeout_sec, max_retries=max_retries)

    def assess(self, request: TriageRequest) -> TriageVerdict:
        """Judge one alert. Never raises.

        A triage layer that propagated an exception would take out the alert it was meant to
        annotate — the operator would lose a real incident because a third-party API had a bad
        minute. Every failure becomes `uncertain` plus an `error` string instead, which the UI
        renders as "assessment unavailable" while leaving the alert itself untouched.
        """
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_content(request)},
                ],
                response_format=_RESPONSE_SCHEMA,
                # Deterministic-ish: this is a classification, not a writing task.
                temperature=0.0,
                max_tokens=200,
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            return TriageVerdict(
                verdict=payload.get("verdict", "uncertain"),
                # Clamped rather than trusted: a model returning 0-100 instead of 0-1 would
                # otherwise fail Pydantic validation and lose an otherwise-good verdict.
                confidence=min(1.0, max(0.0, float(payload.get("confidence", 0.0)))),
                rationale=str(payload.get("rationale", "")).strip(),
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        except Exception as exc:  # noqa: BLE001 - see docstring; no failure may reach the alert
            logger.warning(f"VLM triage failed: {type(exc).__name__}: {exc}")
            return TriageVerdict(
                verdict="uncertain",
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_content(self, request: TriageRequest) -> list[dict]:
        """Text block first, then the frames labelled BEFORE / AFTER in order."""
        content: list[dict] = [{"type": "text", "text": self._build_prompt(request)}]
        for index, jpeg in enumerate(request.frames):
            phase = "BEFORE" if index < request.pre_count else "AFTER"
            content.append({"type": "text", "text": f"[{phase} frame {index + 1}]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("utf-8"),
                        # Surveillance stills at tile resolution; "high" doubles the token
                        # cost per image for detail this judgement does not turn on.
                        "detail": "low",
                    },
                }
            )
        return content

    @staticmethod
    def _build_prompt(request: TriageRequest) -> str:
        lines = [
            f"Alert type: {request.label}",
            f"Camera: {request.camera_name or 'unknown'}",
        ]
        if request.zone_name:
            lines.append(f"Zone: {request.zone_name}")
        if request.reason:
            # Verbatim. The indicator values and their thresholds are the evidence.
            lines.append(f"Rule engine evidence: {request.reason}")
        lines.append(
            f"Frames: {request.pre_count} before the trigger, "
            f"{max(0, len(request.frames) - request.pre_count)} after, in chronological order."
        )

        if request.examples:
            lines.append("")
            lines.append("Past incidents on this camera, as judged by the on-duty guard:")
            lines.extend(f"- {example}" for example in request.examples)
            lines.append("Use these to calibrate what this particular camera's false alarms look like.")

        return "\n".join(lines)


def build_triage_client(settings) -> VLMTriage:  # noqa: ANN001 - Settings, avoided to dodge a config import cycle
    """Pick a triage client from settings, degrading to the null client rather than failing.

    Both switches matter: `vlm_enabled` is the deliberate off switch, and a missing API key
    is the accidental one. Treating a missing key as an error would make the whole monitoring
    service refuse to start over an optional annotation layer.
    """
    if not getattr(settings, "vlm_enabled", False):
        return NullVLMTriage("VLM triage disabled (VLM_ENABLED=false)")
    if not settings.openai_api_key:
        logger.warning("VLM_ENABLED is true but OPENAI_API_KEY is empty; alerts will carry no assessment.")
        return NullVLMTriage("No OPENAI_API_KEY configured")

    try:
        return OpenAIVLMTriage(
            api_key=settings.openai_api_key,
            model=settings.vlm_model,
            timeout_sec=settings.vlm_timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001 - a missing openai package must not block startup
        logger.error(f"Could not build the VLM client, continuing without triage: {exc}")
        return NullVLMTriage(f"VLM client unavailable: {exc}")
