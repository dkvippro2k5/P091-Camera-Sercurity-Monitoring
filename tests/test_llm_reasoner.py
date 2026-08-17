"""Tests for the optional Vision -> VLM -> LLM extension (CascadingTriage + LLMReasoner).

Mirrors `tests/test_vlm_triage.py`'s structure and offline-only philosophy: no network call,
fake doubles standing in for the real clients. What these pin:

  - CascadingTriage never runs the LLM stage when the VLM itself already failed
  - a failing LLM stage falls back to the VLM's own verdict, never loses it
  - the happy path's final verdict/rationale/model come from the LLM, not the VLM
  - the JSON sent to the LLM carries rule evidence, VLM verdict, and camera history
  - build_triage_client only builds CascadingTriage when both flags are on
"""

from src.services.llm_reasoner import CompatibleLLMReasoner, LLMVerdict
from src.services.vlm import CascadingTriage, CompatibleVLMTriage, TriageRequest, TriageVerdict, build_triage_client


class FakeVLM:
    """Records what it was asked and returns a scripted verdict."""

    def __init__(self, verdict: str = "real", rationale: str = "vlm says real", error: str | None = None) -> None:
        self.verdict = verdict
        self.rationale = rationale
        self.error = error
        self.requests: list[TriageRequest] = []

    def assess(self, request: TriageRequest) -> TriageVerdict:
        self.requests.append(request)
        if self.error:
            return TriageVerdict(verdict="uncertain", model="fake-vlm", error=self.error)
        return TriageVerdict(
            verdict=self.verdict, confidence=0.8, rationale=self.rationale, model="fake-vlm", total_tokens=100
        )


class FakeLLMReasoner:
    """Records what it was asked and returns a scripted verdict."""

    def __init__(self, verdict: str = "false_positive", rationale: str = "llm disagrees") -> None:
        self.verdict = verdict
        self.rationale = rationale
        self.calls: list[tuple[TriageVerdict, TriageRequest]] = []

    def assess(self, vlm_verdict: TriageVerdict, request: TriageRequest) -> LLMVerdict:
        self.calls.append((vlm_verdict, request))
        return LLMVerdict(verdict=self.verdict, confidence=0.6, rationale=self.rationale, model="fake-llm", total_tokens=50)


class ExplodingLLMReasoner:
    """An LLM reasoner having a bad day. Nothing it does may lose the VLM's own verdict."""

    def assess(self, vlm_verdict: TriageVerdict, request: TriageRequest) -> LLMVerdict:
        return LLMVerdict(verdict="uncertain", model="fake-llm", error="RuntimeError: upstream is on fire")


def _request(**overrides) -> TriageRequest:
    defaults = dict(label="fall_detection", reason="3/3 indicators [...]", camera_name="Lobby", frames=[b"x"])
    defaults.update(overrides)
    return TriageRequest(**defaults)


# ------------------------------------------------------------------ CascadingTriage.assess
def test_cascading_happy_path_returns_the_llm_verdict():
    """When both stages succeed, the final verdict/rationale/model come from the LLM."""
    vlm = FakeVLM(verdict="real", rationale="vlm says real")
    llm = FakeLLMReasoner(verdict="false_positive", rationale="llm disagrees")
    triage = CascadingTriage(vlm=vlm, reasoner=llm)

    result = triage.assess(_request())

    assert result.verdict == "false_positive"
    assert result.rationale == "llm disagrees"
    assert result.model == "fake-vlm+fake-llm"
    assert result.error is None
    # Cost/latency accumulate across both calls -- the benchmark needs the true total.
    assert result.total_tokens == 150


def test_cascading_skips_the_llm_when_vlm_already_failed():
    """No verdict to review -> running the LLM would mean judging from rule_evidence alone,
    which is exactly what the VLM stage exists to avoid."""
    vlm = FakeVLM(error="APITimeoutError: Request timed out.")
    llm = FakeLLMReasoner()
    triage = CascadingTriage(vlm=vlm, reasoner=llm)

    result = triage.assess(_request())

    assert result.error == "APITimeoutError: Request timed out."
    assert llm.calls == []  # never called


def test_cascading_falls_back_to_vlm_verdict_when_llm_fails():
    """A bad LLM call must not downgrade an alert that already had a usable VLM verdict."""
    vlm = FakeVLM(verdict="real", rationale="vlm says real")
    triage = CascadingTriage(vlm=vlm, reasoner=ExplodingLLMReasoner())

    result = triage.assess(_request())

    assert result.verdict == "real"
    assert result.rationale == "vlm says real"
    assert result.model == "fake-vlm"
    assert result.error is None  # the alert still gets a usable annotation


# ------------------------------------------------------------------ prompt construction
def test_llm_user_message_carries_rule_evidence_vlm_verdict_and_history():
    """The JSON sent to the LLM must carry everything CascadingTriage promised it would."""
    reason = "3/3 indicators [+V_y=1.8H/s(>=1.2) +AR=1.6(>=1.2) +dH=72%(>=50%)]"
    request = _request(
        label="fall_detection",
        reason=reason,
        zone_name="Lobby Zone",
        camera_name="cam_platform_demo",
        pre_count=1,
        frames=[b"x"] * 4,
        examples=["fall_detection: real - person stayed down 6s"],
    )
    vlm_verdict = TriageVerdict(verdict="real", confidence=0.92, rationale="person collapses and stays down")

    message = CompatibleLLMReasoner._build_user_message(vlm_verdict, request)

    assert reason in message
    assert "person collapses and stays down" in message
    assert "0.92" in message
    assert "fall_detection: real - person stayed down 6s" in message
    assert "Lobby Zone" in message
    assert "cam_platform_demo" in message


def test_llm_user_message_is_valid_json():
    """Not free text like vlm.py's own prompt builder -- must parse as a single JSON object."""
    import json

    request = _request()
    vlm_verdict = TriageVerdict(verdict="uncertain", confidence=0.5, rationale="not sure")

    message = CompatibleLLMReasoner._build_user_message(vlm_verdict, request)
    parsed = json.loads(message)

    assert parsed["vlm_assessment"]["verdict"] == "uncertain"
    assert parsed["frame_summary"] == {"before_count": 0, "after_count": 1}


# ------------------------------------------------------------------ client selection
def test_build_triage_client_is_vlm_only_when_llm_reasoning_disabled():
    settings = type(
        "S",
        (),
        {
            "vlm_enabled": True,
            "vlm_api_key": "sk-test-not-used",
            "vlm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vlm_model": "qwen3.7-flash",
            "vlm_timeout_sec": 5.0,
            "llm_reasoning_enabled": False,
        },
    )()
    assert isinstance(build_triage_client(settings), CompatibleVLMTriage)


def test_build_triage_client_degrades_to_vlm_only_without_an_llm_key():
    """Enabled but unkeyed must degrade, not lose the working VLM layer with it."""
    settings = type(
        "S",
        (),
        {
            "vlm_enabled": True,
            "vlm_api_key": "sk-test-not-used",
            "vlm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vlm_model": "qwen3.7-flash",
            "vlm_timeout_sec": 5.0,
            "llm_reasoning_enabled": True,
            "llm_api_key": "",
        },
    )()
    assert isinstance(build_triage_client(settings), CompatibleVLMTriage)


def test_build_triage_client_builds_cascading_when_both_flags_configured():
    settings = type(
        "S",
        (),
        {
            "vlm_enabled": True,
            "vlm_api_key": "sk-test-not-used",
            "vlm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vlm_model": "qwen3.7-flash",
            "vlm_timeout_sec": 5.0,
            "llm_reasoning_enabled": True,
            "llm_api_key": "sk-test-not-used",
            "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "qwen3.7-flash",
            "llm_reasoning_timeout_sec": 5.0,
        },
    )()
    # Construction only -- no request is made, so no network call happens here.
    assert isinstance(build_triage_client(settings), CascadingTriage)
