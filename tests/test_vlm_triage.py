"""Tests for the second-stage VLM triage layer and the HITL feedback loop.

Entirely offline and weight-free, like `test_monitoring_api.py`: the detector is the
supported null detector and the triage client is a fake, so nothing here loads YOLO or
touches the network. That is a hard requirement, not a convenience — a suite that made a
paid API call would be flaky, billable, and unrunnable in CI.

What these pin, in order of how expensive the bug would be:

  - the alert reaches the operator *before* triage runs (latency contract)
  - a false_positive verdict does not suppress the alert (the "annotate only" decision)
  - triage never runs on the caller's thread (the callback-must-not-block constraint)
  - the pre-roll ring holds distinct frames (the aliasing bug)
"""

import threading
import time

import numpy as np
import pytest

from src.models.camera_schemas import Alert
from src.services import monitoring as monitoring_module
from src.services.alert_history import AlertHistory
from src.services.monitoring import MonitoringService, _evenly_spaced
from src.services.vlm import CompatibleVLMTriage, NullVLMTriage, TriageRequest, TriageVerdict, build_triage_client


class FakeTriage:
    """Records what it was asked and returns a scripted verdict."""

    def __init__(self, verdict: str = "real", rationale: str = "looks real", delay: float = 0.0) -> None:
        self.verdict = verdict
        self.rationale = rationale
        self.delay = delay
        self.requests: list[TriageRequest] = []
        self.thread_names: list[str] = []
        self.called = threading.Event()

    def assess(self, request: TriageRequest) -> TriageVerdict:
        self.thread_names.append(threading.current_thread().name)
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        self.called.set()
        return TriageVerdict(verdict=self.verdict, confidence=0.8, rationale=self.rationale, model="fake")


class ExplodingTriage:
    """A triage client having a bad day. Nothing it does may reach the alert."""

    def assess(self, request: TriageRequest) -> TriageVerdict:
        raise RuntimeError("upstream is on fire")


@pytest.fixture
def make_service(tmp_path, isolated_data):
    """Builds a MonitoringService with a null detector, isolated files, and a given triage.

    `isolated_data` (conftest) redirects the detector and both JSONL logs; the history is
    then passed explicitly as well because several tests read it back by path.
    """

    def _build(triage=None, **overrides):
        service = MonitoringService(
            triage=triage or FakeTriage(),
            history=AlertHistory(path=str(tmp_path / "history.jsonl")),
        )
        # Post-roll defaults to 2s; tests should not sit through it.
        service._post_roll_sec = overrides.get("post_roll_sec", 0.0)
        return service

    services: list[MonitoringService] = []

    def _tracked(**kwargs):
        service = _build(**kwargs)
        services.append(service)
        return service

    yield _tracked
    for service in services:
        service.shutdown()


def _frame(value: int = 0) -> np.ndarray:
    return np.full((60, 80, 3), value, dtype=np.uint8)


def _emit(service, camera_id="cam_1", label="fall_detection", reason="2/3 indicators") -> Alert:
    service._emit_alert(camera_id=camera_id, label=label, confidence=0.9, frame=_frame(), reason=reason)
    return service._alerts[0]


# ------------------------------------------------------------------ latency contract
def test_alert_is_published_before_triage_completes(make_service):
    """The whole reason triage is asynchronous.

    A fall must reach the guard in one frame time. If the alert waited on the assessment,
    a slow API would add seconds to a medical emergency.
    """
    triage = FakeTriage(delay=0.5)
    service = make_service(triage=triage)

    published: list[str] = []
    service._publish = lambda event_type, data: published.append(event_type)

    alert = _emit(service)

    # _emit_alert has returned while the fake is still sleeping inside assess().
    assert published == ["alert.created"]
    assert alert.triage_status == "pending"

    assert triage.called.wait(timeout=5), "triage never ran"


def test_triage_runs_off_the_caller_thread(make_service):
    """`_on_incident` runs on the camera's processor thread; blocking it stalls that camera."""
    triage = FakeTriage()
    service = make_service(triage=triage)
    caller = threading.current_thread().name

    _emit(service)
    assert triage.called.wait(timeout=5)

    assert triage.thread_names[0] != caller
    assert triage.thread_names[0].startswith("vlm-triage")


def test_verdict_event_is_published_when_triage_finishes(make_service):
    triage = FakeTriage(verdict="real", rationale="person is prone and motionless")
    service = make_service(triage=triage)

    published: list[tuple[str, dict]] = []
    service._publish = lambda event_type, data: published.append((event_type, data))

    alert = _emit(service)
    assert triage.called.wait(timeout=5)
    _wait_until(lambda: any(event == "alert.verdict" for event, _ in published))

    assert alert.triage_status == "complete"
    assert alert.triage_verdict == "real"
    assert alert.triage_rationale == "person is prone and motionless"
    assert [event for event, _ in published] == ["alert.created", "alert.verdict"]


# ------------------------------------------------------------------ annotate only
def test_false_positive_verdict_does_not_suppress_the_alert(make_service):
    """The design decision this layer rests on.

    The model annotates; it never hides. A suppressed alert is one the operator cannot
    overrule and one that never enters the labelled history — so a wrong suppression would
    be invisible *and* self-reinforcing.
    """
    service = make_service(triage=FakeTriage(verdict="false_positive"))

    alert = _emit(service)
    _wait_until(lambda: alert.triage_status == "complete")

    assert alert.triage_verdict == "false_positive"
    assert alert in service._alerts
    assert service.list_alerts(unacknowledged_only=True) == [alert]


def test_triage_failure_leaves_the_alert_intact(make_service):
    """A third-party API having a bad minute must not cost the operator an incident."""
    service = make_service(triage=ExplodingTriage())

    alert = _emit(service)
    _wait_until(lambda: alert.triage_status == "complete")

    assert alert.triage_verdict == "uncertain"
    assert "upstream is on fire" in (alert.triage_error or "")
    assert alert in service._alerts
    assert alert.image_data_url.startswith("data:image/jpeg;base64,")


def test_saturated_queue_skips_triage_instead_of_queueing(make_service):
    """Bounded on purpose: a flickering camera must not build an unbounded backlog of
    paid calls whose verdicts arrive long after the operator has moved on."""
    service = make_service(triage=FakeTriage(delay=2.0))
    service._triage_in_flight = monitoring_module._MAX_TRIAGE_IN_FLIGHT

    alert = _emit(service)

    assert alert.triage_status == "skipped"
    assert alert.triage_error == "triage queue saturated"


def test_null_triage_leaves_status_skipped(make_service):
    """With the layer off, an alert is exactly what it was before this feature existed."""
    service = make_service(triage=NullVLMTriage())

    alert = _emit(service)

    assert alert.triage_status == "skipped"
    assert alert.triage_verdict is None


# ------------------------------------------------------------------ frame ring
def test_ring_holds_distinct_frames(make_service):
    """Guards the aliasing bug.

    `process_frame` may return the reader thread's shared buffer, so a ring of ndarrays
    would silently become N references to the newest frame — a "clip" of identical images
    that looks fine in every log line. Encoding to JPEG at push time is what prevents it.
    """
    service = make_service()
    for value in (10, 120, 240):
        service._on_frame("cam_1", None, _frame(value))

    stored = [jpeg for _, jpeg in service._ring["cam_1"]]
    assert len(stored) == 3
    assert len(set(stored)) == 3, "ring collapsed to duplicate frames"


def test_clip_splits_into_pre_and_post_roll(make_service):
    """The split point is when the rule fired, not when the collector woke up."""
    service = make_service()
    service._post_roll_sec = 0.2
    service._pre_roll_sec = 5.0

    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", __import__("collections").deque(maxlen=64))
    # Four frames before the event, two after (event is at now - 0.2).
    for offset, payload in [(-3.0, b"a"), (-2.0, b"b"), (-1.0, b"c"), (-0.5, b"d")]:
        ring.append((now + offset, payload))
    for offset, payload in [(-0.1, b"e"), (0.0, b"f")]:
        ring.append((now + offset, payload))

    frames, pre_count = service._collect_clip("cam_1")

    assert pre_count > 0, "no pre-roll frames selected"
    assert len(frames) > pre_count, "no post-roll frames selected"
    assert frames[:pre_count] == [f for f in frames[:pre_count] if f in {b"a", b"b", b"c", b"d"}]
    assert set(frames[pre_count:]) <= {b"e", b"f"}


def test_pre_roll_reaches_past_the_fall_dwell(make_service):
    """A 3s dwell means the event fires 3s after the fall.

    With a shorter pre-roll the "before" frames show someone already on the ground, the
    model sees no fall, and it correctly calls a real fall a false positive. The default
    has to clear dwell (3.0) plus the velocity window (0.5).
    """
    service = make_service()
    assert service._pre_roll_sec >= 3.5


def test_empty_ring_yields_no_frames(make_service):
    service = make_service()
    assert service._collect_clip("cam_unknown") == ([], 0)


def test_evenly_spaced_keeps_the_ends():
    """Taking the first N would describe the first half-second and nothing else."""
    items = [bytes([i]) for i in range(10)]

    assert _evenly_spaced(items, 4) == [b"\x00", b"\x03", b"\x06", b"\x09"]
    assert _evenly_spaced(items, 20) == items
    assert _evenly_spaced([], 4) == []
    assert _evenly_spaced(items, 0) == []


# ------------------------------------------------------------------ HITL + history
def test_verdict_records_exactly_one_history_row(make_service, tmp_path):
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service)

    assert service.record_operator_verdict(alert.id, "real") is True

    rows = service.history.all_rows()
    assert len(rows) == 1
    assert rows[0]["operator_verdict"] == "real"
    assert rows[0]["label"] == "fall_detection"
    assert rows[0]["reason"] == "2/3 indicators"
    assert alert.operator_verdict == "real"
    assert alert.acknowledged is True


def test_plain_acknowledge_writes_no_history(make_service):
    """Dismissing is not judging. An unlabelled row would sit in the dataset the
    false-alarm rate is computed from, counted as neither a hit nor a miss."""
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service)

    assert service.acknowledge(alert.id) is True

    assert service.history.all_rows() == []
    assert alert.operator_verdict is None
    assert alert.acknowledged is True


def test_verdict_on_a_missing_alert_is_false(make_service):
    service = make_service(triage=NullVLMTriage())
    assert service.record_operator_verdict("nope", "real") is False


def test_history_filters_by_camera_and_label(tmp_path):
    """A fall on the stairwell says nothing about a bag in the lobby — mixing them would
    spend prompt tokens actively misleading the model."""
    history = AlertHistory(path=str(tmp_path / "h.jsonl"))
    history.record("a1", "cam_1", "fall_detection", "real", reason="fall evidence")
    history.record("a2", "cam_1", "abandoned_object", "false", reason="bag evidence")
    history.record("a3", "cam_2", "fall_detection", "false", reason="other camera")

    examples = history.recent_examples("cam_1", "fall_detection")

    assert len(examples) == 1
    assert "fall evidence" in examples[0]
    assert "REAL" in examples[0]


def test_history_respects_the_example_limit(tmp_path):
    history = AlertHistory(path=str(tmp_path / "h.jsonl"))
    for index in range(10):
        history.record(f"a{index}", "cam_1", "intrusion", "real", reason=f"evidence {index}")

    examples = history.recent_examples("cam_1", "intrusion", limit=3)

    assert len(examples) == 3
    # Most recent last: a model scanning the list reads recency as relevance.
    assert "evidence 9" in examples[-1]
    assert history.recent_examples("cam_1", "intrusion", limit=0) == []


def test_history_survives_a_corrupt_line(tmp_path):
    """A half-written final line is normal if the process died mid-append."""
    path = tmp_path / "h.jsonl"
    history = AlertHistory(path=str(path))
    history.record("a1", "cam_1", "intrusion", "real", reason="good row")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"alert_id": "truncated"\n')

    assert len(history.recent_examples("cam_1", "intrusion")) == 1


def test_missing_history_file_is_not_an_error(tmp_path):
    history = AlertHistory(path=str(tmp_path / "absent.jsonl"))
    assert history.recent_examples("cam_1", "intrusion") == []
    assert history.all_rows() == []


def test_history_reaches_the_triage_prompt(make_service):
    """The loop closing: a judgement made now shapes the next assessment."""
    triage = FakeTriage()
    service = make_service(triage=triage)
    service.history.record("old", "cam_1", "fall_detection", "false", reason="prior evidence")

    _emit(service)
    assert triage.called.wait(timeout=5)

    assert any("prior evidence" in example for example in triage.requests[0].examples)


# ------------------------------------------------------------------ client selection
def test_build_triage_client_is_null_when_disabled():
    settings = type("S", (), {"vlm_enabled": False, "vlm_api_key": "sk-real"})()
    assert isinstance(build_triage_client(settings), NullVLMTriage)


def test_build_triage_client_is_null_without_a_key():
    """Enabled but unkeyed must degrade, not refuse to start — this layer is optional."""
    settings = type("S", (), {"vlm_enabled": True, "vlm_api_key": ""})()
    assert isinstance(build_triage_client(settings), NullVLMTriage)


def test_build_triage_client_builds_the_real_one_when_configured():
    settings = type(
        "S",
        (),
        {
            "vlm_enabled": True,
            "vlm_api_key": "sk-test-not-used",
            "vlm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vlm_model": "qwen3.7-flash",
            "vlm_timeout_sec": 5.0,
        },
    )()
    # Construction only — no request is made, so no network call happens here.
    assert isinstance(build_triage_client(settings), CompatibleVLMTriage)


def test_prompt_carries_the_rule_evidence_verbatim():
    """The measured indicator values are the evidence; paraphrasing them loses the point."""
    reason = "2/3 indicators [+V_y=1.62H/s(>=1.2) -AR=0.83(>=1.2) +dH=65%(>=50%)]"
    prompt = CompatibleVLMTriage._build_prompt(
        TriageRequest(label="fall_detection", reason=reason, camera_name="Lobby", pre_count=4, frames=[b"x"] * 6)
    )

    assert reason in prompt
    assert "fall_detection" in prompt
    assert "Lobby" in prompt


def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Poll a condition set by a worker thread instead of sleeping a fixed amount."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")
