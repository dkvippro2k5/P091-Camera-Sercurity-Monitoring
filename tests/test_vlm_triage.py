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

import json
import threading
import time
from collections import deque

import numpy as np
import pytest

from src.models.camera_schemas import Alert
from src.services import monitoring as monitoring_module
from src.services.alert_history import AlertHistory
from src.services.evidence_store import EvidenceStore
from src.services.monitoring import MonitoringService, _evenly_spaced, _evidence_frames
from src.services.vlm import (
    LocalSmolVLMTriage,
    NullVLMTriage,
    OpenAIVLMTriage,
    TriageRequest,
    TriageVerdict,
    build_triage_client,
)


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
    now = time.monotonic()
    ring = service._ring.setdefault(camera_id, deque(maxlen=service._default_ring_size))
    ring.append((now - 1.0, b"fake_jpeg_before"))
    ring.append((now + 0.1, b"fake_jpeg_after"))
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

    published: list[tuple[str, dict]] = []
    service._publish = lambda event_type, data: published.append((event_type, data))

    alert = _emit(service)

    # _emit_alert has returned while the fake is still sleeping inside assess().
    # alert.evidence comes from the unrelated evidence-store upload, submitted to its own
    # thread pool in the same call; it races independently of triage and is filtered here
    # because this test pins the triage latency contract, not evidence persistence.
    triage_events = [event for event, _ in published if event != "alert.evidence"]
    assert triage_events == ["alert.created"]
    assert published[0][1]["triage_status"] == "pending"
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
    # See test_alert_is_published_before_triage_completes for why alert.evidence is filtered.
    triage_events = [event for event, _ in published if event != "alert.evidence"]
    assert triage_events == ["alert.created", "alert.verdict"]


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
    assert alert.evidence_url == f"/api/v1/evidence/{alert.id}"


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


def test_missing_after_evidence_is_uncertain(make_service):
    """No aftermath means no outcome to judge, so the call is not worth making.

    The camera died at the moment of the event. Whatever the lead-up showed, nothing can say
    whether the person got up — which is the question the assessment exists to answer.
    """
    triage = FakeTriage(verdict="real")
    service = make_service(triage=triage)
    service.triage = NullVLMTriage()
    alert = _emit(service)
    service.triage = triage
    # Everything sits before the split point, so `post` is empty.
    stale = time.monotonic() - service._post_roll_sec - 1.0
    service._ring["cam_1"] = deque([(stale, b"before")], maxlen=8)
    service._triage_in_flight = 1

    service._run_triage(alert)

    assert alert.triage_verdict == "uncertain"
    assert "no frames after the event" in (alert.triage_error or "")
    assert triage.requests == []


def test_missing_before_evidence_is_still_assessed(make_service):
    """The regression this pair exists to prevent: a startup alert must still be judged.

    Both sides used to be required, which quietly made every alert in a camera's first
    seconds unassessable — the ring fills from empty, so `pre` is 0 at t≈1s on a 5fps camera
    with a 5s pre-roll. The operator saw "không thể đánh giá" on exactly the alerts a freshly
    started clip produces, with nothing they could do about it.

    Aftermath alone still answers most of the question, and the prompt tells the model the
    split is 0-before so it can weigh a thin clip itself rather than being second-guessed here.
    """
    triage = FakeTriage(verdict="real")
    service = make_service(triage=triage)
    service.triage = NullVLMTriage()
    alert = _emit(service)
    service.triage = triage
    # Stamped ahead of the split so the frame counts as "after". `_collect_clip` computes the
    # split as `now - post_roll` at call time, and post_roll is 0 in these tests — so a frame
    # stamped at the current instant would already be in the past by the time the split is
    # taken, and would land on the "before" side instead of the side under test.
    service._ring["cam_1"] = deque([(time.monotonic() + 1.0, b"after")], maxlen=8)
    service._triage_in_flight = 1

    service._run_triage(alert)

    assert alert.triage_verdict == "real"
    assert alert.triage_error is None
    # The model was actually consulted, and told there is no lead-up to reason from.
    assert len(triage.requests) == 1
    assert triage.requests[0].pre_count == 0


def test_ring_capacity_uses_camera_target_fps(make_service):
    service = make_service()
    camera = type("Camera", (), {"target_fps": 30.0})()
    service._cameras["cam_1"] = camera

    service._on_frame("cam_1", None, _frame(10))

    assert service._ring["cam_1"].maxlen >= int(30 * service._pre_roll_sec * 2)


def test_evenly_spaced_keeps_the_ends():
    """Taking the first N would describe the first half-second and nothing else."""
    items = [bytes([i]) for i in range(10)]

    assert _evenly_spaced(items, 4) == [b"\x00", b"\x03", b"\x06", b"\x09"]
    assert _evenly_spaced(items, 20) == items
    assert _evenly_spaced([], 4) == []
    assert _evenly_spaced(items, 0) == []


def test_evidence_frames_are_labelled_for_the_frontend(tmp_path):
    store = EvidenceStore(backend="local", local_dir=str(tmp_path))
    evidence = _evidence_frames([b"before-a", b"before-b", b"after-a"], 2, "alert-1", store)

    assert [frame.index for frame in evidence] == [1, 2, 3]
    assert [frame.phase for frame in evidence] == ["before", "before", "after"]
    # URLs into the authenticated evidence route, and the bytes really are on disk — the
    # frontend has nothing to render if the strip is described but never stored.
    assert [frame.image_url for frame in evidence] == [
        "/api/v1/evidence/alert-1-f1",
        "/api/v1/evidence/alert-1-f2",
        "/api/v1/evidence/alert-1-f3",
    ]
    assert store.fetch("alert-1-f1") == b"before-a"
    assert store.fetch("alert-1-f3") == b"after-a"


def test_evidence_frames_skip_a_frame_the_store_rejects(tmp_path):
    """A storage failure loses that frame, not the whole evidence strip or the verdict."""

    class HalfBrokenStore(EvidenceStore):
        def put(self, evidence_id: str, jpeg: bytes) -> str:
            if evidence_id.endswith("-f2"):
                raise OSError("disk full")
            return super().put(evidence_id, jpeg)

    store = HalfBrokenStore(backend="local", local_dir=str(tmp_path))
    evidence = _evidence_frames([b"a", b"b", b"c"], 1, "alert-2", store)

    assert [frame.index for frame in evidence] == [1, 3]


# ------------------------------------------------------------------ HITL + history
def test_verdict_records_exactly_one_history_row(make_service, tmp_path):
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service)

    assert service.record_operator_verdict(
        alert.id,
        "real",
        operator_note="Employee slipped near the loading bay.",
        actual_event_type="real_incident",
    ) is True

    rows = service.history.all_rows()
    assert len(rows) == 1
    assert rows[0]["operator_verdict"] == "real"
    assert rows[0]["label"] == "fall_detection"
    assert rows[0]["reason"] == "2/3 indicators"
    assert rows[0]["operator_note"] == "Employee slipped near the loading bay."
    assert rows[0]["actual_event_type"] == "real_incident"
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
    settings = type("S", (), {"vlm_enabled": False, "openai_api_key": "sk-real"})()
    assert isinstance(build_triage_client(settings), NullVLMTriage)


def test_build_triage_client_is_null_without_a_key():
    """Enabled but unkeyed must degrade, not refuse to start — this layer is optional."""
    settings = type("S", (), {"vlm_enabled": True, "openai_api_key": ""})()
    assert isinstance(build_triage_client(settings), NullVLMTriage)


def test_build_triage_client_builds_local_without_api_key():
    settings = type(
        "S",
        (),
        {
            "vlm_enabled": True,
            "vlm_provider": "local",
            "openai_api_key": "",
            "vlm_local_model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "vlm_local_device": "auto",
            "vlm_local_max_image_size": 384,
            "vlm_local_max_new_tokens": 256,
        },
    )()

    client = build_triage_client(settings)

    assert isinstance(client, LocalSmolVLMTriage)
    assert client.max_image_size == 384
    assert client._model is None


def test_local_vlm_posture_state_is_conservative():
    assert LocalSmolVLMTriage._posture_state("The person is standing upright.") == "upright"
    assert LocalSmolVLMTriage._posture_state("The person is lying on the floor.") == "lying"
    assert LocalSmolVLMTriage._posture_state("The person is sitting on a chair.") == "other"
    assert LocalSmolVLMTriage._posture_state("The view is blocked.") == "unknown"


def test_local_vlm_infers_only_an_upright_to_floor_transition():
    request = TriageRequest(label="fall_detection", pre_count=2, frames=[b"1"] * 4)
    verdict = LocalSmolVLMTriage._infer_temporal_verdict(
        request,
        ["A person is standing upright.", "The person is walking.", "The person is lying on the floor.", "The person is prone on the floor."],
    )

    assert verdict[0] == "real"
    assert verdict[1] == 0.7


def test_local_vlm_rejects_explicit_push_up_activity():
    request = TriageRequest(label="fall_detection", pre_count=2, frames=[b"1"] * 4)
    verdict = LocalSmolVLMTriage._infer_temporal_verdict(
        request,
        [
            "The person is standing upright.",
            "The person starts a push-up exercise.",
            "The person is prone on the floor.",
            "The person continues the push-up workout.",
        ],
    )

    assert verdict[0] == "false_positive"
    assert verdict[1] == 0.85


def test_local_vlm_does_not_call_sitting_a_fall():
    request = TriageRequest(label="fall_detection", pre_count=1, frames=[b"1"] * 2)
    verdict = LocalSmolVLMTriage._infer_temporal_verdict(request, ["The person is standing.", "The person is sitting on a chair."])

    assert verdict[0] == "uncertain"


def test_local_vlm_does_not_reason_about_other_alert_types():
    request = TriageRequest(label="intrusion", pre_count=1, frames=[b"1"] * 2)
    verdict = LocalSmolVLMTriage._infer_temporal_verdict(request, ["The person is standing.", "The person is lying on the floor."])

    assert verdict[0] == "uncertain"


def test_build_triage_client_builds_the_real_one_when_configured():
    settings = type(
        "S",
        (),
        {
            "vlm_enabled": True,
            "openai_api_key": "sk-test-not-used",
            "vlm_model": "gpt-4o-mini",
            "vlm_timeout_sec": 5.0,
        },
    )()
    # Construction only — no request is made, so no network call happens here.
    assert isinstance(build_triage_client(settings), OpenAIVLMTriage)


def test_prompt_carries_the_rule_evidence_verbatim():
    """The measured indicator values are the evidence; paraphrasing them loses the point."""
    reason = "2/3 indicators [+V_y=1.62H/s(>=1.2) -AR=0.83(>=1.2) +dH=65%(>=50%)]"
    prompt = OpenAIVLMTriage._build_prompt(
        TriageRequest(label="fall_detection", reason=reason, camera_name="Lobby", pre_count=4, frames=[b"x"] * 6)
    )

    assert reason in prompt
    assert "fall_detection" in prompt
    assert "Lobby" in prompt
    assert "push-ups" in prompt
    assert "uncontrolled" in prompt


def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Poll a condition set by a worker thread instead of sleeping a fixed amount."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


# ------------------------------------------------------------------ observation reconciliation
class _StubOpenAIClient:
    """Just enough of the OpenAI SDK surface for `OpenAIVLMTriage.assess`."""

    def __init__(self, content: str) -> None:
        message = type("M", (), {"content": content})()
        choice = type("C", (), {"message": message})()
        response = type("R", (), {"choices": [choice]})()

        class _Completions:
            def create(self, **_):
                return response

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _assess_with(content: str, frames: int = 6):
    triage = OpenAIVLMTriage.__new__(OpenAIVLMTriage)
    triage.model = "stub-vlm"
    triage._client = _StubOpenAIClient(content)
    return triage.assess(
        TriageRequest(label="fall_detection", reason="2/3 indicators", pre_count=4, frames=[b"x"] * frames)
    )


def test_a_count_mismatch_keeps_the_verdict_and_drops_the_notes():
    """The regression this guards against.

    A model that writes five notes for six frames used to lose its whole judgement: the
    count check raised, the blanket `except` caught it, and a confident "real" on a fall
    became `uncertain` with a zeroed confidence. The notes are an annotation; the verdict is
    the product.

    The notes go entirely, not trimmed to length: the response has no frame indices, so a
    short list could be missing its end or its middle, and a middle gap silently shifts
    every later BEFORE/AFTER caption onto the wrong frame.
    """
    verdict = _assess_with(
        json.dumps({
            "verdict": "real",
            "confidence": 0.95,
            "rationale": "Người đàn ông ngã xuống sàn và nằm yên.",
            "frame_observations": ["a", "b", "c", "d", "e"],
        })
    )

    assert verdict.verdict == "real"
    assert verdict.confidence == 0.95
    assert verdict.rationale == "Người đàn ông ngã xuống sàn và nằm yên."
    assert verdict.error is None
    assert verdict.frame_observations == []


def test_a_long_observation_list_also_drops_the_notes_not_truncates():
    verdict = _assess_with(
        json.dumps({
            "verdict": "false_positive",
            "confidence": 0.7,
            "rationale": "Người đang tập thể dục.",
            "frame_observations": ["a", "b", "c", "d", "e", "f", "g", "h"],
        })
    )

    assert verdict.verdict == "false_positive"
    assert verdict.frame_observations == []


def test_a_matching_observation_list_is_kept_verbatim():
    verdict = _assess_with(
        json.dumps({
            "verdict": "real",
            "confidence": 0.8,
            "rationale": "x",
            "frame_observations": ["a", "b", "c", "d", "e", "f"],
        })
    )

    assert verdict.frame_observations == ["a", "b", "c", "d", "e", "f"]


def test_malformed_json_still_fails_closed():
    """Reconciling a count is not the same as trusting a broken response: a reply that is not
    JSON at all carries no verdict to keep, and must still surface as an error."""
    verdict = _assess_with("{not json at all")

    assert verdict.verdict == "uncertain"
    assert verdict.confidence == 0.0
    assert verdict.error is not None


def test_the_failure_reason_reaches_the_history(make_service):
    """Every failure path lands on `uncertain` with an empty rationale, so without the error
    string the history cannot tell "the model could not decide" from "the provider was never
    reached" -- which is exactly how a misconfigured base URL failed silently for weeks."""
    service = make_service(triage=ExplodingTriage())
    alert = _emit(service)
    _wait_until(lambda: alert.triage_status == "complete")

    service.record_operator_verdict(alert.id, "real")

    row = service.history.all_rows()[0]
    assert row["vlm_verdict"] == "uncertain"
    assert "upstream is on fire" in row["vlm_error"]


def test_a_clean_assessment_records_no_error(make_service):
    service = make_service(triage=FakeTriage(verdict="real"))
    alert = _emit(service)
    _wait_until(lambda: alert.triage_status == "complete")

    service.record_operator_verdict(alert.id, "real")

    assert service.history.all_rows()[0]["vlm_error"] is None
