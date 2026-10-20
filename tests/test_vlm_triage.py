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

import cv2
import numpy as np
import pytest

from src.models.camera_schemas import Alert
from src.services import monitoring as monitoring_module
from src.services.alert_history import AlertHistory
from src.services.evidence_store import EvidenceStore
from src.services.monitoring import (
    MonitoringService,
    _crop_track,
    _evenly_spaced,
    _evidence_frames,
    embedding_text,
)
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


def _jpeg(width: int = 200, height: int = 100) -> bytes:
    """A real, decodable JPEG -- `_crop_track` needs something `cv2.imdecode` can read,
    unlike the placeholder `b"fake_jpeg"` bytes most other tests use."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", array)
    assert ok
    return buffer.tobytes()


def _ring_frame(
    stamp: float, jpeg: bytes, tracks: dict[str, tuple[float, float, float, float]] | None = None
) -> monitoring_module.RingFrame:
    """Build one ring entry. `tracks` defaults to empty -- most tests care about pre/post-roll
    splitting, not cropping, and an empty dict is exactly what a camera with no inference yet
    (or no track match) leaves in a real ring entry."""
    return monitoring_module.RingFrame(stamp, jpeg, tracks or {})


def _emit(service, camera_id="cam_1", label="fall_detection", reason="2/3 indicators") -> Alert:
    now = time.monotonic()
    ring = service._ring.setdefault(camera_id, deque(maxlen=service._default_ring_size))
    ring.append(_ring_frame(now - 1.0, b"fake_jpeg_before"))
    ring.append(_ring_frame(now + 0.1, b"fake_jpeg_after"))
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


def test_false_positive_verdict_is_not_telegram_eligible(make_service):
    service = make_service(triage=FakeTriage(verdict="false_positive"))
    published: list[tuple[str, dict]] = []
    service._publish = lambda event_type, data: published.append((event_type, data))

    alert = _emit(service)
    assert service._notification_worker.notifier.enabled is False
    assert service._notification_worker._queue.empty()
    assert alert.triage_status == "pending"

    assert service.notifier.enabled is False


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


# ------------------------------------------------------------------ post-roll scheduling
def test_submit_triage_returns_immediately_despite_the_post_roll_wait(make_service):
    """The post-roll wait must not block the caller — it runs on a `threading.Timer`,
    never inside `_submit_triage` itself, so a slow post-roll cannot stall the camera's
    processor thread that calls `_emit_alert`."""
    service = make_service(triage=FakeTriage(), post_roll_sec=2.0)

    started = time.perf_counter()
    _emit(service)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"_emit_alert (and thus _submit_triage) took {elapsed:.2f}s"


def test_verdict_still_arrives_after_the_post_roll_elapses(make_service):
    """The timer must actually fire and hand the alert to the triage pool."""
    triage = FakeTriage()
    service = make_service(triage=triage, post_roll_sec=0.2)

    alert = _emit(service)
    assert alert.triage_status == "pending"

    assert triage.called.wait(timeout=5), "triage never ran after the post-roll timer fired"
    _wait_until(lambda: alert.triage_status == "complete")
    assert alert.triage_verdict == "real"


def test_post_roll_wait_does_not_occupy_a_triage_pool_worker(make_service):
    """Regression guard for the bug this stage fixes: sleeping for the post-roll used to
    happen *inside* a `_triage_pool` worker (`_wait_for_post_roll`'s old `time.sleep`), so
    a handful of alerts firing close together could leave most of the pool asleep instead
    of answering API calls. All pool workers must be free to run `assess()` concurrently
    even while several alerts are still waiting out their post-roll window."""
    triage = FakeTriage(delay=0.3)
    service = make_service(triage=triage, post_roll_sec=0.3)

    # One more alert than there are pool workers, fired back to back. If the post-roll wait
    # occupied a pool worker, the pool would be fully consumed by sleeping alone and later
    # alerts would queue behind it instead of running concurrently once their timers fire.
    alert_count = monitoring_module._TRIAGE_POOL_WORKERS + 1
    for i in range(alert_count):
        _emit(service, camera_id=f"cam_{i}")

    _wait_until(lambda: triage.called.is_set(), timeout=5.0)
    _wait_until(lambda: len(triage.requests) == alert_count, timeout=5.0)
    assert len(triage.requests) == alert_count


def test_shutdown_cancels_pending_post_roll_timers(make_service):
    """A timer scheduled just before shutdown must not fire into a closed pool."""
    service = make_service(triage=FakeTriage(), post_roll_sec=5.0)

    _emit(service)
    assert len(service._pending_triage_timers) == 1

    service.shutdown()

    assert service._pending_triage_timers == set()


# ------------------------------------------------------------------ frame ring
def test_ring_holds_distinct_frames(make_service):
    """Guards the aliasing bug.

    The raw frame handed to `_on_raw_frame` may be the reader thread's shared buffer, so a
    ring of ndarrays would silently become N references to the newest frame — a "clip" of
    identical images that looks fine in every log line. Encoding to JPEG at push time is
    what prevents it.
    """
    service = make_service()
    for value in (10, 120, 240):
        # The feed is throttled to a few fps; reset the per-camera clock so all three land.
        service._ring_last_fed["cam_1"] = 0.0
        service._on_raw_frame("cam_1", _frame(value))

    stored = [frame.jpeg for frame in service._ring["cam_1"]]
    assert len(stored) == 3
    assert len(set(stored)) == 3, "ring collapsed to duplicate frames"


def test_clip_splits_into_pre_and_post_roll(make_service):
    """The split point is when the rule fired, not when the collector woke up."""
    service = make_service()
    service._post_roll_sec = 0.2
    service._pre_roll_sec = 5.0

    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", deque(maxlen=64))
    # Four frames before the event, two after (event is at now - 0.2).
    for offset, payload in [(-3.0, b"a"), (-2.0, b"b"), (-1.0, b"c"), (-0.5, b"d")]:
        ring.append(_ring_frame(now + offset, payload))
    for offset, payload in [(-0.1, b"e"), (0.0, b"f")]:
        ring.append(_ring_frame(now + offset, payload))

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


def test_a_one_sided_clip_is_still_assessed(make_service):
    """One-sided clips are now sent to the model, not refused here.

    A clip with only lead-up still shows the fall itself; one with only aftermath still
    shows whether the person stayed down. `pre_count` and the prompt's split line tell the
    model what it is missing, so it can answer `uncertain` itself — a better judgement than
    this branch refusing to call the API. On a slow CPU host, an event landing at the very
    end of the ring (all "before") is the common case, not an edge one.
    """
    triage = FakeTriage(verdict="real")
    service = make_service(triage=triage)
    service.triage = NullVLMTriage()
    alert = _emit(service)
    service.triage = triage
    # Two frames, both before the split point — `post` is empty.
    stale = time.monotonic() - service._post_roll_sec - 1.0
    service._ring["cam_1"] = deque(
        [_ring_frame(stale - 1.0, b"before1"), _ring_frame(stale, b"before2")], maxlen=8
    )
    service._triage_in_flight = 1

    service._run_triage(alert)

    assert alert.triage_verdict == "real"
    assert alert.triage_error is None
    assert len(triage.requests) == 1
    # Everything landed on the "before" side.
    assert triage.requests[0].pre_count == len(triage.requests[0].frames)


def test_a_single_frame_clip_is_refused(make_service):
    """The one genuinely unassessable case: too few frames to show any transition."""
    triage = FakeTriage(verdict="real")
    service = make_service(triage=triage)
    service.triage = NullVLMTriage()
    alert = _emit(service)
    service.triage = triage
    service._ring["cam_1"] = deque([_ring_frame(time.monotonic() + 1.0, b"only-one")], maxlen=8)
    service._triage_in_flight = 1

    service._run_triage(alert)

    assert alert.triage_verdict == "uncertain"
    assert "too low" in (alert.triage_error or "")
    assert triage.requests == []


def test_ring_capacity_covers_the_whole_clip_window(make_service):
    """The ring must hold pre_roll + post_roll seconds of frames at the throttled feed rate,
    with headroom, so nothing is evicted before _collect_clip reads it."""
    service = make_service()
    service._on_raw_frame("cam_1", _frame(10))

    assert service._ring["cam_1"].maxlen == service._default_ring_size
    window = service._pre_roll_sec + service._post_roll_sec
    assert service._default_ring_size >= monitoring_module._RING_FEED_FPS * window


def test_evenly_spaced_keeps_the_ends():
    """Taking the first N would describe the first half-second and nothing else."""
    items = [bytes([i]) for i in range(10)]

    assert _evenly_spaced(items, 4) == [b"\x00", b"\x03", b"\x06", b"\x09"]
    assert _evenly_spaced(items, 20) == items
    assert _evenly_spaced([], 4) == []
    assert _evenly_spaced(items, 0) == []


# ------------------------------------------------------------------ track-cropped clips
def test_collect_clip_crops_to_the_given_track(make_service):
    """The whole point of stage 3: a clip asked for one track_id isolates that person."""
    service = make_service()
    jpeg = _jpeg(width=200, height=100)
    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", deque(maxlen=64))
    ring.append(_ring_frame(now - 1.0, jpeg, tracks={"person_1": (0.1, 0.1, 0.4, 0.9)}))
    ring.append(_ring_frame(now + 0.1, jpeg, tracks={"person_1": (0.1, 0.1, 0.4, 0.9)}))

    full_frames, _ = service._collect_clip("cam_1", track_id=None)
    cropped_frames, _ = service._collect_clip("cam_1", track_id="person_1")

    assert cropped_frames != full_frames
    assert all(frame != jpeg for frame in cropped_frames)
    # The crop must actually be smaller -- it is a sub-region of the original image.
    for frame in cropped_frames:
        decoded = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[1] < 200 or decoded.shape[0] < 100


def test_collect_clip_with_no_track_id_matches_pre_crop_behavior(make_service):
    """Backward compatibility: omitting track_id must reproduce the old, uncropped clip."""
    service = make_service()
    jpeg = _jpeg()
    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", deque(maxlen=64))
    ring.append(_ring_frame(now - 1.0, jpeg, tracks={"person_1": (0.1, 0.1, 0.4, 0.9)}))
    ring.append(_ring_frame(now + 0.1, jpeg, tracks={"person_1": (0.1, 0.1, 0.4, 0.9)}))

    frames, pre_count = service._collect_clip("cam_1")

    assert frames == [jpeg, jpeg]
    assert pre_count == 1


def test_collect_clip_falls_back_to_full_frame_when_the_track_is_missing(make_service):
    """A track absent from a frame (tracking gap, or an id that never existed here) must not
    raise or drop the frame -- it degrades to that frame's full scene."""
    service = make_service()
    jpeg = _jpeg()
    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", deque(maxlen=64))
    ring.append(_ring_frame(now - 1.0, jpeg, tracks={"person_other": (0.1, 0.1, 0.4, 0.9)}))
    ring.append(_ring_frame(now + 0.1, jpeg))

    frames, _ = service._collect_clip("cam_1", track_id="person_1")

    assert frames == [jpeg, jpeg]


def test_crop_track_returns_none_below_the_minimum_useful_size(make_service):
    """A crop too small to carry information must not be sent as if it were evidence."""
    jpeg = _jpeg(width=200, height=100)
    tiny_box = (0.0, 0.0, 0.01, 0.01)  # a couple of pixels wide even with padding

    assert _crop_track(jpeg, tiny_box) is None


def test_triage_request_is_marked_cropped_when_the_alert_has_a_track(make_service):
    """The prompt annotation (`TriageRequest.cropped`) must track whether a crop was
    attempted, not just whether the ring happened to have a match for every frame."""
    triage = FakeTriage()
    service = make_service(triage=triage)
    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", deque(maxlen=service._default_ring_size))
    ring.append(_ring_frame(now - 1.0, _jpeg(), tracks={"person_1": (0.1, 0.1, 0.4, 0.9)}))
    ring.append(_ring_frame(now + 0.1, _jpeg(), tracks={"person_1": (0.1, 0.1, 0.4, 0.9)}))

    service._emit_alert(
        camera_id="cam_1",
        label="fall_detection",
        confidence=0.9,
        frame=_frame(),
        reason="2/3 indicators",
        track_id="person_1",
    )

    assert triage.called.wait(timeout=5)
    assert triage.requests[0].cropped is True


def test_triage_request_is_not_marked_cropped_without_a_track_id(make_service):
    """An alert with no track_id (e.g. abandoned_object has none) must not claim a crop."""
    triage = FakeTriage()
    service = make_service(triage=triage)

    _emit(service)  # _emit's alert carries no track_id

    assert triage.called.wait(timeout=5)
    assert triage.requests[0].cropped is False


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


# ------------------------------------------------------------------ similar-incident text
def test_embedding_text_drops_the_zone_name(make_service):
    """The zone name is an operator-chosen landmark like 'Kho hàng chính'. Left in a
    fifteen-word sentence it dominates the vector, and the lookup starts matching *same
    camera* instead of *same behaviour* — which is what it was actually doing in production.
    """
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service, label="intrusion", reason="t3 entered restricted zone 'Kho hàng chính' during 24/7")

    text = embedding_text(alert)

    assert "Kho hàng chính" not in text
    assert "a restricted zone" in text
    # The rest of the sentence has to survive, or there is no behaviour left to match on.
    assert "t3 entered" in text
    assert "during 24/7" in text


def test_embedding_text_includes_the_vlm_reading(make_service):
    """The VLM's description was computed, shown to the guard, and then left out of the
    comparison entirely. It is the richest text in the system, so it belongs in the vector."""
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service)
    alert.triage_rationale = "Người đàn ông ngã xuống sàn và nằm yên."
    alert.triage_frame_observations = ["Người đứng thẳng.", "Người nằm trên sàn."]

    text = embedding_text(alert)

    assert "2/3 indicators" in text
    assert "Người đàn ông ngã xuống sàn" in text
    assert "Người nằm trên sàn." in text


def test_embedding_text_without_a_vlm_is_unchanged(make_service):
    """A deployment with VLM_ENABLED off must produce exactly the text it did before this
    existed, or its stored vectors stop being comparable for a feature it does not even use."""
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service)

    assert embedding_text(alert) == "[rule] 2/3 indicators"


def test_similar_incidents_needs_no_reason_when_the_vlm_spoke(make_service):
    """The old guard was `if not alert.reason: return []`. An alert with no rule reason but a
    full VLM reading has plenty to compare on, and silently returned nothing."""
    service = make_service(triage=NullVLMTriage())
    alert = _emit(service, reason=None)
    alert.triage_rationale = "Người đàn ông ngã xuống sàn."

    assert embedding_text(alert) == "[ai] Người đàn ông ngã xuống sàn."


def test_triage_examples_is_only_recent_history_when_embeddings_are_off(make_service):
    """The default. `self.embeddings`/`self._embed_text` are None in every test, so the
    similar-incident lookup contributes nothing and behaviour is exactly as before."""
    service = make_service(triage=NullVLMTriage())
    service.history.record("old", "cam_1", "fall_detection", "false", reason="camera-local prior")
    alert = _emit(service)

    examples = service._triage_examples(alert)

    assert len(examples) == 1
    assert "camera-local prior" in examples[0]


def test_triage_examples_prepends_similar_incidents_when_embeddings_are_on(make_service):
    """With embeddings on, the pgvector neighbours lead and this camera's own history
    trails, so the newest on-point judgement is the last thing the model reads."""
    service = make_service(triage=NullVLMTriage())
    service.history.record("old", "cam_1", "fall_detection", "false", reason="camera-local prior")
    alert = _emit(service)

    # Stand in for a configured pgvector store: the guards check both are non-None, and
    # `similar_incidents` itself is replaced with the rows it would have returned.
    service.embeddings = object()
    service._embed_text = lambda _text: ([0.0], "stub-model")
    service.similar_incidents = lambda alert_id, limit, min_similarity: [
        {"similarity": 0.91, "camera_name": "Sảnh chính", "operator_verdict": "false",
         "content": "[rule] người ngồi xổm"},
    ]

    examples = service._triage_examples(alert)

    assert len(examples) == 2
    assert "giống 91%" in examples[0]
    assert "BÁO ĐỘNG GIẢ" in examples[0]
    assert "camera-local prior" in examples[-1]


# ------------------------------------------------------------------ neighbour frame strips
def test_incident_frames_rebuilds_a_stored_strip(make_service, tmp_path):
    """`AlertEvidenceFrame` is never persisted, so a past incident's strip has to be
    reconstructed from the `{alert_id}-f{n}` keys the writer derives."""
    service = make_service(triage=NullVLMTriage())
    _evidence_frames([b"a", b"b", b"c"], 2, "alert-9", service.evidence_store)

    frames = service.incident_frames("alert-9")

    assert [frame.index for frame in frames] == [1, 2, 3]
    # Two thirds before, matching the budget the live strip is sampled with.
    assert [frame.phase for frame in frames] == ["before", "before", "after"]
    assert frames[0].image_url == "/api/v1/evidence/alert-9-f1"


def test_incident_frames_is_empty_when_nothing_is_stored(make_service):
    """An older incident whose frames have aged out is a panel with no strip, not an error."""
    service = make_service(triage=NullVLMTriage())
    assert service.incident_frames("never-stored") == []


def test_incident_frames_stops_at_the_first_gap(make_service):
    """Bounds the probe. Without the stop, a missing f1 would still cost `vlm_max_frames`
    storage round trips — per neighbour, and on S3 those are network calls."""
    service = make_service(triage=NullVLMTriage())
    _evidence_frames([b"a", b"b"], 1, "alert-8", service.evidence_store)
    service.evidence_store.local_path("alert-8-f1").unlink()

    assert service.incident_frames("alert-8") == []


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


def test_thinking_mode_is_disabled_only_for_a_qwen3_model_on_a_compatible_endpoint():
    """qwen3* models default to a thinking mode that adds 8-25s/alert; `enable_thinking: False`
    is a DashScope-only extension. The vision models (qwen-vl-*) don't run thinking mode and
    some reject the field, and the real OpenAI API rejects any unknown field."""
    base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen3_flash = OpenAIVLMTriage(api_key="sk-x", model="qwen3.7-flash", base_url=base)
    qwen_vl_plus = OpenAIVLMTriage(api_key="sk-x", model="qwen-vl-plus", base_url=base)
    openai = OpenAIVLMTriage(api_key="sk-x", model="gpt-4o-mini")
    qwen3_no_base_url = OpenAIVLMTriage(api_key="sk-x", model="qwen3.7-flash")

    assert qwen3_flash._extra_body == {"enable_thinking": False}
    assert qwen_vl_plus._extra_body == {}
    assert openai._extra_body == {}
    assert qwen3_no_base_url._extra_body == {}


def test_assess_passes_extra_body_through_to_the_client():
    captured = {}

    class _CapturingClient:
        def __init__(self) -> None:
            message = type("M", (), {"content": '{"verdict": "uncertain", "confidence": 0.0, '
                                                '"rationale": "x", "frame_observations": []}'})()
            choice = type("C", (), {"message": message})()
            response = type("R", (), {"choices": [choice]})()

            class _Completions:
                def create(self, **kwargs):
                    captured.update(kwargs)
                    return response

            self.chat = type("Chat", (), {"completions": _Completions()})()

    triage = OpenAIVLMTriage.__new__(OpenAIVLMTriage)
    triage.model = "qwen3.7-flash"
    triage._client = _CapturingClient()
    triage._extra_body = {"enable_thinking": False}
    triage.assess(TriageRequest(label="fall_detection", pre_count=1, frames=[b"x", b"y"]))

    assert captured["extra_body"] == {"enable_thinking": False}


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


def test_prompt_tells_the_model_to_defer_to_contradicting_judgements():
    """Prior judgements are not just context — a consistent contradiction should pull the
    model's confidence down."""
    with_examples = OpenAIVLMTriage._build_prompt(
        TriageRequest(
            label="fall_detection",
            camera_name="Lobby",
            pre_count=2,
            frames=[b"x"] * 4,
            examples=["Bảo vệ kết luận là BÁO ĐỘNG GIẢ: người ngồi xổm"],
        )
    )
    without = OpenAIVLMTriage._build_prompt(
        TriageRequest(label="fall_detection", camera_name="Lobby", pre_count=2, frames=[b"x"] * 4)
    )

    assert "hạ confidence" in with_examples
    assert "mâu thuẫn với một xu hướng nhất quán" in with_examples
    assert "hạ confidence" not in without


def test_system_prompt_describes_the_confidence_bands_and_the_suppression_cost():
    """Confidence is self-reported and drives auto-suppression, so the prompt has to say
    what the numbers mean and that a wrong high-confidence false_positive is the worst call."""
    from src.services.vlm import _SYSTEM_PROMPT

    assert "0.9-1.0" in _SYSTEM_PROMPT
    assert "tự động ẩn" in _SYSTEM_PROMPT


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


# ------------------------------------------------------------------ left objects: before/after
def test_a_background_model_abandoned_alert_sends_its_before_after_last(make_service):
    from types import SimpleNamespace

    from src.vision.schemas import BoundingBox

    triage = FakeTriage()
    service = make_service(triage=triage)
    background = np.full((60, 80, 3), 200, dtype=np.uint8)
    service.manager.background_image = lambda camera_id, width, height: background
    now = time.monotonic()
    ring = service._ring.setdefault("cam_1", deque(maxlen=service._default_ring_size))
    ring.append(_ring_frame(now - 1.0, b"fake_jpeg_before"))
    ring.append(_ring_frame(now + 0.1, b"fake_jpeg_after"))

    event = SimpleNamespace(
        label="abandoned_object",
        confidence=1.0,
        reason="static_object (static_3) unattended for 2.0s with none of its 1 owner(s) within 20px",
        zone_name=None,
        track_id="static_3",
        bbox=BoundingBox(xmin=30, ymin=20, xmax=40, ymax=30),
    )
    service._on_incident("cam_1", SimpleNamespace(events=[event]), _frame())
    alert = service._alerts[0]
    assert alert.triage_comparison is not None
    assert "triage_comparison" not in alert.model_dump()
    assert triage.called.wait(timeout=5)

    request = triage.requests[0]
    assert request.frames == [b"fake_jpeg_before", b"fake_jpeg_after", alert.triage_comparison]
    assert request.comparison_last and not request.cropped
    comparison = cv2.imdecode(np.frombuffer(alert.triage_comparison, np.uint8), cv2.IMREAD_COLOR)
    assert comparison.shape[0] == monitoring_module._COMPARISON_HEIGHT
    # Left half is the background (200), right half the frame (0), apart from the outlines.
    assert comparison[150:170, 5:15].mean() > 150 and comparison[150:170, -15:-5].mean() < 50


def test_other_alerts_get_no_before_after(make_service):
    from types import SimpleNamespace

    from src.vision.schemas import BoundingBox

    service = make_service()
    service.manager.background_image = lambda camera_id, width, height: np.zeros((60, 80, 3), np.uint8)
    box = BoundingBox(xmin=30, ymin=20, xmax=40, ymax=30)
    detected_bag = SimpleNamespace(label="abandoned_object", reason="suitcase (track_9) unattended", bbox=box)
    fall = SimpleNamespace(label="fall_detection", reason="static_object-like text, wrong label", bbox=box)
    assert service._left_object_comparison("cam_1", detected_bag, _frame()) is None
    assert service._left_object_comparison("cam_1", fall, _frame()) is None


def test_the_prompt_explains_the_before_after_frame():
    request = TriageRequest(label="abandoned_object", frames=[b"a", b"b"], pre_count=1, comparison_last=True)
    prompt = OpenAIVLMTriage._build_prompt(request)
    assert "before the object appeared" in prompt
    assert "before the object appeared" not in OpenAIVLMTriage._build_prompt(
        TriageRequest(label="abandoned_object", frames=[b"a", b"b"], pre_count=1)
    )
