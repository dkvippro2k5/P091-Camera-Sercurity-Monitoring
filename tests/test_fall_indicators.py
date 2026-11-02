"""Tests for the three-indicator bounding-box fall filter.

No weights, no pose model, no decoded frames — every case is a hand-built box trajectory,
so the whole file runs in milliseconds and needs nothing that is gitignored.

Frames are fed ~0.2s apart throughout, which is not cosmetic. All three indicators measure
change across a 0.5s window (`_VELOCITY_WINDOW_SEC`), so a test that jumps the clock by
seconds between frames leaves a single sample per window and measures nothing at all. 0.2s
is also roughly the real per-camera cadence: ~5 fps at 720p on 8 CPU cores.
"""

from datetime import datetime

import pytest

from src.vision.fall_motion import FallMotionEvidence
from src.vision.fall_pose import PoseObservation
from src.vision.rule_engine import LABEL_FALL, RuleEngine
from src.vision.schemas import BoundingBox, IncidentDetection

WIDTH, HEIGHT = 640, 480
FRAME_DT = 0.2

# Standing and flat boxes share a feet line at y=320 (150+170 == 270+50), which is what a
# real fall looks like: the head travels a body height while the feet barely move.
STAND_W, STAND_H, STAND_TOP = 50, 170, 150
FLAT_W, FLAT_H, FLAT_TOP = 170, 50, 270


def box(width: int, height: int, y_top: int, track_id: str = "person_1") -> IncidentDetection:
    """A person detection at a given size and vertical position."""
    return IncidentDetection(
        label="person",
        track_id=track_id,
        confidence=0.9,
        bbox=BoundingBox(xmin=200, ymin=y_top, xmax=200 + width, ymax=y_top + height),
    )


def standing(track_id: str = "person_1") -> IncidentDetection:
    return box(STAND_W, STAND_H, STAND_TOP, track_id)


def flat(track_id: str = "person_1") -> IncidentDetection:
    return box(FLAT_W, FLAT_H, FLAT_TOP, track_id)


def run(
    engine: RuleEngine,
    frames: list[tuple[float, IncidentDetection]],
    camera_id: str | None = None,
) -> list[IncidentDetection]:
    """Feed (timestamp, detection) pairs in order and return every event produced."""
    now = datetime.now()
    events: list[IncidentDetection] = []
    for timestamp, det in frames:
        events.extend(engine.evaluate([det], WIDTH, HEIGHT, timestamp, now, camera_id))
    return events


def hold(
    det: IncidentDetection, start: float, seconds: float, dt: float = FRAME_DT
) -> list[tuple[float, IncidentDetection]]:
    """Identical frames for `seconds` after `start` — someone lying still, or standing still.

    `round` rather than `int`: 3.4 / 0.2 is 16.999... in binary floating point, and
    truncating it would quietly drop the last frame the dwell assertions depend on.
    """
    return [(start + i * dt, det) for i in range(1, int(round(seconds / dt)) + 1)]


def collapse(down: IncidentDetection | None = None) -> list[tuple[float, IncidentDetection]]:
    """Two upright frames then one collapsed frame, all inside a single 0.5s window."""
    return [(0.0, standing()), (FRAME_DT, standing()), (2 * FRAME_DT, down or flat())]


def scoring_engine(**kwargs) -> RuleEngine:
    """An engine with the PRD dwell disabled, for tests about *what arms the suspicion*.

    The production default is 3.0s (PRD 3.1). Tests that are asking whether a trajectory
    scores 2 of 3 shouldn't also have to stage 3 seconds of frames to find out — and a
    test that did would fail for either reason, so a scoring bug and a timer bug would be
    indistinguishable. The dwell itself is covered by the latch tests below.
    """
    return RuleEngine(fall_dwell_sec=0.0, **kwargs)


# ---------------------------------------------------------------------------
# The three indicators
# ---------------------------------------------------------------------------


def test_sudden_collapse_trips_all_three_indicators():
    events = run(scoring_engine(), collapse())

    assert len(events) == 1
    assert events[0].label == LABEL_FALL
    assert events[0].track_id == "person_1"
    reason = events[0].reason or ""
    assert "3/3 indicators" in reason
    assert "+V_y=" in reason and "+AR=" in reason and "+dH=" in reason
    assert "foot(" in reason and "aA=" in reason


def test_optical_flow_evidence_reaches_the_fall_event_reason():
    engine = scoring_engine()
    now = datetime.now()
    engine.evaluate([standing()], WIDTH, HEIGHT, 0.0, now)
    engine.evaluate([standing()], WIDTH, HEIGHT, FRAME_DT, now)
    events = engine.evaluate(
        [flat()],
        WIDTH,
        HEIGHT,
        2 * FRAME_DT,
        now,
        fall_motion_evidence={
            "person_1": FallMotionEvidence(
                downward_flow_ratio=0.8,
                downward_point_fraction=0.9,
                tracked_points=12,
            )
        },
    )

    assert len(events) == 1
    assert "+flow=0.80H/s" in (events[0].reason or "")


def test_wide_box_alone_is_not_a_fall():
    """The crouch-under-a-wide-lens case: aspect ratio says fall, nothing actually moved.

    This is the false positive pose confirmation used to absorb. With pose gone, requiring
    two indicators is what rejects it — a box that holds still has no velocity and loses no
    height, so it scores 1 of 3 no matter how flat it looks.

    The static-lying layer is switched off here: this pins what the *transition* layer
    rejects. The static layer does fire on a crouch held this long; see its tests below.
    """
    crouched = box(200, 150, 200)  # AR 1.33, comfortably over the 1.2 threshold
    assert run(scoring_engine(fall_static_dwell_sec=0.0), [(0.0, crouched)] + hold(crouched, 0.0, 3.0)) == []


def test_lowering_yourself_slowly_does_not_fire():
    """Sitting down over two seconds: the same end pose, none of the suddenness.

    Every indicator is rate-limited by the window, so a gradual descent never crosses one:
    the 70px the head travels spreads to ~14px per window, and height loss to ~12%.
    """
    frames = [
        (
            i * FRAME_DT,
            box(
                width=int(STAND_W + 30 * (i / 10)),
                height=int(STAND_H - 70 * (i / 10)),
                y_top=int(STAND_TOP + 70 * (i / 10)),
            ),
        )
        for i in range(11)
    ]
    assert run(scoring_engine(), frames) == []


def test_fall_away_from_the_camera_arms_on_two_indicators():
    """Velocity and height collapse, with the box never going flat.

    Someone dropping straight toward or away from the lens keeps a narrow box, so aspect
    ratio never crosses 1.2. This is precisely the fall a pure aspect-ratio rule misses, and
    it is why the default is 2 of 3 rather than 3: at 3 this case is silently dropped.
    """
    crumpled = box(50, 60, 260)  # AR 0.83 — neither walking nor flat
    frames = collapse(crumpled)

    assert len(run(scoring_engine(), frames)) == 1
    assert run(scoring_engine(fall_min_indicators=3), frames) == []


def test_fall_toward_camera_uses_center_descent_and_area_growth():
    """Perspective enlargement is evidence when it accompanies a rapid downward centre."""
    toward_lens = box(100, 220, 180)  # larger and still narrow; centre drops despite perspective
    frames = [(0.0, standing()), (FRAME_DT, standing()), (2 * FRAME_DT, toward_lens)]

    events = run(scoring_engine(), frames)

    assert len(events) == 1
    reason = events[0].reason or ""
    assert "+motion" in reason
    assert "+dA=" in reason
    assert "-AR=" in reason
    assert "-dH=" in reason


def test_walking_toward_camera_without_downward_center_does_not_fire():
    larger_upright = box(100, 300, 40)
    frames = [(0.0, standing()), (FRAME_DT, standing()), (2 * FRAME_DT, larger_upright)]

    assert run(scoring_engine(), frames) == []


def test_the_transition_layer_never_fires_on_someone_already_on_the_ground():
    """Indicators 1 and 3 are *changes*, so someone the camera first sees lying still scores
    aspect ratio alone: 1 of 3. The transition layer detects *falling*, not *lying*; anyone
    who fell before the camera saw them is the static-lying layer's job (tests below).
    """
    assert run(scoring_engine(fall_static_dwell_sec=0.0), [(0.0, flat())] + hold(flat(), 0.0, 5.0)) == []


# ---------------------------------------------------------------------------
# Arm / sustain latch
# ---------------------------------------------------------------------------


def test_dwell_survives_the_indicators_decaying_to_nothing():
    """Why the rule latches instead of re-scoring every frame.

    Within 0.5s of landing, the window holds only post-impact frames: V_y and ΔH read zero
    and the live score falls to 1 of 3. Re-scoring per frame and demanding the score hold
    would make *any* dwell longer than the window mathematically unreachable — the PRD's 3
    seconds included — so the rule would look tuned while never firing.
    """
    engine = RuleEngine(fall_dwell_sec=3.0)
    events = run(engine, collapse() + hold(flat(), 2 * FRAME_DT, 3.4))

    assert len(events) == 1
    assert events[0].timestamp == pytest.approx(3.4, abs=FRAME_DT)
    # Fired on a frame whose live score was 1/3; the evidence is the snapshot taken when the
    # suspicion armed, which is the only moment the transition was still measurable.
    assert "3/3 indicators" in (events[0].reason or "")


def test_getting_back_up_before_the_dwell_drops_the_suspicion():
    engine = RuleEngine(fall_dwell_sec=3.0)
    frames = (
        collapse()  # armed at t=0.4
        + hold(flat(), 2 * FRAME_DT, 1.0)  # a second on the ground, dwell not met
        + hold(standing(), 1.4, 4.0)  # upright again, well past the 3s mark
    )
    assert run(engine, frames) == []


def test_upright_shape_alone_does_not_count_as_recovery():
    """A tracker mismatch can be narrow without being the fallen person standing up."""
    engine = RuleEngine(fall_dwell_sec=1.0)
    mismatched_upright = box(20, 70, STAND_TOP)
    frames = collapse() + hold(flat(), 2 * FRAME_DT, 0.4) + hold(mismatched_upright, 0.8, 1.0)

    events = run(engine, frames)

    assert events == []
    state = engine._tracks[(None, "person_1")]
    assert state.fall_phase == "upright"


def test_upright_baseline_uses_a_median_to_resist_one_noisy_box():
    engine = RuleEngine()
    frames = [
        (0.0, standing()),
        (FRAME_DT, box(STAND_W + 2, STAND_H - 2, STAND_TOP + 1)),
        (2 * FRAME_DT, box(10, STAND_H * 2, 20)),
    ]

    run(engine, frames)

    baseline = engine._tracks[(None, "person_1")].upright_baseline
    assert baseline is not None
    assert baseline.height == pytest.approx(STAND_H)
    assert baseline.top == pytest.approx(STAND_TOP)


def test_fall_state_progresses_and_rearms_only_after_recovery():
    engine = RuleEngine(fall_dwell_sec=0.4)
    now = datetime.now()

    engine.evaluate([standing()], WIDTH, HEIGHT, 0.0, now)
    state = engine._tracks[(None, "person_1")]
    assert state.fall_phase == "upright"

    engine.evaluate([standing()], WIDTH, HEIGHT, FRAME_DT, now)
    engine.evaluate([flat()], WIDTH, HEIGHT, 2 * FRAME_DT, now)
    engine.evaluate([flat()], WIDTH, HEIGHT, 4 * FRAME_DT, now)
    assert state.fall_phase in {"prone", "persistent"}

    engine.evaluate([flat()], WIDTH, HEIGHT, 6 * FRAME_DT, now)
    assert state.fall_phase == "persistent"

    # One upright-looking frame is not a recovery: it must hold for persistence_grace_sec.
    engine.evaluate([standing()], WIDTH, HEIGHT, 7 * FRAME_DT, now)
    assert state.fall_phase == "persistent"
    for step in (8, 9, 10):
        engine.evaluate([standing()], WIDTH, HEIGHT, step * FRAME_DT, now)
    assert state.fall_phase == "recovered"
    engine.evaluate([standing()], WIDTH, HEIGHT, 11 * FRAME_DT, now)
    assert state.fall_phase == "upright"


def test_one_upright_looking_frame_mid_descent_does_not_latch_recovered():
    """UR Fall fall-01: the indicators arm as the person starts to pitch forward, while the
    box is still tall and narrow. That next frame used to read as "stood back up", latching
    "recovered", and the real landing a moment later could never re-arm."""
    engine = RuleEngine(fall_dwell_sec=0.4)
    now = datetime.now()

    engine.evaluate([standing()], WIDTH, HEIGHT, 0.0, now)
    engine.evaluate([standing()], WIDTH, HEIGHT, FRAME_DT, now)
    engine.evaluate([flat()], WIDTH, HEIGHT, 2 * FRAME_DT, now)
    state = engine._tracks[(None, "person_1")]
    assert state.fall_phase != "upright"

    engine.evaluate([standing()], WIDTH, HEIGHT, 3 * FRAME_DT, now)  # one tall frame mid-fall
    assert state.fall_phase != "recovered"

    events = []
    for step in range(4, 12):
        events += engine.evaluate([flat()], WIDTH, HEIGHT, step * FRAME_DT, now)
    assert len(events) == 1


def test_the_ambiguous_aspect_band_holds_the_timer_instead_of_flapping():
    """AR between the spec's walking figure (0.7) and its fallen figure (1.2) is evidence of
    neither, so it must not disarm. A fallen person's box drifts through that band as limbs
    move and the tracker jitters; treating that as "stood up" would reset the dwell on noise.
    """
    engine = RuleEngine(fall_dwell_sec=3.0)
    ambiguous = box(90, 100, 220)  # AR 0.90
    events = run(engine, collapse() + hold(ambiguous, 2 * FRAME_DT, 3.4))
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Velocity: scale, sign, and the dt guard
# ---------------------------------------------------------------------------


def test_velocity_threshold_is_scale_invariant():
    """The same fall at a quarter the on-screen size, i.e. down the far end of a corridor.

    Velocity is divided by the person's own box height, so both trip indicator 1. A raw
    pixel-per-second threshold tuned near the lens would be ~4x too high here, and every
    fall away from the camera would be invisible — the classic way this rule fails in the
    field while passing on close-up test footage.
    """
    far = [
        (0.0, box(13, 43, 150)),
        (FRAME_DT, box(13, 43, 150)),
        (2 * FRAME_DT, box(43, 13, 180)),
    ]
    assert len(run(scoring_engine(), collapse())) == 1
    assert len(run(scoring_engine(), far)) == 1


def test_standing_up_quickly_is_not_a_fall():
    """Image Y grows downward, so falling makes y_top *increase*.

    The spec's "Y giảm đột ngột" describes physical elevation dropping. Implementing it as
    the pixel value decreasing inverts the sign and yields a stand-up detector — which
    would still fire on footage of people falling (they get up afterwards) and so would
    pass a careless end-to-end check.
    """
    frames = [(0.0, flat()), (FRAME_DT, flat()), (2 * FRAME_DT, standing())]
    assert run(scoring_engine(), frames + hold(standing(), 2 * FRAME_DT, 2.0)) == []


def test_samples_closer_than_the_guard_report_no_velocity():
    """Two frames a hair apart make dt tiny, and dividing by it turns a pixel of tracker
    jitter into a triple-digit velocity. Below the guard the indicator abstains rather than
    guessing.
    """
    engine = scoring_engine()
    now = datetime.now()
    dropped = box(STAND_W, STAND_H, STAND_TOP + 100)

    engine.evaluate([standing()], WIDTH, HEIGHT, 0.0, now)
    engine.evaluate([dropped], WIDTH, HEIGHT, 0.02, now)

    state = engine._tracks[(None, "person_1")]
    score, breakdown = engine._fall_indicators(dropped, state, 0.02)
    # Unguarded this reads ~29 body-heights/sec off a 100px step in 20ms.
    assert "V_y=n/a" in breakdown
    assert score == 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_indicator_state_is_per_camera():
    """Two YOLO trackers both emit 'person_1'. Without the camera in the key, one camera's
    upright frames would keep clearing the other camera's suspicion.
    """
    engine = scoring_engine()
    now = datetime.now()
    events_a: list[IncidentDetection] = []
    events_b: list[IncidentDetection] = []

    for timestamp, det in collapse():
        events_a.extend(engine.evaluate([det], WIDTH, HEIGHT, timestamp, now, "cam_a"))
        events_b.extend(engine.evaluate([standing()], WIDTH, HEIGHT, timestamp, now, "cam_b"))

    assert len(events_a) == 1
    assert events_a[0].camera_id == "cam_a"
    assert events_b == []


def test_reason_reports_the_indicators_that_did_not_fire():
    """The PRD routes `reason` to an LLM for false-positive filtering, and a bare verdict
    gives that layer nothing to weigh. The breakdown carries all three indicators with their
    measured values and a +/- for whether each crossed, so a 2-of-3 alert says which one it
    was missing.
    """
    events = run(scoring_engine(), collapse(box(50, 60, 260)))

    assert len(events) == 1
    reason = events[0].reason or ""
    assert "2/3 indicators" in reason
    assert "-AR=0.83" in reason  # measured and reported even though it did not cross
    assert "+V_y=" in reason and "+dH=" in reason


def test_the_default_dwell_is_the_prds_three_seconds():
    """The shipped default matches the documented three-second requirement."""
    engine = RuleEngine()
    assert engine.fall_dwell_sec == 3.0

    # The collapse arms the latch. On its own that produces nothing — the 3s has not elapsed.
    assert run(engine, collapse()) == []

    # Held on the ground past the dwell, it fires exactly once — on the first frame at or
    # after arm time + 3s, not at the end of the run.
    events = run(engine, hold(flat(), 2 * FRAME_DT, 3.2))
    assert len(events) == 1
    assert events[0].label == LABEL_FALL
    assert events[0].timestamp == pytest.approx(2 * FRAME_DT + 3.0, abs=FRAME_DT)


# ---------------------------------------------------------------------------
# Descent speed: lying down on purpose is slow, falling is not
# ---------------------------------------------------------------------------


def _slow_lie_down() -> list[tuple[float, IncidentDetection]]:
    """Box flattens into a lying shape while its top creeps down 12 px per 0.2 s (~0.35 H/s)."""
    top = STAND_TOP
    frames = [(i * FRAME_DT, standing()) for i in range(4)]
    for i, (width, height) in enumerate(((STAND_W + 30, 140), (STAND_W + 90, 100), (FLAT_W, FLAT_H)), 1):
        frames.append(((3 + i) * FRAME_DT, box(width, height, top + 12 * i)))
    lying = box(FLAT_W, FLAT_H, top + 36)
    return frames + hold(lying, 6 * FRAME_DT, 4.0)


def test_with_the_gate_on_a_slow_lie_down_that_reaches_the_dwell_is_not_reported():
    engine = RuleEngine(fall_min_descent_speed=0.5)
    events = run(engine, _slow_lie_down())

    state = engine._tracks[(None, "person_1")]
    assert events == []
    assert state.fall_phase == "prone"  # armed and down for the full dwell...
    assert state.fall_suppressed_slow  # ...but stopped by the speed gate
    assert state.descent_peak_speed < engine.fall_min_descent_speed


def test_the_descent_speed_gate_is_off_by_default():
    """On held-out CAUCAFall the gate halved recall (46% -> 24%): staged falls are slow too."""
    assert len(run(RuleEngine(), _slow_lie_down())) == 1


def test_with_the_gate_on_a_fast_collapse_still_fires_and_reports_its_descent_speed():
    engine = RuleEngine(fall_min_descent_speed=0.5)
    events = run(engine, collapse() + hold(flat(), 2 * FRAME_DT, 3.4))

    assert len(events) == 1
    assert engine._tracks[(None, "person_1")].descent_peak_speed >= engine.fall_min_descent_speed
    assert "peak descent" in events[0].reason


# ---------------------------------------------------------------------------
# Static-lying layer: lying for the dwell with no transition seen
# ---------------------------------------------------------------------------


def _bed(xmin: int = 150, ymin: int = 240, xmax: int = 450, ymax: int = 360) -> IncidentDetection:
    """A bed under flat(), whose box spans x 200-370, y 270-320."""
    return IncidentDetection(
        label="bed", track_id="bed_1", confidence=0.9, bbox=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    )


def test_someone_first_seen_lying_fires_once_after_the_dwell():
    """The coverage gap the transition layer leaves: a fall from sitting, or a track first
    picked up on the floor. On GMDCSA24 these were most of the falls step 1 still missed."""
    events = run(RuleEngine(), [(0.0, flat())] + hold(flat(), 0.0, 5.0))

    assert len(events) == 1
    assert events[0].timestamp == pytest.approx(3.0)
    assert "static layer" in events[0].reason


def test_lying_on_a_bed_does_not_fire():
    engine = RuleEngine()
    now = datetime.now()
    events = []
    for i in range(26):  # 5 s
        events += engine.evaluate([flat(), _bed()], WIDTH, HEIGHT, i * FRAME_DT, now)
    assert events == []


def test_a_bed_seen_once_still_counts_while_its_detection_flickers():
    engine = RuleEngine()
    now = datetime.now()
    events = engine.evaluate([flat(), _bed()], WIDTH, HEIGHT, 0.0, now)
    for i in range(1, 26):
        events += engine.evaluate([flat()], WIDTH, HEIGHT, i * FRAME_DT, now)
    assert events == []


def test_a_crouch_held_past_the_dwell_fires_on_the_static_layer():
    """The known cost of this layer, pinned: a box shape cannot tell a crouch or someone
    sitting on the floor from a person lying there. Measured on the benchmarks. Without pose
    evidence nothing holds it back; with it, see test_a_crouch_with_an_upright_torso_is_held_back."""
    crouched = box(200, 150, 200)
    assert len(run(RuleEngine(), [(0.0, crouched)] + hold(crouched, 0.0, 3.4))) == 1


def test_getting_up_rearms_the_static_layer():
    frames = [(0.0, flat())] + hold(flat(), 0.0, 3.4)
    frames += [(3.4 + i * FRAME_DT, standing()) for i in range(1, 6)]
    frames += hold(flat(), 4.4, 3.4)
    assert len(run(RuleEngine(), frames)) == 2


# ---------------------------------------------------------------------------
# One incident, one alert: falls merged across tracks
# ---------------------------------------------------------------------------


def _lying_at(xmin: int, track_id: str) -> IncidentDetection:
    return IncidentDetection(
        label="person",
        track_id=track_id,
        confidence=0.9,
        bbox=BoundingBox(xmin=xmin, ymin=FLAT_TOP, xmax=xmin + FLAT_W, ymax=FLAT_TOP + FLAT_H),
    )


def _run_tracks(engine: RuleEngine, tracks: list[IncidentDetection], seconds: float) -> list[IncidentDetection]:
    now = datetime.now()
    events = []
    for i in range(int(round(seconds / FRAME_DT)) + 1):
        events += engine.evaluate(tracks, WIDTH, HEIGHT, i * FRAME_DT, now)
    return events


def test_one_person_split_across_two_tracks_raises_one_fall_alert():
    """GMDCSA24 s2-fall-04: a full-body box and a partial box of the same person on the
    floor are two tracks; each reached the dwell and alerted, 2.8 s apart."""
    engine = RuleEngine(alert_cooldown_sec=30.0)
    events = _run_tracks(engine, [_lying_at(200, "person_1"), _lying_at(230, "person_2")], 5.0)
    assert len(events) == 1


def test_two_people_down_apart_raise_two_alerts():
    engine = RuleEngine(alert_cooldown_sec=30.0)
    events = _run_tracks(engine, [_lying_at(10, "person_1"), _lying_at(450, "person_2")], 5.0)
    assert len(events) == 2


def test_a_merged_track_does_not_report_once_the_window_lapses():
    """The duplicate's own layer already marked it fired; the merge must not just delay it."""
    engine = RuleEngine(alert_cooldown_sec=5.0)
    events = _run_tracks(engine, [_lying_at(200, "person_1"), _lying_at(230, "person_2")], 12.0)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Pose gate: a fall due to fire is held back while the torso stays upright
# ---------------------------------------------------------------------------

UPRIGHT = PoseObservation(torso_deg=15.0, full_body=True)
LYING = PoseObservation(torso_deg=85.0, full_body=True)


def run_with_pose(
    engine: RuleEngine,
    frames: list[tuple[float, IncidentDetection]],
    pose_at,
) -> list[IncidentDetection]:
    """Like run(), but hands the engine `pose_at(timestamp)` for whichever tracks it asks about,
    the way VisionPipeline only runs pose for `pose_candidates`."""
    now = datetime.now()
    events: list[IncidentDetection] = []
    for timestamp, det in frames:
        evidence = {track_id: pose_at(timestamp) for track_id in engine.pose_candidates([det])}
        events.extend(engine.evaluate([det], WIDTH, HEIGHT, timestamp, now, fall_pose_evidence=evidence))
    return events


def _crouch(seconds: float = 3.4) -> list[tuple[float, IncidentDetection]]:
    crouched = box(200, 150, 200)
    return [(0.0, crouched)] + hold(crouched, 0.0, seconds)


def test_a_crouch_with_an_upright_torso_is_held_back():
    """The static layer's known cost (see test_a_crouch_held_past_the_dwell_fires_on_the_static_layer),
    which the pose gate exists for: CAUCAFall kneel, sit-down, pick-up-object."""
    engine = RuleEngine()
    assert run_with_pose(engine, _crouch(), lambda _: UPRIGHT) == []
    assert engine._tracks[(None, "person_1")].fall_suppressed_pose


def test_lying_flat_by_pose_still_fires_and_reports_the_torso_angle():
    events = run_with_pose(RuleEngine(), _crouch(), lambda _: LYING)
    assert len(events) == 1
    assert "torso median 85 deg" in events[0].reason


def test_the_gate_fails_open_when_the_whole_body_is_not_visible():
    """GMDCSA24 s1-fall-14 (ankles out of frame) read upright from the torso alone."""
    cut_off = PoseObservation(torso_deg=10.0, full_body=False)
    events = run_with_pose(RuleEngine(), _crouch(), lambda _: cut_off)
    assert len(events) == 1
    assert "whole body seen in 0/" in events[0].reason


def test_the_gate_fails_open_when_pose_finds_nobody():
    assert len(run_with_pose(RuleEngine(), _crouch(), lambda _: PoseObservation())) == 1


def test_a_held_back_transition_fall_fires_once_the_torso_goes_down():
    """Re-checked every frame: kneeling first and collapsing after is still a fall."""
    frames = collapse() + hold(flat(), 2 * FRAME_DT, 7.0)
    events = run_with_pose(RuleEngine(), frames, lambda t: UPRIGHT if t < 3.0 else LYING)
    assert len(events) == 1
    # Held back at the dwell (~3.4 s); fires once most of the last 3 s reads lying.
    assert events[0].timestamp > 4.0
    assert "upright->prone" in events[0].reason


def test_pose_is_only_asked_for_while_a_fall_is_being_timed():
    engine = RuleEngine()
    now = datetime.now()
    engine.evaluate([standing()], WIDTH, HEIGHT, 0.0, now)
    assert engine.pose_candidates([standing()]) == set()
    assert engine.pose_candidates([flat()]) == {"person_1"}

    run(engine, hold(flat(), 0.0, 3.4))
    assert engine.pose_candidates([flat()]) == set()  # fired: nothing left to decide
