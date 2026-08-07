"""Tests for the pose-confirmed fall cascade.

Every test here runs on synthetic keypoints, never a real pose model: CI has no
checkpoint and no network, and the point is to pin the *decision logic*, which is pure
geometry and needs no inference.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.fall_classifier import classify_pose
from src.vision.pose import (
    KP_LEFT_ANKLE,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_ANKLE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
)
from src.vision.rule_engine import LABEL_FALL, RuleEngine
from src.vision.schemas import BoundingBox, IncidentDetection

FRAME_W, FRAME_H = 640, 480


def _keypoints(
    shoulder_y: float,
    hip_y: float,
    shoulder_x: float = 300.0,
    hip_x: float = 300.0,
    nose_y: float = 100.0,
    conf: float = 0.9,
) -> list[tuple[float, float, float]]:
    """Build a 17-point COCO array, filling only the joints the classifier reads."""
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[KP_NOSE] = (shoulder_x, nose_y, conf)
    kps[KP_LEFT_SHOULDER] = (shoulder_x - 20, shoulder_y, conf)
    kps[KP_RIGHT_SHOULDER] = (shoulder_x + 20, shoulder_y, conf)
    kps[KP_LEFT_HIP] = (hip_x - 15, hip_y, conf)
    kps[KP_RIGHT_HIP] = (hip_x + 15, hip_y, conf)
    kps[KP_LEFT_ANKLE] = (hip_x - 15, hip_y + 100, conf)
    kps[KP_RIGHT_ANKLE] = (hip_x + 15, hip_y + 100, conf)
    return kps


def _person(bbox: BoundingBox, track_id: str = "track_1", keypoints=None) -> IncidentDetection:
    return IncidentDetection(
        label="person",
        track_id=track_id,
        confidence=0.9,
        timestamp=0.0,
        bbox=bbox,
        keypoints=keypoints,
    )


# --- classifier geometry ----------------------------------------------------------


def test_upright_torso_is_not_prone():
    # Shoulders above hips, same x: a perfectly vertical torso.
    verdict = classify_pose(_keypoints(shoulder_y=150, hip_y=300))
    assert verdict is not None
    assert verdict.is_prone is False
    assert verdict.torso_angle_deg < 10


def test_horizontal_torso_is_prone():
    # Shoulders and hips at the same height, far apart in x: lying down.
    verdict = classify_pose(_keypoints(shoulder_y=400, hip_y=405, shoulder_x=200, hip_x=400, nose_y=400))
    assert verdict is not None
    assert verdict.is_prone is True
    assert verdict.torso_angle_deg > 80


def test_crouch_is_rejected_where_aspect_ratio_would_fire():
    """The whole reason pose was added.

    A crouching person has a wide, short bbox — aspect ratio 1.5, well past the 1.2
    heuristic — but the torso is still vertical, so this must not be a fall.
    """
    crouch_bbox = BoundingBox(xmin=200, ymin=300, xmax=380, ymax=420)
    width = crouch_bbox.xmax - crouch_bbox.xmin
    height = crouch_bbox.ymax - crouch_bbox.ymin
    assert width / height >= 1.2, "fixture must be one the old heuristic would fire on"

    verdict = classify_pose(_keypoints(shoulder_y=310, hip_y=400, nose_y=305))
    assert verdict is not None
    assert verdict.is_prone is False


def test_head_below_hips_is_prone_even_when_torso_looks_short():
    """Collapse toward the camera foreshortens the torso, so the angle stays small."""
    verdict = classify_pose(_keypoints(shoulder_y=300, hip_y=320, shoulder_x=300, hip_x=305, nose_y=350))
    assert verdict is not None
    assert verdict.is_prone is True
    assert "hip level" in verdict.reason


def test_low_confidence_keypoints_are_undecided():
    verdict = classify_pose(_keypoints(shoulder_y=400, hip_y=405, hip_x=400, conf=0.05))
    assert verdict is None, "low-confidence pose must be undecided, not 'upright'"


def test_missing_and_short_keypoints_are_undecided():
    assert classify_pose(None) is None
    assert classify_pose([]) is None
    assert classify_pose([(0.0, 0.0, 0.9)] * 5) is None


# --- integration with the dwell timer ---------------------------------------------


def _run(engine: RuleEngine, person_at, times) -> list:
    """Feed the engine one person per timestamp, returning every event emitted."""
    fired = []
    for t in times:
        fired.extend(
            engine.evaluate(
                detections=[person_at(t)],
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                timestamp=t,
                now=datetime(2026, 1, 1, 3, 0, 0),
            )
        )
    return fired


def test_pose_fall_still_requires_the_prd_three_second_dwell():
    """Pose changes what counts as collapsed; it must never shorten the 3s dwell."""
    engine = RuleEngine(fall_dwell_sec=3.0)
    prone = _keypoints(shoulder_y=400, hip_y=405, shoulder_x=200, hip_x=400, nose_y=400)
    flat = BoundingBox(xmin=200, ymin=380, xmax=420, ymax=430)

    def person_at(_t):
        return _person(flat, keypoints=prone)

    # Under the threshold: nothing yet.
    assert _run(engine, person_at, [0.0, 1.0, 2.0, 2.9]) == []

    # Crossing it: exactly one event.
    fired = _run(engine, person_at, [3.0])
    assert len(fired) == 1
    assert fired[0].label == LABEL_FALL
    assert "torso" in fired[0].reason


def test_crouching_never_fires_however_long_it_lasts():
    engine = RuleEngine(fall_dwell_sec=3.0)
    crouch = _keypoints(shoulder_y=310, hip_y=400, nose_y=305)
    wide = BoundingBox(xmin=200, ymin=300, xmax=380, ymax=420)

    def person_at(_t):
        return _person(wide, keypoints=crouch)

    assert _run(engine, person_at, [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]) == []


def test_falls_back_to_aspect_ratio_when_pose_is_absent():
    """With no keypoints the engine must reproduce its original behaviour exactly."""
    engine = RuleEngine(fall_dwell_sec=3.0, fall_aspect_ratio=1.2)
    flat = BoundingBox(xmin=200, ymin=380, xmax=420, ymax=430)

    def person_at(_t):
        return _person(flat, keypoints=None)

    assert _run(engine, person_at, [0.0, 1.0, 2.0]) == []
    fired = _run(engine, person_at, [3.0])
    assert len(fired) == 1
    assert "no pose available" in fired[0].reason


def test_fall_candidate_gate_is_looser_than_the_fall_threshold():
    """The cascade's pre-filter must admit cases the old heuristic rejected."""
    engine = RuleEngine(fall_aspect_ratio=1.2, fall_pose_gate_ratio=0.8)

    # Ratio 1.0: below the old fall threshold, above the candidate gate.
    square = BoundingBox(xmin=200, ymin=300, xmax=300, ymax=400)
    assert engine.is_fall_candidate(_person(square)) is True

    # A clearly upright person is not worth pose inference.
    tall = BoundingBox(xmin=280, ymin=200, xmax=330, ymax=450)
    assert engine.is_fall_candidate(_person(tall)) is False


# --- alert cooldown ---------------------------------------------------------------


def test_alert_cooldown_suppresses_repeat_labels_on_one_camera():
    engine = RuleEngine(fall_dwell_sec=3.0, alert_cooldown_sec=30.0, track_ttl_sec=1000.0)
    prone = _keypoints(shoulder_y=400, hip_y=405, shoulder_x=200, hip_x=400, nose_y=400)
    flat = BoundingBox(xmin=200, ymin=380, xmax=420, ymax=430)

    def two_people(t):
        return [
            _person(flat, track_id="track_1", keypoints=prone),
            _person(flat, track_id="track_2", keypoints=prone),
        ]

    fired = []
    for t in [0.0, 3.0]:
        fired.extend(
            engine.evaluate(
                detections=two_people(t),
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                timestamp=t,
                now=datetime(2026, 1, 1, 3, 0, 0),
            )
        )

    # Both tracks fall simultaneously, but the operator gets one alert, not two.
    assert len(fired) == 1


def test_zero_cooldown_disables_suppression():
    engine = RuleEngine(fall_dwell_sec=3.0, alert_cooldown_sec=0.0, track_ttl_sec=1000.0)
    prone = _keypoints(shoulder_y=400, hip_y=405, shoulder_x=200, hip_x=400, nose_y=400)
    flat = BoundingBox(xmin=200, ymin=380, xmax=420, ymax=430)

    fired = []
    for t in [0.0, 3.0]:
        fired.extend(
            engine.evaluate(
                detections=[
                    _person(flat, track_id="track_1", keypoints=prone),
                    _person(flat, track_id="track_2", keypoints=prone),
                ],
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                timestamp=t,
                now=datetime(2026, 1, 1, 3, 0, 0),
            )
        )

    assert len(fired) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
