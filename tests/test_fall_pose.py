"""FallPoseEstimator: torso angle and visibility from COCO keypoints, matched to tracked boxes.

The pose model is faked: these tests pin the geometry and the matching, not the network.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.vision.fall_pose import FallPoseEstimator, PoseObservation
from src.vision.schemas import BoundingBox, IncidentDetection


class _Arr:
    """Stands in for a torch tensor: `.cpu().numpy()`."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values

    def __len__(self):
        return len(self._values)


def _keypoints(head, shoulders, hips, ankles, conf=0.9):
    """17 COCO keypoints; parts passed as None are left invisible."""
    kp = np.zeros((17, 3), dtype=np.float32)
    for indices, point in (((0,), head), ((5, 6), shoulders), ((11, 12), hips), ((15, 16), ankles)):
        if point is not None:
            for i in indices:
                kp[i] = (*point, conf)
    return kp


STANDING = _keypoints(head=(100, 20), shoulders=(100, 50), hips=(100, 120), ankles=(100, 200))
LYING = _keypoints(head=(20, 100), shoulders=(50, 100), hips=(120, 105), ankles=(200, 110))


class _Boxes:
    def __init__(self, boxes):
        self.xyxy = _Arr(boxes)

    def __len__(self):
        return len(self.xyxy)


class _FakeModel:
    def __init__(self, boxes, keypoints):
        self.calls = 0
        self._result = SimpleNamespace(boxes=_Boxes(boxes), keypoints=SimpleNamespace(data=_Arr(keypoints)))

    def __call__(self, frame, verbose=False):
        self.calls += 1
        return [self._result]


def _estimator(boxes, keypoints) -> FallPoseEstimator:
    estimator = FallPoseEstimator(model_path=None)
    estimator.model = _FakeModel(boxes, keypoints)
    return estimator


def _person(track_id, xmin, ymin, xmax, ymax):
    return IncidentDetection(
        label="person", track_id=track_id, confidence=0.9, bbox=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    )


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def test_torso_angle_reads_standing_as_upright_and_lying_as_flat():
    estimator = FallPoseEstimator(model_path=None)
    assert estimator._observe(STANDING).torso_deg == pytest.approx(0.0)
    assert estimator._observe(LYING).torso_deg == pytest.approx(85.9, abs=0.1)
    assert estimator._observe(STANDING).full_body


def test_a_body_cut_off_at_the_ankles_is_not_full_body():
    cut = _keypoints(head=(100, 20), shoulders=(100, 50), hips=(100, 120), ankles=None)
    observation = FallPoseEstimator(model_path=None)._observe(cut)
    assert observation.torso_deg == pytest.approx(0.0)
    assert not observation.full_body


def test_each_candidate_gets_the_pose_that_overlaps_its_box():
    estimator = _estimator([[0, 0, 100, 200], [180, 100, 320, 160]], [STANDING, LYING])
    persons = [_person("standing", 0, 0, 100, 200), _person("lying", 185, 95, 315, 165)]

    observations = estimator.measure(FRAME, persons, {"standing", "lying"})

    assert observations["standing"].torso_deg == pytest.approx(0.0)
    assert observations["lying"].torso_deg > 80


def test_a_candidate_with_no_overlapping_pose_gets_an_empty_observation():
    estimator = _estimator([[0, 0, 100, 200]], [STANDING])
    observations = estimator.measure(FRAME, [_person("p", 200, 0, 300, 100)], {"p"})
    assert observations == {"p": PoseObservation()}


def test_no_candidates_means_no_pose_inference():
    estimator = _estimator([[0, 0, 100, 200]], [STANDING])
    assert estimator.measure(FRAME, [_person("p", 0, 0, 100, 200)], set()) == {}
    assert estimator.model.calls == 0


def test_missing_weights_disable_the_estimator_instead_of_failing():
    estimator = FallPoseEstimator(model_path="does/not/exist-pose.pt")
    assert not estimator.enabled
    assert estimator.measure(FRAME, [_person("p", 0, 0, 100, 200)], {"p"}) == {}


def test_each_camera_worker_gets_its_own_pose_model_and_the_gate_settings(monkeypatch):
    from src.vision.detector import IncidentDetector
    from src.vision.pipeline import VisionPipeline
    from src.vision.rule_engine import RuleEngine
    from src.vision.stream_manager import MultiStreamManager

    monkeypatch.setattr(FallPoseEstimator, "_load", lambda self: setattr(self, "model", object()))
    template = VisionPipeline(
        detector=IncidentDetector(model_path=None),
        rule_engine=RuleEngine(fall_pose_upright_deg=40.0),
        fall_pose_estimator=FallPoseEstimator("weights/pose.pt"),
    )
    factory = MultiStreamManager._make_default_factory(template)
    first, second = factory(), factory()

    assert first.fall_pose_estimator is not second.fall_pose_estimator
    assert first.fall_pose_estimator.model_path == "weights/pose.pt"
    assert first.rule_engine.fall_pose_upright_deg == 40.0

    template.fall_pose_estimator = None
    assert MultiStreamManager._make_default_factory(template)().fall_pose_estimator is None
