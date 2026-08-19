"""Tests for the lightweight sparse optical-flow fall evidence."""

import cv2
import numpy as np

from src.vision.fall_motion import SparseFallMotionAnalyzer
from src.vision.schemas import BoundingBox, IncidentDetection


def _person(track_id: str = "person_1") -> IncidentDetection:
    return IncidentDetection(
        label="person",
        track_id=track_id,
        confidence=0.9,
        bbox=BoundingBox(xmin=20, ymin=20, xmax=100, ymax=120),
    )


def test_sparse_flow_measures_downward_motion_inside_candidate_box():
    first = np.zeros((160, 140, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    for x, y in ((35, 40), (55, 55), (75, 70), (90, 85), (45, 95), (80, 35)):
        cv2.circle(first, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(second, (x, y + 9), 3, (255, 255, 255), -1)

    analyzer = SparseFallMotionAnalyzer(analysis_width=140, min_points=3)
    person = _person()
    analyzer.measure(first, [person], set(), 0.0)
    evidence = analyzer.measure(second, [person], {"person_1"}, 0.2)["person_1"]

    assert evidence.tracked_points >= 3
    assert evidence.downward_point_fraction >= 0.6
    assert evidence.downward_flow_ratio > 0.2


def test_sparse_flow_does_not_pay_feature_tracking_for_non_candidates():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    analyzer = SparseFallMotionAnalyzer(analysis_width=100)
    person = _person()

    analyzer.measure(frame, [person], set(), 0.0)
    assert analyzer.measure(frame, [person], set(), 0.2) == {}
