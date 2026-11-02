"""Tests for the background model that finds objects the detector does not box."""

import numpy as np

from src.vision import BoundingBox, IncidentDetection
from src.vision.static_objects import LABEL_STATIC, StaticObjectDetector

WIDTH, HEIGHT = 384, 288


def floor() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)


def with_bag(frame: np.ndarray, x: int = 180, y: int = 140, size: int = 24) -> np.ndarray:
    frame = frame.copy()
    frame[y : y + size, x : x + size] = 40
    return frame


def person(xmin: int, ymin: int, xmax: int, ymax: int) -> IncidentDetection:
    return IncidentDetection(
        label="person",
        track_id="person_1",
        confidence=0.9,
        bbox=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
    )


def run(model: StaticObjectDetector, frames, persons=()):
    """Feed (timestamp, frame) pairs at 5 fps; return the regions of the last frame."""
    regions = []
    for t, frame in frames:
        regions = model.measure(frame, t, list(persons), "cam")
    return regions


def test_an_object_put_down_becomes_a_region_after_static_sec_with_the_time_it_appeared():
    model = StaticObjectDetector(static_sec=8.0)
    run(model, [(t / 5, floor()) for t in range(10)])  # empty scene, 0-1.8 s
    assert run(model, [(2.0 + t / 5, with_bag(floor())) for t in range(39)]) == []  # until 9.6 s

    regions = run(model, [(10.0, with_bag(floor()))])
    assert len(regions) == 1
    region = regions[0]
    assert region.detection.label == LABEL_STATIC
    assert region.since == 2.0
    assert 170 <= region.detection.bbox.xmin <= 182 and 202 <= region.detection.bbox.xmax <= 214

    # The same region keeps its ID on the next frame.
    again = run(model, [(10.2, with_bag(floor()))])
    assert [r.detection.track_id for r in again] == [region.detection.track_id]


def test_a_person_standing_still_is_not_a_region():
    model = StaticObjectDetector(static_sec=8.0)
    run(model, [(0.0, floor())])
    standing = person(170, 100, 210, 170)
    regions = run(model, [(0.2 + t / 5, with_bag(floor())) for t in range(60)], persons=[standing])
    assert regions == []


def test_a_ghost_of_something_that_was_in_the_first_frame_is_not_a_region():
    """The background starts with the bag in it; once it is gone the floor there is no object."""
    model = StaticObjectDetector(static_sec=8.0)
    run(model, [(0.0, with_bag(floor()))])
    assert run(model, [(0.2 + t / 5, floor()) for t in range(60)]) == []


def test_a_lighting_change_relearns_the_background_so_a_bag_left_afterwards_is_still_found():
    model = StaticObjectDetector(static_sec=8.0)
    run(model, [(0.0, floor())])
    brighter = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
    assert run(model, [(0.2 + t / 5, brighter) for t in range(9)]) == []
    regions = run(model, [(2.0 + t / 5, with_bag(brighter)) for t in range(41)])
    assert len(regions) == 1 and regions[0].since == 2.0


def test_something_moving_through_the_scene_is_not_a_region():
    model = StaticObjectDetector(static_sec=8.0)
    run(model, [(0.0, floor())])
    passing = [(0.2 + t / 5, with_bag(floor(), x=10 + 6 * t)) for t in range(55)]
    assert run(model, passing) == []


def test_each_camera_worker_gets_its_own_background_with_the_template_s_settings():
    from src.vision.detector import IncidentDetector
    from src.vision.pipeline import VisionPipeline
    from src.vision.stream_manager import MultiStreamManager

    template = VisionPipeline(
        detector=IncidentDetector(model_path=None),
        static_object_detector=StaticObjectDetector(static_sec=12.0),
    )
    template.static_object_detector.measure(floor(), 0.0, [], "cam")  # the template has seen a frame
    factory = MultiStreamManager._make_default_factory(template)
    first, second = factory(), factory()

    assert first.static_object_detector is not second.static_object_detector
    assert first.static_object_detector.static_sec == 12.0
    assert first.static_object_detector._background is None

    template.static_object_detector = None
    assert MultiStreamManager._make_default_factory(template)().static_object_detector is None
