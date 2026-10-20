"""LeftObjectCheck: what an open-vocabulary detector sees in a static region, as a note on the alert.

The detector is faked: these tests pin the crop geometry, the choice of word and where the
note goes, not the network.
"""

from types import SimpleNamespace

import numpy as np

from src.vision.left_object_check import VOCABULARY, LeftObjectCheck
from src.vision.schemas import BoundingBox, IncidentDetection

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
REGION = BoundingBox(xmin=300, ymin=200, xmax=340, ymax=240)  # cropped from (260, 160)
NAMES = dict(enumerate(VOCABULARY))


class _List:
    """Stands in for a torch tensor: `.tolist()`."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeModel:
    def __init__(self, detections):
        """detections: (xyxy in crop pixels, word, confidence)."""
        self.crops = []
        boxes = SimpleNamespace(
            xyxy=_List([list(box) for box, _, _ in detections]),
            cls=_List([float(VOCABULARY.index(word)) for _, word, _ in detections]),
            conf=_List([conf for _, _, conf in detections]),
        )
        self._result = SimpleNamespace(boxes=boxes, names=NAMES)

    def predict(self, crop, imgsz, conf, verbose):
        self.crops.append(crop.shape)
        return [self._result]


def make_check(detections) -> LeftObjectCheck:
    check = LeftObjectCheck(model_path=None)
    check.model = _FakeModel(detections)
    return check


def test_the_best_detection_covering_the_region_names_it():
    check = make_check(
        [
            ((40, 40, 80, 80), "suitcase", 0.4),  # on the region
            ((42, 42, 78, 78), "train", 0.6),  # on it too, and more confident
            ((0, 0, 20, 20), "person", 0.9),  # beside it: says nothing about the region
        ]
    )
    assert check.describe(FRAME, REGION) == ("train", 0.6)
    # The region plus 40 px (its size) of context each side.
    assert check.model.crops == [(120, 120, 3)]


def test_the_note_says_whether_the_word_is_a_left_object():
    assert "looks like suitcase (0.40), a left object" in make_check([((40, 40, 80, 80), "suitcase", 0.4)]).note(
        FRAME, REGION
    )
    assert "not a left object" in make_check([((40, 40, 80, 80), "train", 0.5)]).note(FRAME, REGION)
    assert make_check([]).note(FRAME, REGION) == "open-vocabulary check recognised nothing there"


def test_a_weak_guess_draws_no_conclusion():
    """Infrared bags on ABODA came back as "person" at 0.06 and 0.22."""
    note = make_check([((40, 40, 80, 80), "person", 0.06)]).note(FRAME, REGION)
    assert note == "open-vocabulary check recognised nothing clearly (weak guess: person 0.06)"
    assert "not a left object" not in note


def test_missing_weights_disable_the_check():
    check = LeftObjectCheck(model_path="does/not/exist.pt")
    assert not check.enabled
    assert check.note(FRAME, REGION) is None


def test_the_pipeline_notes_only_the_background_model_s_abandoned_alerts():
    from src.vision.pipeline import VisionPipeline

    pipeline = VisionPipeline(detector=SimpleNamespace(), zones_path=None, left_object_check=make_check([]))

    def alert(reason):
        return IncidentDetection(label="abandoned_object", track_id="t", confidence=1.0, bbox=REGION, reason=reason)

    static, detected = alert("static_object (static_3) unattended for 2.0s"), alert("suitcase (track_9) unattended")
    pipeline._note_left_objects(FRAME, [static, detected])
    assert static.reason.endswith("; open-vocabulary check recognised nothing there")
    assert detected.reason == "suitcase (track_9) unattended"
