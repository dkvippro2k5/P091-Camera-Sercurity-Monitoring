"""Second opinion on a static region reported as a left object: what does it look like?

The background model (static_objects.py) finds anything newly static, which is what lets it
catch bags YOLO never boxes - and also what makes it report a train at the platform, a door
left open or someone YOLO lost. An open-vocabulary detector (YOLOE) asked about bags and the
usual impostors tells them apart on the clear cases: on ABODA + AVSS 2007 + CAVIAR
(2026-10-20) it saw no bag in any of the background model's false alarms and named most of
them (train 0.25-0.54, door 0.38, person 0.54), while clearly visible bags scored 0.6-0.8.

It is a note on the alert, not a gate. The bags the background model exists for - small, far,
infrared - score as low as the false alarms (0.00 on four of the eight it found), so gating on
it would throw away the detections the background model was added to make. The note goes into
the alert's reason, which the operator sees and the VLM triage reads as evidence.

The vocabulary is baked into the weights by scripts/build_left_object_vocab.py, so running it
needs only ultralytics, not the CLIP text encoder YOLOE uses to read prompts.
"""

import logging
import os

import numpy as np

from src.vision.schemas import BoundingBox

logger = logging.getLogger(__name__)

# Not shipped in the repo (*.pt is gitignored); built by scripts/build_left_object_vocab.py.
DEFAULT_LEFT_OBJECT_WEIGHTS_PATH = os.path.join("src", "vision", "weights", "yoloe-26s-left-objects.pt")

LEFT_OBJECT_WORDS = ("bag", "backpack", "handbag", "suitcase", "box", "package", "luggage")
# What else a newly static region turns out to be, named so the detector has somewhere to put it.
IMPOSTOR_WORDS = ("person", "train", "door", "screen")
VOCABULARY = LEFT_OBJECT_WORDS + IMPOSTOR_WORDS


class LeftObjectCheck:
    """Names what an open-vocabulary detector sees in a region.

    Args:
        model_path: Weights with `VOCABULARY` baked in. Missing weights disable the check.
        image_size: Inference size for the crop; the crop is the region plus its own size
            (at least 32 px) of context on every side.
        min_conf: Detections below this are ignored.
        min_cover: A detection must cover this fraction of the region to describe it.
        sure_conf: Below this the note gives the word as a weak guess and draws no
            conclusion. On ABODA two real bags (night lobby, infrared) came back as
            "person" at 0.06 and 0.22; told "not a left object", the VLM and the guard
            would have been steered away from them.
    """

    def __init__(
        self,
        model_path: str | None,
        image_size: int = 320,
        min_conf: float = 0.05,
        min_cover: float = 0.3,
        sure_conf: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.min_conf = min_conf
        self.min_cover = min_cover
        self.sure_conf = sure_conf
        self.model = None
        self._load()

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if not self.model_path:
            return
        from src.vision.detector import IncidentDetector

        resolved = IncidentDetector._resolve_weights(self.model_path)
        if resolved is None:
            logger.warning(f"Left-object weights not found at '{self.model_path}'; alerts go out without the check.")
            return
        from ultralytics import YOLO  # type: ignore

        self.model_path = resolved
        self.model = YOLO(resolved)
        logger.info(f"Loaded left-object check from {resolved}")

    def describe(self, frame: np.ndarray, box: BoundingBox) -> tuple[str, float] | None:
        """The best-scoring word whose detection covers the region, or None if nothing does."""
        if self.model is None:
            return None
        height, width = frame.shape[:2]
        pad = max(box.xmax - box.xmin, box.ymax - box.ymin, 32)
        x0, y0 = max(0, box.xmin - pad), max(0, box.ymin - pad)
        x1, y1 = min(width, box.xmax + pad), min(height, box.ymax + pad)
        result = self.model.predict(
            frame[y0:y1, x0:x1], imgsz=self.image_size, conf=self.min_conf, verbose=False
        )[0]
        area = max(1, (box.xmax - box.xmin) * (box.ymax - box.ymin))
        best = None
        for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            w = min(xyxy[2] + x0, box.xmax) - max(xyxy[0] + x0, box.xmin)
            h = min(xyxy[3] + y0, box.ymax) - max(xyxy[1] + y0, box.ymin)
            if w <= 0 or h <= 0 or w * h < self.min_cover * area:
                continue
            if best is None or conf > best[1]:
                best = (result.names[int(cls)], float(conf))
        return best

    def note(self, frame: np.ndarray, box: BoundingBox) -> str | None:
        """The check's finding as a clause for an alert reason, or None when it is disabled."""
        if self.model is None:
            return None
        found = self.describe(frame, box)
        if found is None:
            return "open-vocabulary check recognised nothing there"
        word, conf = found
        if conf < self.sure_conf:
            return f"open-vocabulary check recognised nothing clearly (weak guess: {word} {conf:.2f})"
        kind = "a left object" if word in LEFT_OBJECT_WORDS else "not a left object"
        return f"open-vocabulary check: looks like {word} ({conf:.2f}), {kind}"
