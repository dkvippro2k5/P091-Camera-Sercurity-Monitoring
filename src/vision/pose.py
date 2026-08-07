"""Optional pose estimation used to confirm fall candidates.

This exists because bounding-box shape alone cannot tell lying down from crouching to
tie a shoe, sitting on the floor, or standing close to a wide-angle lens — see
`rule_engine._check_falls`. Keypoints resolve that ambiguity.

Deliberately a *confirmation* stage, not a replacement for detection: the caller gates
which crops reach `estimate()`, so in the common case (everybody upright) this model
never runs and the added cost is zero.
"""

import logging
import os

import numpy as np

from src.vision.schemas import BoundingBox

logger = logging.getLogger(__name__)

# COCO 17-keypoint layout emitted by ultralytics pose models. Named here because the
# fall classifier indexes into it and bare integers would be unreadable.
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

DEFAULT_POSE_WEIGHTS = os.path.join("src", "vision", "weights", "yolo11n-pose.pt")

# Below this the crop is a few pixels of noise; the pose head will emit confident-looking
# garbage rather than nothing, so reject before inference instead of trusting the scores.
_MIN_CROP_PIXELS = 24

# One VisionPipeline is built per camera worker, so an unconditional warning here would
# print once per camera on every start. The condition is process-wide, so say it once.
_missing_weights_warned = False


class PoseEstimator:
    """Wraps an ultralytics pose model, degrading to unavailable instead of raising.

    `available` is False when the weights are missing and cannot be fetched. Callers must
    check it and fall back — CI has neither the checkpoint nor network access, and a
    surveillance pipeline must not stop detecting falls because a pose model is absent.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_POSE_WEIGHTS,
        conf_threshold: float = 0.5,
        auto_download: bool = False,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.available = False

        self._initialize_model(auto_download)

    def _initialize_model(self, auto_download: bool) -> None:
        """Load the pose model, logging once and staying unavailable on any failure."""
        global _missing_weights_warned
        from src.vision.detector import IncidentDetector

        resolved = IncidentDetector._resolve_weights(self.model_path)
        # ultralytics silently downloads a bare name like "yolo11n-pose.pt" on construction.
        # That is a network call in the middle of frame processing, so it is opt-in.
        if resolved is None and not auto_download:
            if not _missing_weights_warned:
                _missing_weights_warned = True
                logger.warning(
                    f"Pose weights not found at '{self.model_path}'. Fall detection will fall back "
                    "to the bounding-box heuristic. Pass auto_download=True or place the checkpoint "
                    "in src/vision/weights/ to enable pose confirmation."
                )
            return

        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(resolved or self.model_path)
            self.available = True
            logger.info(f"Loaded pose model from {resolved or self.model_path}")
        except Exception as exc:
            # Broad by design: a missing checkpoint, a failed download and a corrupt file
            # are all "no pose available", and none of them should kill the pipeline.
            logger.warning(
                f"Could not load pose model '{self.model_path}': {exc}. "
                "Falling back to the bounding-box fall heuristic."
            )
            self.model = None
            self.available = False

    def estimate(self, frame: np.ndarray, bbox: BoundingBox) -> list[tuple[float, float, float]] | None:
        """Return 17 COCO keypoints as (x, y, confidence) in *frame* pixel coordinates.

        Runs on the person crop rather than the full frame: the crop is a fraction of the
        pixels, and it stops a second person in view from being matched to this track.

        Returns None when pose is unavailable, the crop is degenerate, or the model found
        no person — all of which mean "undecided", not "not fallen".
        """
        if not self.available or self.model is None or frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        # Clamp: detections may sit partly outside the frame after inference-size rescaling.
        x1 = max(0, min(int(bbox.xmin), width - 1))
        y1 = max(0, min(int(bbox.ymin), height - 1))
        x2 = max(x1 + 1, min(int(bbox.xmax), width))
        y2 = max(y1 + 1, min(int(bbox.ymax), height))

        if (x2 - x1) < _MIN_CROP_PIXELS or (y2 - y1) < _MIN_CROP_PIXELS:
            return None

        crop = frame[y1:y2, x1:x2]
        try:
            results = self.model(crop, conf=self.conf_threshold, verbose=False)
        except Exception as exc:
            logger.debug(f"Pose inference failed on crop: {exc}")
            return None

        for r in results:
            kps = getattr(r, "keypoints", None)
            if kps is None or kps.data is None or len(kps.data) == 0:
                continue

            # data is (num_people, 17, 3). The crop is one tracked person, so take the first.
            arr = kps.data[0].cpu().numpy()
            if arr.shape[0] < KP_RIGHT_ANKLE + 1:
                continue

            # Offset back into frame coordinates so the classifier and any overlay share
            # the same space as the bounding boxes.
            return [(float(x) + x1, float(y) + y1, float(c)) for x, y, c in arr]

        return None


def pose_weights_available(model_path: str = DEFAULT_POSE_WEIGHTS) -> bool:
    """Cheap check used by tests and startup logging to avoid constructing a model."""
    from src.vision.detector import IncidentDetector

    return IncidentDetector._resolve_weights(model_path) is not None or os.path.exists(model_path)
