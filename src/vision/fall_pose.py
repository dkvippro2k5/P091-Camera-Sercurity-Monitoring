"""Pose evidence for fall candidates: is the person's torso upright or lying?

A box-shape rule cannot tell kneeling, sitting down or picking something up from lying on
the floor — all of them make the box wide and short. The torso can: shoulders above hips
means upright, whatever the box looks like. The estimator only runs on frames where the
rule engine is already timing a fall, so a scene with nobody falling never pays for it.
"""

import logging
import math
import os
from dataclasses import dataclass

import numpy as np

from src.vision.schemas import BoundingBox, IncidentDetection

logger = logging.getLogger(__name__)

# Not shipped in the repo (*.pt is gitignored); ultralytics publishes it as yolo26s-pose.pt.
DEFAULT_POSE_WEIGHTS_PATH = os.path.join("src", "vision", "weights", "yolo26s-pose.pt")

# COCO keypoint indices.
_HEAD = (0, 1, 2, 3, 4)
_SHOULDERS = (5, 6)
_HIPS = (11, 12)
_ANKLES = (15, 16)


@dataclass(frozen=True)
class PoseObservation:
    """One frame's pose reading for one tracked person.

    `torso_deg` is the shoulder-to-hip line's angle from vertical: 0 standing, 90 flat,
    180 head-down. `full_body` is whether a head and an ankle point were both visible; a
    person cut off by the frame edge or hidden behind furniture gives a torso angle that
    says little about the body as a whole.
    """

    torso_deg: float | None = None
    full_body: bool = False


class FallPoseEstimator:
    """Runs a COCO-keypoint YOLO pose model and matches each pose to a tracked box."""

    def __init__(
        self,
        model_path: str | None,
        keypoint_conf: float = 0.4,
        min_match_iou: float = 0.3,
    ) -> None:
        self.model_path = model_path
        self.keypoint_conf = keypoint_conf
        self.min_match_iou = min_match_iou
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
            # Optional layer: a deployment without the pose weights keeps every alert the
            # box rules raise rather than refusing to start.
            logger.warning(f"Pose weights not found at '{self.model_path}'; fall pose gate disabled.")
            return
        from ultralytics import YOLO  # type: ignore

        self.model_path = resolved
        self.model = YOLO(resolved)
        logger.info(f"Loaded fall pose model from {resolved}")

    def measure(
        self,
        frame: np.ndarray,
        persons: list[IncidentDetection],
        candidate_track_ids: set[str],
    ) -> dict[str, PoseObservation]:
        """One pose pass over the frame, matched to each candidate's box by IoU.

        A candidate with no matching pose still gets an (empty) observation, so the rule
        engine can tell "pose ran and found nothing" from "pose never ran".
        """
        if self.model is None or not candidate_track_ids:
            return {}
        boxes = {p.track_id: p.bbox for p in persons if p.track_id in candidate_track_ids}
        if not boxes:
            return {}

        result = self.model(frame, verbose=False)[0]
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            poses = list(zip(result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy(), strict=False))

        observations: dict[str, PoseObservation] = {}
        for track_id, box in boxes.items():
            best, best_iou = None, self.min_match_iou
            for pose_box, keypoints in poses:
                overlap = _iou(box, pose_box)
                if overlap > best_iou:
                    best, best_iou = keypoints, overlap
            observations[track_id] = self._observe(best) if best is not None else PoseObservation()
        return observations

    def _observe(self, keypoints: np.ndarray) -> PoseObservation:
        shoulders = self._mean(keypoints, _SHOULDERS)
        hips = self._mean(keypoints, _HIPS)
        torso_deg = None
        if shoulders is not None and hips is not None:
            dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
            torso_deg = math.degrees(math.atan2(abs(dx), dy))
        full_body = self._mean(keypoints, _HEAD) is not None and self._mean(keypoints, _ANKLES) is not None
        return PoseObservation(torso_deg=torso_deg, full_body=full_body)

    def _mean(self, keypoints: np.ndarray, indices: tuple[int, ...]) -> np.ndarray | None:
        points = [keypoints[i][:2] for i in indices if keypoints[i][2] >= self.keypoint_conf]
        return np.mean(points, axis=0) if points else None


def _iou(box: BoundingBox, other: np.ndarray) -> float:
    ix = max(0.0, min(box.xmax, other[2]) - max(box.xmin, other[0]))
    iy = max(0.0, min(box.ymax, other[3]) - max(box.ymin, other[1]))
    inter = ix * iy
    union = (box.xmax - box.xmin) * (box.ymax - box.ymin) + (other[2] - other[0]) * (other[3] - other[1]) - inter
    return float(inter / union) if union > 0 else 0.0

