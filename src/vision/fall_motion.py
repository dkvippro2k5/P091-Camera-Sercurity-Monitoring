"""Low-cost optical-flow evidence for ambiguous fall transitions."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.schemas import BoundingBox, IncidentDetection


@dataclass(frozen=True)
class FallMotionEvidence:
    """Sparse motion measured inside one tracked person's previous box."""

    downward_flow_ratio: float = 0.0
    downward_point_fraction: float = 0.0
    tracked_points: int = 0


class SparseFallMotionAnalyzer:
    """Track a few corners instead of running another neural-network model."""

    def __init__(
        self,
        analysis_width: int = 320,
        max_corners: int = 32,
        min_points: int = 4,
        min_downward_fraction: float = 0.6,
    ) -> None:
        self.analysis_width = analysis_width
        self.max_corners = max_corners
        self.min_points = min_points
        self.min_downward_fraction = min_downward_fraction
        self._previous_gray: np.ndarray | None = None
        self._previous_timestamp: float | None = None
        self._previous_boxes: dict[str, BoundingBox] = {}
        self._previous_frame_size: tuple[int, int] | None = None

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_timestamp = None
        self._previous_boxes.clear()
        self._previous_frame_size = None

    def measure(
        self,
        frame: np.ndarray,
        persons: list[IncidentDetection],
        candidate_track_ids: set[str],
        timestamp: float,
    ) -> dict[str, FallMotionEvidence]:
        """Measure sparse vertical motion, then retain this frame for the next call."""
        frame_height, frame_width = frame.shape[:2]
        gray = self._prepare_gray(frame)
        evidence: dict[str, FallMotionEvidence] = {}

        if (
            self._previous_gray is not None
            and self._previous_timestamp is not None
            and self._previous_frame_size == (frame_width, frame_height)
        ):
            dt = timestamp - self._previous_timestamp
            if dt > 0:
                current_by_id = {person.track_id: person for person in persons if person.track_id}
                for track_id in candidate_track_ids:
                    person = current_by_id.get(track_id)
                    previous_box = self._previous_boxes.get(track_id)
                    if person is None or previous_box is None:
                        continue
                    measured = self._measure_track(
                        self._previous_gray,
                        gray,
                        previous_box,
                        frame_width,
                        frame_height,
                        dt,
                    )
                    if measured is not None:
                        evidence[track_id] = measured

        self._previous_gray = gray
        self._previous_timestamp = timestamp
        self._previous_frame_size = (frame_width, frame_height)
        self._previous_boxes = {
            person.track_id: person.bbox
            for person in persons
            if person.track_id is not None
        }
        return evidence

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width > self.analysis_width:
            scaled_height = max(1, round(height * self.analysis_width / width))
            frame = cv2.resize(
                frame,
                (self.analysis_width, scaled_height),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _measure_track(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_box: BoundingBox,
        frame_width: int,
        frame_height: int,
        dt: float,
    ) -> FallMotionEvidence | None:
        scale_x = previous_gray.shape[1] / frame_width
        scale_y = previous_gray.shape[0] / frame_height
        x1 = max(0, min(previous_gray.shape[1] - 1, round(previous_box.xmin * scale_x)))
        y1 = max(0, min(previous_gray.shape[0] - 1, round(previous_box.ymin * scale_y)))
        x2 = max(x1 + 1, min(previous_gray.shape[1], round(previous_box.xmax * scale_x)))
        y2 = max(y1 + 1, min(previous_gray.shape[0], round(previous_box.ymax * scale_y)))
        box_height = y2 - y1
        if box_height < 8 or x2 - x1 < 6:
            return None

        mask = np.zeros_like(previous_gray)
        # Exclude the lowest 15% so a stable foot line does not dominate torso motion.
        mask[y1 : y1 + max(1, round(box_height * 0.85)), x1:x2] = 255
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=mask,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=5,
        )
        if points is None or len(points) < self.min_points:
            return None

        moved, status, _errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )
        if moved is None or status is None:
            return None

        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < self.min_points:
            return None
        vertical = moved.reshape(-1, 2)[valid, 1] - points.reshape(-1, 2)[valid, 1]
        downward_fraction = float(np.mean(vertical > 0.5))
        if downward_fraction < self.min_downward_fraction:
            return FallMotionEvidence(
                downward_point_fraction=downward_fraction,
                tracked_points=int(valid.sum()),
            )

        downward = vertical[vertical > 0]
        flow_ratio = float(np.median(downward) / max(1.0, box_height) / dt)
        return FallMotionEvidence(
            downward_flow_ratio=max(0.0, flow_ratio),
            downward_point_fraction=downward_fraction,
            tracked_points=int(valid.sum()),
        )
