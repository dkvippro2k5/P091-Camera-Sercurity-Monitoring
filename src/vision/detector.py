import logging
import os
import time

import numpy as np

from src.vision.schemas import BoundingBox, IncidentDetection

logger = logging.getLogger(__name__)


class IncidentDetector:
    """Object detector feeding the rule engine.

    This layer answers "what objects are in this frame", not "is this an incident".
    It emits raw classes (person, backpack, ...) with stable track IDs; deciding whether
    they add up to intrusion / fall / abandoned_object is the RuleEngine's job, because
    those labels need history that a single frame cannot supply.

    Falls back to a scripted mock scenario when weights are missing or ultralytics is
    not installed, so the rule layer stays testable without a trained model.
    """

    def __init__(
        self,
        model_path: str | None = "data/vision/weights/yolo11n.pt",
        conf_threshold: float = 0.5,
        target_labels: list[str] | None = None,
        use_tracking: bool = True,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # None keeps every class the model emits; a list filters to those classes.
        self.target_labels = target_labels
        self.use_tracking = use_tracking
        self.model = None
        self.is_mock = True

        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.model_path and os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO  # type: ignore

                self.model = YOLO(self.model_path)
                self.is_mock = False
                logger.info(f"Successfully loaded YOLO model from {self.model_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load YOLO model from {self.model_path}: {e}. Falling back to mock detection mode."
                )
                self.is_mock = True
        else:
            logger.info(
                f"Model weights not found at '{self.model_path}'. Running in Mock Detection mode."
            )
            self.is_mock = True

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        camera_id: str | None = None,
        timestamp: float | None = None,
    ) -> list[IncidentDetection]:
        """Detect tracked objects in a single BGR frame.

        Args:
            frame: Image array (H, W, C) in BGR format.
            frame_id: Frame index number.
            camera_id: Identifier of the source camera, stamped onto each detection.
            timestamp: Seconds into the stream. Only the mock uses it, to place itself
                in its scripted timeline; None derives it from frame_id at 30fps.

        Returns:
            List of IncidentDetection carrying raw object classes and track IDs.
        """
        if frame is None or frame.size == 0:
            return []

        if not self.is_mock and self.model is not None:
            return self._detect_real(frame, camera_id=camera_id)
        return self._detect_mock(frame, frame_id=frame_id, camera_id=camera_id, timestamp=timestamp)

    def _detect_real(self, frame: np.ndarray, camera_id: str | None = None) -> list[IncidentDetection]:
        """Run YOLO inference, preferring tracking so track IDs stay stable across frames."""
        if self.use_tracking:
            # persist=True carries the tracker's state between calls, which is what makes
            # box.id stable. Without it every frame restarts numbering and the temporal
            # rules can never accumulate history.
            results = self.model.track(
                frame, conf=self.conf_threshold, persist=True, verbose=False
            )
        else:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections: list[IncidentDetection] = []
        current_time = round(time.time(), 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                label_name = (
                    r.names[cls_id]
                    if hasattr(r, "names") and cls_id in r.names
                    else f"class_{cls_id}"
                )
                if self.target_labels and label_name not in self.target_labels:
                    continue

                # Left as None when the tracker gave us nothing. A synthesized ID would
                # collide across objects and silently corrupt every temporal rule.
                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = f"track_{int(box.id[0].cpu().numpy())}"

                detections.append(
                    IncidentDetection(
                        label=label_name,
                        track_id=track_id,
                        confidence=round(conf, 4),
                        timestamp=current_time,
                        camera_id=camera_id,
                        bbox=BoundingBox(
                            xmin=int(xyxy[0]),
                            ymin=int(xyxy[1]),
                            xmax=int(xyxy[2]),
                            ymax=int(xyxy[3]),
                        ),
                    )
                )

        return detections

    def _detect_mock(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        camera_id: str | None = None,
        timestamp: float | None = None,
    ) -> list[IncidentDetection]:
        """Replay a scripted scenario so every rule can be exercised without weights.

        Timeline (seconds), designed so all three rules fire inside ~25s:
          - person_1 walks left to right, crossing the restricted zone around t=6s
            -> intrusion
          - person_2 stands until t=5s, then collapses and stays down
            -> fall_detection at t=8s (after the 3s dwell)
          - backpack_1 sits still all along; its owner person_3 walks off at t=6s
            -> abandoned_object at ~t=16s (after the 10s unattended window)
        """
        height, width = frame.shape[:2]
        t = timestamp if timestamp is not None else frame_id / 30.0
        current_time = round(time.time(), 2)

        def _box(cx: float, cy: float, bw: float, bh: float) -> BoundingBox:
            """Build a pixel bbox from a normalized center and size."""
            return BoundingBox(
                xmin=max(0, int((cx - bw / 2) * width)),
                ymin=max(0, int((cy - bh / 2) * height)),
                xmax=min(width, int((cx + bw / 2) * width)),
                ymax=min(height, int((cy + bh / 2) * height)),
            )

        def _add(label: str, track_id: str, conf: float, bbox: BoundingBox) -> None:
            detections.append(
                IncidentDetection(
                    label=label,
                    track_id=track_id,
                    confidence=conf,
                    timestamp=current_time,
                    camera_id=camera_id,
                    bbox=bbox,
                )
            )

        detections: list[IncidentDetection] = []

        # person_1: walks rightwards along the corridor, entering the zone near t=6s.
        walk_x = min(0.95, 0.05 + 0.05 * t)
        _add("person", "person_1", 0.94, _box(walk_x, 0.62, 0.08, 0.35))

        # person_2: upright, then collapsed (bbox flips from tall to wide) from t=5s.
        if t < 5.0:
            _add("person", "person_2", 0.91, _box(0.75, 0.62, 0.08, 0.35))
        else:
            _add("person", "person_2", 0.88, _box(0.75, 0.76, 0.30, 0.09))

        # backpack_1: never moves. person_3 stands beside it, then leaves at t=6s.
        _add("backpack", "backpack_1", 0.86, _box(0.18, 0.30, 0.06, 0.08))
        if t < 6.0:
            _add("person", "person_3", 0.90, _box(0.22, 0.32, 0.08, 0.35))

        return detections
