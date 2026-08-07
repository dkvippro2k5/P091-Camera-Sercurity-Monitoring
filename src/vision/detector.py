import logging
import os
import time

import numpy as np

from src.vision.schemas import BoundingBox, IncidentDetection

logger = logging.getLogger(__name__)

# src/vision/detector.py -> project root, so repo-relative defaults resolve regardless
# of the directory the process was launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Repo-relative on purpose. An absolute path here would short-circuit _resolve_weights
# (it returns early for absolute paths) and break on every machine but the author's,
# including CI and Docker. The default is the medium YOLO11 — the small checkpoint the
# capacity table in stream_manager.py was measured against still works if passed
# explicitly, but the medium model is the one the team runs with.
DEFAULT_WEIGHTS_PATH = os.path.join("src", "vision", "weights", "yolo11m.pt")


class IncidentDetector:
    """Object detector feeding the rule engine.

    This layer answers "what objects are in this frame", not "is this an incident".
    It emits raw classes (person, backpack, ...) with stable track IDs; deciding whether
    they add up to intrusion / fall / abandoned_object is the RuleEngine's job, because
    those labels need history that a single frame cannot supply.

    A model_path of None means no model is loaded and detect() always returns [] —
    useful for tests that only exercise the rule layer.
    """

    def __init__(
        self,
        model_path: str | None = DEFAULT_WEIGHTS_PATH,
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

        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.model_path is None:
            return

        resolved = self._resolve_weights(self.model_path)
        if resolved is None:
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}' "
                f"(searched relative to CWD '{os.getcwd()}' and the project root)."
            )
        self.model_path = resolved
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(resolved)
        logger.info(f"Successfully loaded YOLO model from {resolved}")

    @staticmethod
    def _resolve_weights(model_path: str | None) -> str | None:
        """Find the weights file, tolerating a CWD that is not the project root."""
        if not model_path:
            return None
        if os.path.isabs(model_path):
            return model_path if os.path.exists(model_path) else None

        candidates = [model_path, os.path.join(_PROJECT_ROOT, model_path)]
        return next((c for c in candidates if os.path.exists(c)), None)

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,      # kept for API compatibility — unused, was mock-only
        camera_id: str | None = None,
        timestamp: float | None = None,  # kept for API compatibility — unused, was mock-only
    ) -> list[IncidentDetection]:
        """Detect tracked objects in a single BGR frame.

        Args:
            frame: Image array (H, W, C) in BGR format.
            frame_id: Frame index number.
            camera_id: Identifier of the source camera, stamped onto each detection.
            timestamp: Wall-clock timestamp for the detection (not used internally).

        Returns:
            List of IncidentDetection carrying raw object classes and track IDs.
        """
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            return []

        return self._detect_real(frame, camera_id=camera_id)

    def _detect_real(self, frame: np.ndarray, camera_id: str | None = None) -> list[IncidentDetection]:
        """Run YOLO inference, preferring tracking so track IDs stay stable across frames."""
        if self.use_tracking:
            # persist=True carries the tracker's state between calls, which is what makes
            # box.id stable. Without it every frame restarts numbering and the temporal
            # rules can never accumulate history.
            results = self.model.track(frame, conf=self.conf_threshold, persist=True, verbose=False)
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

                label_name = r.names[cls_id] if hasattr(r, "names") and cls_id in r.names else f"class_{cls_id}"
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
