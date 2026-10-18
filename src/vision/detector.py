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
# including CI and Docker.
#
# `yolo26n.pt` used to be the default here, but that checkpoint is not shipped in this
# repo and nothing fetches it — every caller that builds an IncidentDetector (or a bare
# VisionPipeline()) with no explicit model_path, which is most of the test suite, raised
# FileNotFoundError on construction (verified 2026-09-08). `yolo26s.pt` is the weight file
# actually committed under src/vision/weights/: the .env template's own comment records
# that yolo26n was tried and reverted for missing over half its detections against
# yolo26l on a real clip, while yolo26s matched yolo26l's recall at ~3x the speed. Settings
# in src/config.py carries the same value for MonitoringService's real pipeline; this
# constant is what a bare IncidentDetector()/VisionPipeline() falls back to when
# Settings is bypassed entirely, which is exactly what construction-only tests do.
DEFAULT_WEIGHTS_PATH = os.path.join("src", "vision", "weights", "yolo26s.pt")

# Pinned tracker config; see the file's header for why the library default is not used.
DEFAULT_TRACKER_CONFIG = os.path.join("src", "vision", "trackers", "bytetrack.yaml")

# A track not seen for this many calls is forgotten by the established-track filter. Well
# past the tracker's own track_buffer, so a track the tracker can still revive is never
# forgotten here first.
_ESTABLISHED_TRACK_TTL_CALLS = 300


class IncidentDetector:
    """Object detector feeding the rule engine.

    This layer answers "what objects are in this frame", not "is this an incident".
    It emits raw classes (person, backpack, ...) with stable track IDs; deciding whether
    they add up to intrusion / fall / abandoned_object is the RuleEngine's job, because
    those labels need history that a single frame cannot supply.

    A model_path of None means no model is loaded and detect() always returns [] —
    useful for tests that only exercise the rule layer.

    With tracking on, `conf_threshold` gates *new* objects only. Detections down to the
    tracker's `track_low_thresh` are fed to the tracker, and a track that has once scored at
    or above `conf_threshold` keeps being reported while weaker detections sustain it. That is
    ByteTrack's whole design, and it matters most at the moment this system cares about: a
    person lying on the floor often scores 0.3-0.6, and filtering them out before tracking
    broke the track at impact, so the fall rule never saw anyone land.
    """

    def __init__(
        self,
        model_path: str | None = DEFAULT_WEIGHTS_PATH,
        conf_threshold: float = 0.5,
        target_labels: list[str] | None = None,
        use_tracking: bool = True,
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # None keeps every class the model emits; a list filters to those classes.
        self.target_labels = target_labels
        self.use_tracking = use_tracking
        self.tracker_config = tracker_config
        self.model = None
        # Confidence passed to model.track(): the tracker's low threshold, so its second
        # association stage actually receives the weak detections it exists to use.
        self._track_feed_conf = conf_threshold
        # track_id -> call index it was last reported at, for tracks that have cleared
        # conf_threshold at least once.
        self._established: dict[str, int] = {}
        # track_id -> call index last reported, for tracks that have been a person.
        self._person_tracks: dict[str, int] = {}
        self._calls = 0

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

        if self.use_tracking:
            tracker_path = self._resolve_weights(self.tracker_config)
            if tracker_path is None:
                raise FileNotFoundError(f"Tracker config not found at '{self.tracker_config}'.")
            self.tracker_config = tracker_path
            import yaml  # type: ignore  # ultralytics dependency

            with open(tracker_path, encoding="utf-8") as handle:
                low = float(yaml.safe_load(handle).get("track_low_thresh", self.conf_threshold))
            self._track_feed_conf = min(low, self.conf_threshold)

    @staticmethod
    def _resolve_weights(model_path: str | None) -> str | None:
        """Find a repo-relative file (weights, tracker config), tolerating any CWD."""
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
            results = self.model.track(
                frame,
                conf=self._track_feed_conf,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )
        else:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

        self._calls += 1
        detections: list[IncidentDetection] = []
        current_time = round(time.time(), 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                label_name = r.names[cls_id] if hasattr(r, "names") and cls_id in r.names else f"class_{cls_id}"

                # Left as None when the tracker gave us nothing. A synthesized ID would
                # collide across objects and silently corrupt every temporal rule.
                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = f"track_{int(box.id[0].cpu().numpy())}"

                # The tracker associates boxes regardless of class, and YOLO often calls a
                # person lying on the floor "bed" or "couch". The track carries on under the
                # wrong label: filtered out, the person vanishes mid-fall; kept, a "bed"
                # overwrites the person's track state (GMDCSA24 s2-fall-04). Once a track has
                # been a person, it stays one.
                if track_id is not None and label_name != "person" and track_id in self._person_tracks:
                    label_name = "person"

                if self.target_labels and label_name not in self.target_labels:
                    continue

                if not self._passes_confidence(track_id, conf):
                    continue
                if label_name == "person" and track_id is not None:
                    self._person_tracks[track_id] = self._calls

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

        self._forget_stale_tracks()
        return detections

    def _passes_confidence(self, track_id: str | None, conf: float) -> bool:
        """New objects need conf_threshold; an established track may continue below it."""
        if conf >= self.conf_threshold:
            if track_id is not None:
                self._established[track_id] = self._calls
            return True
        if track_id is not None and track_id in self._established:
            self._established[track_id] = self._calls
            return True
        return False

    def _forget_stale_tracks(self) -> None:
        for registry in (self._established, self._person_tracks):
            stale = [tid for tid, seen in registry.items() if self._calls - seen > _ESTABLISHED_TRACK_TTL_CALLS]
            for tid in stale:
                del registry[tid]
