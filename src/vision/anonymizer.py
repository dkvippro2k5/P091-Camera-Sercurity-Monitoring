import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceAnonymizer:
    """Detects face regions and blurs/pixelates them to support privacy requirements.

    NOTE: backed by OpenCV's frontal-face Haar cascade, which only finds faces looking
    roughly at the camera. Profiles, turned heads, and small or occluded faces pass
    through unblurred, so this reduces identifiability but is not sufficient on its own
    to call a deployment GDPR-compliant. Swap in a modern detector (YuNet, SCRFD) before
    relying on it for that.
    """

    def __init__(
        self,
        blur_kernel_size: tuple[int, int] = (51, 51),
        min_face_size: tuple[int, int] = (30, 30),
    ):
        self.blur_kernel_size = blur_kernel_size
        self.min_face_size = min_face_size

        # Try loading OpenCV built-in Haar Cascade classifier safely
        self.face_cascade = None
        try:
            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            cascade_data = getattr(cv2, "data", None)
            if cascade_cls and cascade_data and hasattr(cascade_data, "haarcascades"):
                cascade_path = cascade_data.haarcascades + "haarcascade_frontalface_default.xml"
                classifier = cascade_cls(cascade_path)
                if not classifier.empty():
                    self.face_cascade = classifier
        except Exception as e:
            logger.warning(f"Could not load Haar Cascade face detector classifier: {e}")

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in frame, returning list of (x, y, w, h)."""
        if frame is None or frame.size == 0:
            return []

        if self.face_cascade is not None and not self.face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size,
            )
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return []

    def blur_faces(
        self,
        frame: np.ndarray,
        faces: list[tuple[int, int, int, int]] | None = None,
        method: str = "gaussian",
    ) -> np.ndarray:
        """Apply blur on detected face bounding regions.

        Args:
            frame: Input BGR numpy image array.
            faces: Optional list of (x, y, w, h). If None, detect_faces will be called.
            method: 'gaussian' or 'pixelate'.

        Returns:
            Anonymized BGR image numpy array. This is `frame` itself, unmodified, when
            there was nothing to blur — callers that intend to draw on the result must
            copy it first.
        """
        if frame is None or frame.size == 0:
            return frame

        if faces is None:
            faces = self.detect_faces(frame)

        # Fast path: no faces detected, skip the expensive frame copy.
        if not faces:
            return frame

        anonymized = frame.copy()

        for (x, y, w, h) in faces:
            # Ensure ROI boundaries are within image bounds
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

            roi = anonymized[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            if method == "gaussian":
                # Ensure kernel sizes are odd numbers
                kx = self.blur_kernel_size[0] if self.blur_kernel_size[0] % 2 == 1 else self.blur_kernel_size[0] + 1
                ky = self.blur_kernel_size[1] if self.blur_kernel_size[1] % 2 == 1 else self.blur_kernel_size[1] + 1
                blurred_roi = cv2.GaussianBlur(roi, (kx, ky), 0)
            elif method == "pixelate":
                # Resize down then up for pixelated effect. Scale back to the clamped ROI
                # size, not the requested (w, h), so boxes running past the frame edge fit.
                small_roi = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                blurred_roi = cv2.resize(
                    small_roi, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            else:
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)

            anonymized[y_min:y_max, x_min:x_max] = blurred_roi

        return anonymized
