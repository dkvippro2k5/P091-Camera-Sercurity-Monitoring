"""Colour signature of a person, to recognise a bag's owner again under a new track ID.

The tracker hands out a new ID whenever someone is hidden for a moment, and in a crowd it
also swaps IDs between people. The abandoned-object rule binds a bag to the people around it
when it is put down, so without a second cue an owner who reappears under a new ID reads as
gone and the bag is reported while they are standing beside it.

The signature is a hue/saturation/value histogram of the torso and one of the legs: what a
camera at a few metres can still tell apart, without a re-identification network or extra
weights. It says "same clothes", not "same person" - two commuters in dark suits look alike -
so the rule only uses it for people already standing near the bag.
"""

import cv2
import numpy as np

from src.vision.schemas import BoundingBox, IncidentDetection

# Vertical bands of the person box, as fractions of its height: torso, then legs. The head
# and feet are left out (hair, shadow, floor), as are the box's side margins.
_PARTS = ((0.15, 0.5), (0.55, 0.9))
_SIDE_MARGIN = 0.2


class AppearanceEncoder:
    """Torso and leg colour histograms per person box.

    Args:
        hue_bins, sat_bins, val_bins: Histogram resolution. Value gets its own bins because
            dark clothing has no reliable hue; black, grey and white are told apart by it.
        min_height_px: Boxes shorter than this are too small to describe and get no entry.
    """

    def __init__(self, hue_bins: int = 12, sat_bins: int = 3, val_bins: int = 4, min_height_px: int = 24):
        self.bins = [hue_bins, sat_bins, val_bins]
        self.min_height_px = min_height_px

    def encode(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
        """(parts, bins) normalised histograms, or None when the box is too small or off-frame."""
        frame_h, frame_w = frame.shape[:2]
        x0, x1 = max(0, bbox.xmin), min(frame_w, bbox.xmax)
        y0, y1 = max(0, bbox.ymin), min(frame_h, bbox.ymax)
        height, width = y1 - y0, x1 - x0
        if height < self.min_height_px or width < 4:
            return None
        inner_x0 = x0 + int(width * _SIDE_MARGIN)
        inner_x1 = max(inner_x0 + 1, x1 - int(width * _SIDE_MARGIN))
        parts = []
        for top, bottom in _PARTS:
            crop = frame[y0 + int(height * top) : y0 + max(int(height * bottom), int(height * top) + 1), inner_x0:inner_x1]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256]).ravel()
            parts.append(hist / max(hist.sum(), 1.0))
        return np.stack(parts).astype(np.float32)

    def measure(self, frame: np.ndarray, persons: list[IncidentDetection], track_ids: set[str]) -> dict[str, np.ndarray]:
        """Signatures for the persons in `track_ids`; people too small to describe are left out."""
        signatures = {}
        for person in persons:
            if person.track_id in track_ids:
                signature = self.encode(frame, person.bbox)
                if signature is not None:
                    signatures[person.track_id] = signature
        return signatures


def appearance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection averaged over the body parts: 1 is identical, 0 disjoint."""
    return float(np.minimum(a, b).sum(axis=1).mean())
