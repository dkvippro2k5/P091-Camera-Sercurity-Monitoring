"""Frame overlays: polygon zones and triggered incidents drawn onto output video.

Kept separate from the pipeline so drawing can be swapped or disabled without touching
detection logic. All drawing happens on the anonymized frame, never the frame the
detector sees.
"""

from datetime import datetime

import cv2
import numpy as np

from src.vision.rule_engine import LABEL_ABANDONED, LABEL_FALL, LABEL_INTRUSION
from src.vision.schemas import IncidentDetection, Zone
from src.vision.zone_config import is_zone_active

# BGR, since OpenCV.
COLOR_ZONE_ACTIVE = (60, 180, 255)
COLOR_ZONE_IDLE = (140, 140, 140)
EVENT_COLORS = {
    LABEL_INTRUSION: (60, 60, 255),
    LABEL_FALL: (0, 140, 255),
    LABEL_ABANDONED: (0, 215, 255),
}
DEFAULT_EVENT_COLOR = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_zones(frame: np.ndarray, zones: list[Zone], now: datetime) -> np.ndarray:
    """Draw each zone polygon, dimmed when outside its active hours."""
    height, width = frame.shape[:2]

    for zone in zones:
        active = is_zone_active(zone, now)
        color = COLOR_ZONE_ACTIVE if active else COLOR_ZONE_IDLE
        pts = np.array(
            [[int(x * width), int(y * height)] for x, y in zone.polygon], dtype=np.int32
        ).reshape((-1, 1, 2))

        # Translucent fill so the scene stays readable underneath.
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        label = zone.name if active else f"{zone.name} (idle)"
        anchor = pts[0][0]
        cv2.putText(
            frame, label, (int(anchor[0]), max(14, int(anchor[1]) - 8)),
            _FONT, 0.5, color, 1, cv2.LINE_AA,
        )

    return frame


def draw_events(frame: np.ndarray, events: list[IncidentDetection]) -> np.ndarray:
    """Draw a labelled box for each triggered incident."""
    for event in events:
        color = EVENT_COLORS.get(event.label, DEFAULT_EVENT_COLOR)
        box = event.bbox
        cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 2)

        caption = f"{event.label.upper()} {event.track_id or ''}".strip()
        if event.zone_name:
            caption += f" @{event.zone_name}"

        (text_w, text_h), _ = cv2.getTextSize(caption, _FONT, 0.5, 1)
        text_y = max(text_h + 4, box.ymin - 6)
        cv2.rectangle(
            frame,
            (box.xmin, text_y - text_h - 4),
            (box.xmin + text_w + 6, text_y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame, caption, (box.xmin + 3, text_y), _FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA
        )

    return frame


def draw_tracks(frame: np.ndarray, detections: list[IncidentDetection]) -> np.ndarray:
    """Draw thin boxes for tracked objects that did not trigger anything."""
    for det in detections:
        box = det.bbox
        cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), (90, 90, 90), 1)
        if det.track_id:
            cv2.putText(
                frame, det.track_id, (box.xmin, min(frame.shape[0] - 4, box.ymax + 12)),
                _FONT, 0.4, (170, 170, 170), 1, cv2.LINE_AA,
            )
    return frame
