"""Temporal rule engine turning per-frame object detections into incident events.

The detector answers "what objects are in this frame". The rules here answer "is this
an incident", which needs history: a person must be *inside a zone*, a fall must *persist*
past a dwell threshold, a bag must be *left alone* for a while. All three need stable
track IDs across frames, so every rule keys its state on track_id.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from src.vision.schemas import BoundingBox, IncidentDetection
from src.vision.zone_config import ZoneManager

logger = logging.getLogger(__name__)

# Raw detector classes the rules consume, matching COCO naming from a stock YOLO model.
PERSON_CLASSES = {"person"}
CARRIABLE_CLASSES = {"backpack", "handbag", "suitcase", "bag", "luggage"}

# Incident labels emitted downstream. Must stay in sync with PRD section 3.1.
LABEL_INTRUSION = "intrusion"
LABEL_FALL = "fall_detection"
LABEL_ABANDONED = "abandoned_object"


@dataclass
class TrackState:
    """Rolling state for one tracked object."""

    track_id: str
    label: str
    first_seen: float
    last_seen: float
    last_bbox: BoundingBox
    # Fall: when the bbox first went wide-and-flat, reset whenever it stands back up.
    fallen_since: float | None = None
    fall_fired: bool = False
    # Abandoned: when the object was last near a person, and where it settled.
    last_owner_seen: float | None = None
    stationary_since: float | None = None
    anchor: tuple[float, float] | None = None
    abandoned_fired: bool = False
    # Intrusion: zones already alerted on, so one entry is not re-fired every frame.
    alerted_zones: set[str] = field(default_factory=set)


class RuleEngine:
    """Stateful evaluator that converts tracked detections into incident events.

    Args:
        zone_manager: Polygon zones used by the intrusion rule.
        fall_dwell_sec: Seconds a person must stay collapsed before a fall fires (PRD: 3s).
        fall_aspect_ratio: Width/height above which a bbox counts as collapsed.
        abandon_sec: Seconds an object must sit unattended before it is abandoned.
        owner_radius_ratio: Owner search radius as a fraction of frame width (PRD's radius R).
        stationary_tolerance_ratio: Movement below this fraction of frame width counts as still.
        track_ttl_sec: Drop track state this long after a track disappears.
    """

    def __init__(
        self,
        zone_manager: ZoneManager | None = None,
        fall_dwell_sec: float = 3.0,
        fall_aspect_ratio: float = 1.2,
        abandon_sec: float = 10.0,
        owner_radius_ratio: float = 0.25,
        stationary_tolerance_ratio: float = 0.02,
        track_ttl_sec: float = 5.0,
    ):
        self.zone_manager = zone_manager or ZoneManager()
        self.fall_dwell_sec = fall_dwell_sec
        self.fall_aspect_ratio = fall_aspect_ratio
        self.abandon_sec = abandon_sec
        self.owner_radius_ratio = owner_radius_ratio
        self.stationary_tolerance_ratio = stationary_tolerance_ratio
        self.track_ttl_sec = track_ttl_sec

        self._tracks: dict[str, TrackState] = {}

    def reset(self) -> None:
        """Clear all track state, e.g. between videos."""
        self._tracks.clear()

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def evaluate(
        self,
        detections: list[IncidentDetection],
        frame_width: int,
        frame_height: int,
        timestamp: float,
        now: datetime,
        camera_id: str | None = None,
    ) -> list[IncidentDetection]:
        """Evaluate raw tracked detections and return incident events for this frame.

        Args:
            detections: Raw object detections carrying stable track_ids.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.
            timestamp: Monotonic seconds for this frame; drives all dwell timers.
            now: Wall-clock time, used only for zone active-hour checks.
            camera_id: Source camera, stamped onto emitted events.

        Returns:
            Incident events triggered on this frame. Empty when nothing fires.
        """
        if frame_width <= 0 or frame_height <= 0:
            return []

        persons = [d for d in detections if d.label in PERSON_CLASSES]
        objects = [d for d in detections if d.label in CARRIABLE_CLASSES]

        self._update_tracks(detections, timestamp)

        events: list[IncidentDetection] = []
        events.extend(
            self._check_intrusion(persons, frame_width, frame_height, timestamp, now, camera_id)
        )
        events.extend(self._check_falls(persons, timestamp, camera_id))
        events.extend(
            self._check_abandoned(objects, persons, frame_width, timestamp, camera_id)
        )

        self._evict_stale(timestamp)
        return events

    def _update_tracks(self, detections: list[IncidentDetection], timestamp: float) -> None:
        for det in detections:
            if not det.track_id:
                continue
            state = self._tracks.get(det.track_id)
            if state is None:
                self._tracks[det.track_id] = TrackState(
                    track_id=det.track_id,
                    label=det.label,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    last_bbox=det.bbox,
                )
            else:
                state.last_seen = timestamp
                state.last_bbox = det.bbox

    def _evict_stale(self, timestamp: float) -> None:
        expired = [
            tid for tid, s in self._tracks.items() if timestamp - s.last_seen > self.track_ttl_sec
        ]
        for tid in expired:
            del self._tracks[tid]

    def _check_intrusion(
        self,
        persons: list[IncidentDetection],
        frame_width: int,
        frame_height: int,
        timestamp: float,
        now: datetime,
        camera_id: str | None,
    ) -> list[IncidentDetection]:
        """Fire when a person's ground point enters an active restricted zone."""
        events = []
        for person in persons:
            if not person.track_id:
                continue
            state = self._tracks.get(person.track_id)
            if state is None:
                continue

            # Bottom-center approximates where the person stands, which is what the
            # zone is drawn around. Using the box centroid would trigger early from
            # a torso overhanging the boundary line.
            foot_x = (person.bbox.xmin + person.bbox.xmax) / 2.0
            foot_y = float(person.bbox.ymax)

            hits = self.zone_manager.zones_containing(
                camera_id, foot_x, foot_y, frame_width, frame_height, now
            )
            current = {z.name for z in hits}

            for zone in hits:
                if zone.name in state.alerted_zones:
                    continue
                state.alerted_zones.add(zone.name)
                window = zone.active_hours or "24/7"
                events.append(
                    IncidentDetection(
                        label=LABEL_INTRUSION,
                        track_id=person.track_id,
                        confidence=person.confidence,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        bbox=person.bbox,
                        zone_name=zone.name,
                        reason=(
                            f"{person.track_id} entered restricted zone "
                            f"'{zone.name}' during {window}"
                        ),
                    )
                )

            # Leaving the zone re-arms it, so a later re-entry alerts again.
            state.alerted_zones &= current

        return events

    def _check_falls(
        self, persons: list[IncidentDetection], timestamp: float, camera_id: str | None
    ) -> list[IncidentDetection]:
        """Fire when a person's bbox stays wide-and-flat past the dwell threshold.

        The dwell is what separates a fall from bending over to pick something up.
        """
        events = []
        for person in persons:
            if not person.track_id:
                continue
            state = self._tracks.get(person.track_id)
            if state is None:
                continue

            width = person.bbox.xmax - person.bbox.xmin
            height = max(1, person.bbox.ymax - person.bbox.ymin)
            collapsed = (width / height) >= self.fall_aspect_ratio

            if not collapsed:
                state.fallen_since = None
                state.fall_fired = False
                continue

            if state.fallen_since is None:
                state.fallen_since = timestamp
                continue

            elapsed = timestamp - state.fallen_since
            if elapsed >= self.fall_dwell_sec and not state.fall_fired:
                state.fall_fired = True
                events.append(
                    IncidentDetection(
                        label=LABEL_FALL,
                        track_id=person.track_id,
                        confidence=person.confidence,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        bbox=person.bbox,
                        reason=(
                            f"{person.track_id} stayed collapsed for {elapsed:.1f}s "
                            f"(threshold {self.fall_dwell_sec}s)"
                        ),
                    )
                )

        return events

    def _check_abandoned(
        self,
        objects: list[IncidentDetection],
        persons: list[IncidentDetection],
        frame_width: int,
        timestamp: float,
        camera_id: str | None,
    ) -> list[IncidentDetection]:
        """Fire when a carriable object sits still with no person inside radius R."""
        events = []
        radius = frame_width * self.owner_radius_ratio
        tolerance = frame_width * self.stationary_tolerance_ratio

        for obj in objects:
            if not obj.track_id:
                continue
            state = self._tracks.get(obj.track_id)
            if state is None:
                continue

            cx = (obj.bbox.xmin + obj.bbox.xmax) / 2.0
            cy = (obj.bbox.ymin + obj.bbox.ymax) / 2.0

            # Reset the stationary clock if the object was carried somewhere new.
            if state.anchor is None or math.dist(state.anchor, (cx, cy)) > tolerance:
                state.anchor = (cx, cy)
                state.stationary_since = timestamp
                state.abandoned_fired = False

            owner_near = any(
                math.dist(
                    ((p.bbox.xmin + p.bbox.xmax) / 2.0, (p.bbox.ymin + p.bbox.ymax) / 2.0),
                    (cx, cy),
                )
                <= radius
                for p in persons
            )
            if owner_near:
                state.last_owner_seen = timestamp
                state.abandoned_fired = False
                continue

            # Never-attended objects age from when they were first seen sitting still.
            unattended_since = state.last_owner_seen or state.stationary_since or timestamp
            elapsed = timestamp - unattended_since

            if elapsed >= self.abandon_sec and not state.abandoned_fired:
                state.abandoned_fired = True
                events.append(
                    IncidentDetection(
                        label=LABEL_ABANDONED,
                        track_id=obj.track_id,
                        confidence=obj.confidence,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        bbox=obj.bbox,
                        reason=(
                            f"{obj.label} ({obj.track_id}) unattended for {elapsed:.1f}s "
                            f"with no person within {radius:.0f}px"
                        ),
                    )
                )

        return events
