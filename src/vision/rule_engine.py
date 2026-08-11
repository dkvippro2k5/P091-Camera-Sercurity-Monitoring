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

# Span of bbox-position history retained per track, and therefore the baseline the fall rule
# measures downward velocity against. One constant so the pruning window and the window the
# velocity is computed over cannot drift apart: `_check_falls` reads `pos_history[0]`, which
# is only the window edge as long as pruning uses this same value.
_VELOCITY_WINDOW_SEC = 2.0


@dataclass
class TrackState:
    """Rolling state for one tracked object."""

    track_id: str
    label: str
    first_seen: float
    last_seen: float
    last_bbox: BoundingBox
    # Fall: when the suspicion armed (>= fall_min_indicators tripped), reset when the AR
    # drops to fall_upright_aspect_ratio or below. armed_reason is a snapshot of the
    # indicator breakdown at arm time — the only moment the transition was still
    # measurable, since post-impact frames read near-zero on velocity and height-drop.
    fallen_since: float | None = None
    armed_reason: str | None = None
    fall_fired: bool = False
    # Abandoned: when the object was last near a person, and where it settled.
    last_owner_seen: float | None = None
    stationary_since: float | None = None
    anchor: tuple[float, float] | None = None
    abandoned_fired: bool = False
    owner_track_id: str | None = None
    # Track Stitching & Motion Velocity
    # (timestamp, y_top, height) feeding the fall rule's V_y and dH indicators.
    pos_history: list[tuple[float, float, float]] = field(default_factory=list)
    aliased_from: str | None = None
    # Intrusion: zones already alerted on, so one entry is not re-fired every frame.
    alerted_zones: set[str] = field(default_factory=set)


class RuleEngine:
    """Stateful evaluator that converts tracked detections into incident events.

    Args:
        zone_manager: Polygon zones used by the intrusion rule.
        fall_dwell_sec: Seconds a person must stay collapsed before a fall fires (PRD §3.1: 3s).
        fall_aspect_ratio: Width/height a bbox must reach or exceed to trip the AR indicator.
        fall_upright_aspect_ratio: Width/height a bbox must drop to or below to count as
            "back on their feet" once a suspicion has armed — the ambiguous band between this
            and fall_aspect_ratio holds the dwell timer instead of resetting it, since a
            fallen person's box drifts through that range as limbs move and the tracker jitters.
        fall_velocity_ratio: Downward velocity indicator threshold, in body-heights per second
            (velocity divided by the person's own box height, so the same fall reads the same
            whether it is close to the camera or down the far end of a corridor).
        fall_height_drop_ratio: Fractional loss of box height within the velocity window
            required to trip the height-drop indicator (a standing box going flat loses most
            of its height; a crouch or a slow sit does not).
        fall_min_indicators: How many of the three indicators (aspect ratio, downward
            velocity, height drop) must trip in the same window to arm a fall suspicion.
            2 of 3 by default — 3 silently drops falls straight toward or away from the
            camera, where the box never goes flat.
        abandon_sec: Seconds an object must sit unattended before it is abandoned.
        owner_radius_ratio: Owner search radius as a fraction of frame width (PRD's radius R).
        stationary_tolerance_ratio: Movement below this fraction of frame width counts as still.
        track_ttl_sec: Drop track state this long after a track disappears.
        track_stitch_window_sec: Grace window to stitch a new track to a recently lost track across brief occlusions.
        track_stitch_radius_ratio: How far a reappearing track may have moved, as a fraction
            of frame width. Normalized for the same reason owner_radius_ratio is: a fixed
            pixel radius means something different at 720p than at 4K, so the stitch would
            silently tighten or loosen with the camera's resolution.
        alert_cooldown_sec: Minimum gap between events of the same label on one camera.
            0 disables. Guards against one flickering track flooding the operator.
    """

    def __init__(
        self,
        zone_manager: ZoneManager | None = None,
        fall_dwell_sec: float = 3.0,
        fall_aspect_ratio: float = 1.2,
        fall_upright_aspect_ratio: float = 0.7,
        fall_velocity_ratio: float = 1.5,
        fall_height_drop_ratio: float = 0.3,
        fall_min_indicators: int = 2,
        abandon_sec: float = 2.0,
        owner_radius_ratio: float = 0.25,
        stationary_tolerance_ratio: float = 0.02,
        track_ttl_sec: float = 5.0,
        track_stitch_window_sec: float = 2.0,
        # 0.12 reproduces the 150px this replaced at the 720p the capacity table was
        # measured on, so behaviour there is unchanged and every other resolution now
        # scales with it instead of drifting.
        track_stitch_radius_ratio: float = 0.12,
        alert_cooldown_sec: float = 0.0,
    ):
        self.zone_manager = zone_manager or ZoneManager()
        self.fall_dwell_sec = fall_dwell_sec
        self.fall_aspect_ratio = fall_aspect_ratio
        self.fall_upright_aspect_ratio = fall_upright_aspect_ratio
        self.fall_velocity_ratio = fall_velocity_ratio
        self.fall_height_drop_ratio = fall_height_drop_ratio
        self.fall_min_indicators = fall_min_indicators
        self.abandon_sec = abandon_sec
        self.owner_radius_ratio = owner_radius_ratio
        self.stationary_tolerance_ratio = stationary_tolerance_ratio
        self.track_ttl_sec = track_ttl_sec
        self.track_stitch_window_sec = track_stitch_window_sec
        self.track_stitch_radius_ratio = track_stitch_radius_ratio
        self.alert_cooldown_sec = alert_cooldown_sec

        # Keyed by (camera_id, track_id) to prevent cross-camera collisions.
        # Two independent YOLO trackers can emit the same track_id (e.g. "track_1");
        # without the camera prefix their temporal states would corrupt each other.
        self._tracks: dict[tuple[str | None, str], TrackState] = {}
        self._expired_tracks: dict[tuple[str | None, str], TrackState] = {}
        # Last emission time per (camera_id, label), for alert_cooldown_sec.
        self._last_alert: dict[tuple[str | None, str], float] = {}

    def reset(self) -> None:
        """Clear all track state, e.g. between videos."""
        self._tracks.clear()
        self._expired_tracks.clear()
        self._last_alert.clear()

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def has_active_tracks(self, camera_id: str | None = None) -> bool:
        """Whether any track state is live for this camera.

        The motion gate consults this before skipping a frame. Skipping while a track is
        alive would freeze its dwell timer mid-count and starve the YOLO tracker, so a
        fall in progress could never reach the 3s threshold.
        """
        return any(key[0] == camera_id for key in self._tracks)

    @staticmethod
    def _track_key(camera_id: str | None, track_id: str) -> tuple[str | None, str]:
        """Composite key so identical track_ids from different cameras stay separate."""
        return (camera_id, track_id)

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

        self._update_tracks(detections, timestamp, camera_id, frame_width)

        events: list[IncidentDetection] = []
        events.extend(self._check_intrusion(persons, frame_width, frame_height, timestamp, now, camera_id))
        events.extend(self._check_falls(persons, timestamp, camera_id))
        events.extend(self._check_abandoned(objects, persons, frame_width, timestamp, camera_id))

        self._evict_stale(timestamp)
        return self._apply_cooldown(events, timestamp, camera_id)

    def _apply_cooldown(
        self, events: list[IncidentDetection], timestamp: float, camera_id: str | None
    ) -> list[IncidentDetection]:
        """Drop events of a label that already fired on this camera too recently.

        Each rule already re-arms per track, but nothing stopped a flickering track — or
        several tracks of the same kind — from emitting a burst an operator cannot triage.
        DeepCamera rate-limits for the same reason (a 200ms frame lock plus a 30-minute
        per-person notification cooldown); this serves the PRD's "giảm báo động giả" goal.

        Applied at the single exit point of evaluate() so no rule can bypass it.
        """
        if self.alert_cooldown_sec <= 0:
            return events

        kept: list[IncidentDetection] = []
        for event in events:
            key = (camera_id, event.label)
            last = self._last_alert.get(key)
            if last is not None and (timestamp - last) < self.alert_cooldown_sec:
                logger.debug(
                    f"Suppressed {event.label} on {camera_id}: "
                    f"{timestamp - last:.1f}s since last, cooldown {self.alert_cooldown_sec}s"
                )
                continue
            self._last_alert[key] = timestamp
            kept.append(event)

        return kept

    def get_active_incidents(
        self,
        detections: list[IncidentDetection],
        camera_id: str | None = None,
    ) -> list[IncidentDetection]:
        """Return all active/ongoing incidents for current frame detections (for persistent overlay drawing)."""
        active: list[IncidentDetection] = []
        for det in detections:
            if not det.track_id:
                continue
            state = self._tracks.get(self._track_key(camera_id, det.track_id))
            if state is None:
                continue

            for zone_name in state.alerted_zones:
                active.append(
                    IncidentDetection(
                        label=LABEL_INTRUSION,
                        track_id=det.track_id,
                        confidence=det.confidence,
                        timestamp=det.timestamp,
                        camera_id=camera_id,
                        bbox=det.bbox,
                        zone_name=zone_name,
                        reason=f"{det.track_id} in restricted zone '{zone_name}'",
                    )
                )

            if state.fall_fired:
                active.append(
                    IncidentDetection(
                        label=LABEL_FALL,
                        track_id=det.track_id,
                        confidence=det.confidence,
                        timestamp=det.timestamp,
                        camera_id=camera_id,
                        bbox=det.bbox,
                        reason=f"{det.track_id} collapsed",
                    )
                )

            if state.abandoned_fired:
                active.append(
                    IncidentDetection(
                        label=LABEL_ABANDONED,
                        track_id=det.track_id,
                        confidence=det.confidence,
                        timestamp=det.timestamp,
                        camera_id=camera_id,
                        bbox=det.bbox,
                        reason=f"{det.label} ({det.track_id}) unattended",
                    )
                )

        return active

    def _update_tracks(
        self,
        detections: list[IncidentDetection],
        timestamp: float,
        camera_id: str | None,
        frame_width: int,
    ) -> None:
        for det in detections:
            if not det.track_id:
                continue
            key = self._track_key(camera_id, det.track_id)
            state = self._tracks.get(key)
            cx = (det.bbox.xmin + det.bbox.xmax) / 2.0
            cy = (det.bbox.ymin + det.bbox.ymax) / 2.0
            ymin = float(det.bbox.ymin)
            box_height = float(max(1, det.bbox.ymax - det.bbox.ymin))

            if state is None:
                state = self._stitch_or_create(det, key, camera_id, timestamp, cx, cy, frame_width)
            else:
                state.last_seen = timestamp
                state.last_bbox = det.bbox

            # Rolling window feeding the fall rule's downward-velocity and height-drop
            # indicators. Pruned to the same span used to compute them, so the oldest sample
            # is the window edge. Tracks the top edge (ymin), not the bottom: a real fall is
            # the head travelling a body height while the feet barely move, so ymax stays
            # nearly flat throughout and would read close to zero velocity on exactly the
            # case this must catch.
            state.pos_history.append((timestamp, ymin, box_height))
            state.pos_history = [p for p in state.pos_history if timestamp - p[0] <= _VELOCITY_WINDOW_SEC]

    def _stitch_or_create(
        self,
        det: IncidentDetection,
        key: tuple[str | None, str],
        camera_id: str | None,
        timestamp: float,
        cx: float,
        cy: float,
        frame_width: int,
    ) -> TrackState:
        """Resume a recently-lost track's state, or start fresh when nothing matches.

        YOLO issues a brand-new track_id whenever a person is briefly occluded — walking
        behind a pillar restarts numbering, which restarts every dwell timer keyed on it.
        Stitching carries the timer across so a 3-second fall behind a chair leg still fires.

        Matching is nearest-first rather than first-found: two objects of the same class can
        both be inside the radius, and picking by dict order would make the outcome depend
        on eviction sequence rather than on geometry.
        """
        # frame_width is guaranteed positive — evaluate() returns early on a degenerate frame.
        radius = frame_width * self.track_stitch_radius_ratio

        best_key: tuple[str | None, str] | None = None
        best_dist = radius
        for exp_key, exp_state in self._expired_tracks.items():
            if exp_key[0] != camera_id or exp_state.label != det.label:
                continue
            if (timestamp - exp_state.last_seen) > self.track_stitch_window_sec:
                continue
            old_cx = (exp_state.last_bbox.xmin + exp_state.last_bbox.xmax) / 2.0
            old_cy = (exp_state.last_bbox.ymin + exp_state.last_bbox.ymax) / 2.0
            dist = math.hypot(cx - old_cx, cy - old_cy)
            if dist < best_dist:
                best_key, best_dist = exp_key, dist

        if best_key is None:
            state = TrackState(
                track_id=det.track_id,
                label=det.label,
                first_seen=timestamp,
                last_seen=timestamp,
                last_bbox=det.bbox,
            )
            self._tracks[key] = state
            return state

        stitched = self._expired_tracks.pop(best_key)
        stitched.aliased_from = stitched.track_id
        stitched.track_id = det.track_id
        stitched.last_seen = timestamp
        stitched.last_bbox = det.bbox
        # The dwell timers are the whole point of stitching, so fallen_since / stationary_since
        # / last_owner_seen ride along. The *fired* flags and position history deliberately do
        # not: keeping `abandoned_fired` would let a bag that already alerted stay silent
        # forever through one occlusion, and stale pos_history would read as a large jump in
        # zero time and fake a downward velocity spike on the first frame after the stitch.
        stitched.fall_fired = False
        stitched.abandoned_fired = False
        stitched.pos_history = []
        self._tracks[key] = stitched
        logger.debug(
            f"Stitched {det.label} {stitched.aliased_from} -> {det.track_id} on {camera_id} "
            f"({best_dist:.0f}px, radius {radius:.0f}px)"
        )
        return stitched

    def _evict_stale(self, timestamp: float) -> None:
        expired_keys = [key for key, s in self._tracks.items() if timestamp - s.last_seen > self.track_ttl_sec]
        for key in expired_keys:
            self._expired_tracks[key] = self._tracks.pop(key)

        # Purge _expired_tracks beyond track_ttl_sec + track_stitch_window_sec
        purge_keys = [
            key for key, s in self._expired_tracks.items()
            if timestamp - s.last_seen > (self.track_ttl_sec + self.track_stitch_window_sec)
        ]
        for key in purge_keys:
            del self._expired_tracks[key]

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
            state = self._tracks.get(self._track_key(camera_id, person.track_id))
            if state is None:
                continue

            # Bottom-center approximates where the person stands, which is what the
            # zone is drawn around. Using the box centroid would trigger early from
            # a torso overhanging the boundary line.
            foot_x = (person.bbox.xmin + person.bbox.xmax) / 2.0
            foot_y = float(person.bbox.ymax)

            hits = self.zone_manager.zones_containing(camera_id, foot_x, foot_y, frame_width, frame_height, now)
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
                        reason=(f"{person.track_id} entered restricted zone '{zone.name}' during {window}"),
                    )
                )

            # Leaving the zone re-arms it, so a later re-entry alerts again.
            state.alerted_zones &= current

        return events

    def _fall_indicators(
        self, person: IncidentDetection, state: TrackState, dt_guard: float = 0.1
    ) -> tuple[int, str]:
        """Score how many of the three fall indicators trip on this frame, and report why.

        Each indicator answers a different question a real fall — not a crouch, not a slow
        sit — answers yes to:
          - AR:  is the box wide-and-flat right now?
          - V_y: did the box's bottom edge move down fast, scaled by the box's own height so
                 the same fall reads the same close to the camera or down a corridor?
          - dH:  did the box lose a large fraction of its height within the window?

        Velocity and height-drop both read the oldest sample still inside pos_history's
        _VELOCITY_WINDOW_SEC, so a gradual descent spreads its motion across many windows and
        never trips either — only a genuinely sudden change does. AR alone needs no history,
        so it is available even on a track's first frame.

        Returns (score, breakdown) where breakdown lists all three with a +/- for whether
        each crossed and the measured value, e.g. "2/3 indicators (+V_y=3.1, +AR=1.40,
        -dH=0.08)" — the PRD routes this to an LLM for false-positive filtering, and a bare
        verdict gives that layer nothing to weigh.
        """
        width = person.bbox.xmax - person.bbox.xmin
        height = max(1, person.bbox.ymax - person.bbox.ymin)
        ar = width / height
        ar_hit = ar >= self.fall_aspect_ratio

        v_y_str = "V_y=n/a"
        v_y_hit = False
        dh_str = "dH=n/a"
        dh_hit = False

        if len(state.pos_history) >= 2:
            first_p = state.pos_history[0]
            dt = state.pos_history[-1][0] - first_p[0]
            if dt > dt_guard:
                dy = float(person.bbox.ymin) - first_p[1]
                v_y = max(0.0, dy / dt) / height
                v_y_hit = v_y >= self.fall_velocity_ratio
                v_y_str = f"V_y={v_y:.1f}"

                first_height = max(1.0, first_p[2])
                drop = max(0.0, (first_height - height) / first_height)
                dh_hit = drop >= self.fall_height_drop_ratio
                dh_str = f"dH={drop:.2f}"

        score = int(ar_hit) + int(v_y_hit) + int(dh_hit)
        breakdown = (
            f"{score}/3 indicators "
            f"({'+' if v_y_hit else '-'}{v_y_str}, "
            f"{'+' if ar_hit else '-'}AR={ar:.2f}, "
            f"{'+' if dh_hit else '-'}{dh_str})"
        )
        return score, breakdown

    def _check_falls(
        self, persons: list[IncidentDetection], timestamp: float, camera_id: str | None
    ) -> list[IncidentDetection]:
        """Fire when a person stays on the ground past the dwell threshold.

        The dwell is what separates a fall from bending over to pick something up, and the
        PRD fixes it at 3 seconds (§3.1). The suspicion latches on arming instead of
        re-scoring every frame: within one velocity window of landing, V_y and dH have
        already decayed to zero, so demanding the score hold for the whole dwell would make
        any dwell longer than the window mathematically unreachable.
        """
        events = []
        for person in persons:
            if not person.track_id:
                continue
            state = self._tracks.get(self._track_key(camera_id, person.track_id))
            if state is None:
                continue

            width = person.bbox.xmax - person.bbox.xmin
            height = max(1, person.bbox.ymax - person.bbox.ymin)
            ar = width / height

            if state.fallen_since is not None and ar <= self.fall_upright_aspect_ratio:
                # Clearly back on their feet — the ambiguous band above this is deliberately
                # NOT treated as "stood up", since a fallen person's box drifts through it as
                # limbs move and the tracker jitters, and resetting on that noise would drop
                # dwells that should have fired.
                state.fallen_since = None
                state.armed_reason = None
                state.fall_fired = False
                # A second fall must re-arm on its own transition, not on samples the prior
                # collapse left inside the velocity window — otherwise the indicators measure
                # zero change against still-collapsed history and never trip again.
                state.pos_history = []
                continue

            if state.fallen_since is None:
                score, breakdown = self._fall_indicators(person, state)
                if score < self.fall_min_indicators:
                    continue
                state.fallen_since = timestamp
                state.armed_reason = breakdown
                if self.fall_dwell_sec > 0:
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
                            f"(threshold {self.fall_dwell_sec}s) — {state.armed_reason}"
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

        for obj in objects:
            if not obj.track_id:
                continue
            state = self._tracks.get(self._track_key(camera_id, obj.track_id))
            if state is None:
                continue

            cx = (obj.bbox.xmin + obj.bbox.xmax) / 2.0
            cy = (obj.bbox.ymin + obj.bbox.ymax) / 2.0

            if state.stationary_since is None:
                state.stationary_since = timestamp

            # Original Owner Tracking (Improvement 3): Bind owner_track_id if not set
            if state.owner_track_id is None and persons:
                close_owners = [
                    p
                    for p in persons
                    if p.track_id
                    and math.dist(
                        ((p.bbox.xmin + p.bbox.xmax) / 2.0, (p.bbox.ymin + p.bbox.ymax) / 2.0),
                        (cx, cy),
                    )
                    <= radius
                ]
                if close_owners:
                    state.owner_track_id = close_owners[0].track_id

            if state.owner_track_id:
                # Check if the specific original owner is present and near
                owner_near = any(
                    p.track_id == state.owner_track_id
                    and math.dist(
                        ((p.bbox.xmin + p.bbox.xmax) / 2.0, (p.bbox.ymin + p.bbox.ymax) / 2.0),
                        (cx, cy),
                    )
                    <= radius
                    for p in persons
                )
            else:
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
            # `is not None` rather than truthiness: process_video passes video-relative
            # time, so an owner last seen on the very first frame gives exactly 0.0, and
            # `or` would discard it and restart the clock from the stationary anchor.
            if state.last_owner_seen is not None:
                unattended_since = state.last_owner_seen
            elif state.stationary_since is not None:
                unattended_since = state.stationary_since
            else:
                unattended_since = timestamp
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
