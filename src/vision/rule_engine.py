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
# velocity is computed over cannot drift apart: `_fall_indicators` reads `pos_history[0]`,
# which is only the window edge as long as pruning uses this same value.
#
# 0.5s is the spec's window ("khoảng 15 frames" at the ~30fps it assumes). It must stay short:
# every indicator measures *change* against `pos_history[0]`, so a window long enough to still
# hold post-impact frames makes that baseline a person already on the ground — V_y and ΔH then
# read ~0 and a second fall on the same track can never re-arm. At 2.0s that is exactly what
# happened, and `test_fall_fires_after_dwell_and_resets_when_upright` caught it.
_VELOCITY_WINDOW_SEC = 0.5

# Below this dt the velocity indicator abstains instead of dividing by a near-zero interval,
# where one pixel of tracker jitter reads as tens of body-heights per second.
_MIN_VELOCITY_DT_SEC = 0.05

# Marks a breakdown string as "this person is standing". `_evaluate_fall_indicators` matches on
# it to clear the latch, so it is a shared constant rather than a literal repeated in two places:
# the two spellings drifting apart would leave the latch permanently armed, and the resulting
# fall would fire the dwell after the person walked away rather than not at all.
_UPRIGHT_PREFIX = "upright"


@dataclass
class TrackState:
    """Rolling state for one tracked object."""

    track_id: str
    label: str
    first_seen: float
    last_seen: float
    last_bbox: BoundingBox
    # Fall: when the indicators first agreed the box collapsed, cleared once the person reads
    # upright again. `fall_reason` is captured at that same moment on purpose — V_y and ΔH are
    # transient, so by the time the dwell elapses they have decayed to zero and could no longer
    # explain the alert they caused.
    fallen_since: float | None = None
    fall_fired: bool = False
    fall_latched: bool = False
    fall_reason: str = ""
    # Abandoned: when the object was last near a person, and where it settled.
    last_owner_seen: float | None = None
    stationary_since: float | None = None
    anchor: tuple[float, float] | None = None
    abandoned_fired: bool = False
    owner_track_id: str | None = None
    # Track stitching & fall indicators: (timestamp, y_top, height) per analyzed frame.
    # The centroid this used to carry was dead weight — stitching matches on `last_bbox`, not on
    # this history, and the fall rule only ever needs the box's top edge and its height.
    pos_history: list[tuple[float, float, float]] = field(default_factory=list)
    aliased_from: str | None = None
    # Intrusion: zones already alerted on, so one entry is not re-fired every frame.
    alerted_zones: set[str] = field(default_factory=set)


class RuleEngine:
    """Stateful evaluator that converts tracked detections into incident events.

    Args:
        zone_manager: Polygon zones used by the intrusion rule.
        fall_dwell_sec: Seconds a suspected fall must persist before it fires (PRD: 3s).
        fall_aspect_ratio: Indicator 2. Width/height at or above which the box counts as
            gone flat. 1.2 is the spec's figure for a fallen person.
        fall_upright_aspect_ratio: Width/height below which the person is walking normally
            again, which disarms a suspected fall. 0.7 is the spec's figure for someone
            standing or walking. The 0.7-1.2 band is deliberately neither — too flat to be
            walking, too tall to be lying — so the dwell timer holds instead of flapping.
        fall_velocity_ratio: Indicator 1. Downward speed of the box's top edge, in the
            person's own body-heights per second, at or above which they are dropping
            rather than lowering themselves. Normalized by body height rather than by
            frame height so a fall at the far end of a corridor still crosses it; a pixel
            threshold would only ever fire near the lens.
        fall_height_drop_ratio: Indicator 3. Fraction of its own height the box must lose
            inside the window to count as collapsing. 0.5 is the spec's "co lại hơn 50%".
        fall_min_indicators: How many of the three indicators must agree before a fall is
            suspected. The spec lists the indicators but not how they combine; 2 of 3 is
            this module's choice — 3/3 only ever catches a textbook fall, while 1/3 lets a
            crouch near a wide-angle lens through on aspect ratio alone.
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
        # PRD 3.1 requires the fall state to persist 3 seconds ("quá 3 giây") so picking
        # something up off the floor is not an alert. It is a requirement, not a tunable —
        # lower it only for a test that is isolating the indicator scoring from the timer.
        # Survivable at this frame rate because _stitch_or_create carries `fallen_since`
        # across the new track_id YOLO issues after a brief occlusion; without that, a
        # chair leg between camera and casualty would restart the 3s from zero.
        fall_dwell_sec: float = 3.0,
        fall_aspect_ratio: float = 1.2,
        fall_upright_aspect_ratio: float = 0.7,
        # A standing adult's head travels roughly one body height during a fall, and the
        # collapse phase of a real fall lasts ~0.4-0.7s, putting it near 1.5-2.5 H/s.
        # Deliberately sitting or crouching moves maybe half a body height over a second or
        # more, well under 0.5 H/s. 1.2 sits between the two with margin on both sides.
        # This is a reasoned default, not one measured against labelled falls.
        fall_velocity_ratio: float = 1.2,
        fall_height_drop_ratio: float = 0.5,
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

            if state is None:
                state = self._stitch_or_create(det, key, camera_id, timestamp, cx, cy, frame_width)
            else:
                state.last_seen = timestamp
                state.last_bbox = det.bbox

            # Rolling window feeding the fall rule's V_y and ΔH indicators. Pruned to the
            # same span they are computed over, so the oldest sample is the window edge.
            state.pos_history.append((timestamp, float(det.bbox.ymin), float(max(1, det.bbox.ymax - det.bbox.ymin))))
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

    def _fall_indicators(self, person: IncidentDetection, state: TrackState, timestamp: float) -> tuple[int, str]:
        """Score the three "sudden change" indicators for one person, and explain the score.

        Returns (how many fired, a breakdown naming all three with their measured values).
        The breakdown is not a debug aid — it becomes the event's `reason`, which PRD 1.2 feeds
        to an LLM to filter false positives. A model handed only "fall detected" has nothing to
        reason about, so every indicator reports its value and its threshold whether it fired or
        not: "-AR=0.83(>=1.2)" is what tells the model the box never went flat, which is the
        difference between a fall directly away from the camera and a false positive.

        All three measure change across `_VELOCITY_WINDOW_SEC` against the oldest sample still in
        the window, which makes them transient by construction. That is why the caller latches on
        this score instead of re-checking it — see `_evaluate_fall_indicators`.
        """
        width = person.bbox.xmax - person.bbox.xmin
        height = max(1, person.bbox.ymax - person.bbox.ymin)
        ar = width / height

        oldest = state.pos_history[0] if state.pos_history else (timestamp, float(person.bbox.ymin), float(height))
        t0, y0, h0 = oldest
        dt = timestamp - t0
        # Both rates baseline on the height at the *window's start*. After the collapse the box
        # is short, so normalizing by the current height would divide a real fall's numbers by a
        # fraction of the person and inflate both indicators on exactly the frames that matter.
        h_ref = max(1.0, h0)

        # Image Y grows downward, so a falling person's top edge ymin INCREASES. PRD 3.1 says
        # "Y giảm đột ngột", which is *elevation* dropping; implementing that as the pixel value
        # decreasing inverts the rule into a stand-up detector, which still fires on fall footage
        # (people get up afterwards) and so survives an end-to-end smoke test.
        v_str = "V_y=n/a"
        vel_hit = False
        if dt >= _MIN_VELOCITY_DT_SEC:
            v_val = max(0.0, (float(person.bbox.ymin) - y0) / dt) / h_ref
            vel_hit = v_val >= self.fall_velocity_ratio
            v_str = f"V_y={v_val:.2f}H/s"

        dh_val = max(0.0, (h0 - height) / h_ref) if dt > 0 else 0.0
        dh_hit = dh_val >= self.fall_height_drop_ratio
        ar_hit = ar >= self.fall_aspect_ratio

        breakdown = (
            f"{'+' if vel_hit else '-'}{v_str}(>={self.fall_velocity_ratio}) "
            f"{'+' if ar_hit else '-'}AR={ar:.2f}(>={self.fall_aspect_ratio}) "
            f"{'+' if dh_hit else '-'}dH={int(dh_val * 100)}%(>={int(self.fall_height_drop_ratio * 100)}%)"
        )

        # Measured before the upright test so a standing person's breakdown still carries the
        # numbers rather than a bare "upright". The pair is what makes a *missed* fall
        # diagnosable from the log alone, which is otherwise the one failure leaving no trace.
        if ar < self.fall_upright_aspect_ratio:
            return 0, f"{_UPRIGHT_PREFIX} (AR {ar:.2f}) [{breakdown}]"

        hits = sum((vel_hit, ar_hit, dh_hit))
        return hits, f"{hits}/3 indicators [{breakdown}]"

    def _evaluate_fall_indicators(self, person: IncidentDetection, state: TrackState, timestamp: float) -> bool:
        """Maintain the fall latch for one track. Returns whether a fall is currently suspected.

        The latch exists because the indicators cannot be re-checked. They measure change over a
        0.5s window, so within half a second of impact the pre-fall samples have pruned out: V_y
        and ΔH decay to zero and a motionless casualty scores AR alone, 1/3. Requiring the score
        to *hold* for the dwell would therefore make any dwell longer than the window
        mathematically unreachable — silently including the PRD's 3 seconds, with the rule looking
        entirely reasonable while never firing once.

        So the score only ever arms the latch, and only the person reading upright again clears
        it. The 0.7-1.2 aspect band is neither, which holds the timer through tracker jitter
        instead of flapping it.
        """
        hits, breakdown = self._fall_indicators(person, state, timestamp)

        if breakdown.startswith(_UPRIGHT_PREFIX):
            state.fall_latched = False
            state.fall_reason = ""
            state.fallen_since = None
            state.fall_fired = False
            return False

        if not state.fall_latched and hits >= self.fall_min_indicators:
            state.fall_latched = True
            # Snapshotted at arm time: by the time the dwell elapses these numbers no longer
            # exist, and `reason` has to explain the alert to an operator and to the LLM.
            state.fall_reason = breakdown

        return state.fall_latched

    def _check_falls(
        self, persons: list[IncidentDetection], timestamp: float, camera_id: str | None
    ) -> list[IncidentDetection]:
        """Fire when a suspected fall persists past the dwell threshold.

        The dwell is what separates a fall from crouching to pick something up, and PRD 3.1 fixes
        it at 3 seconds. The indicators decide *what arms* a fall; they never touch the timer.
        """
        events = []
        for person in persons:
            if not person.track_id:
                continue
            state = self._tracks.get(self._track_key(camera_id, person.track_id))
            if state is None:
                continue

            if not self._evaluate_fall_indicators(person, state, timestamp):
                continue

            if state.fallen_since is None:
                state.fallen_since = timestamp

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
                            f"{person.track_id} collapsed and stayed down {elapsed:.1f}s "
                            f"(threshold {self.fall_dwell_sec}s) — {state.fall_reason}"
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
