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
from statistics import median
from typing import Literal

from src.vision.fall_motion import FallMotionEvidence
from src.vision.schemas import BoundingBox, IncidentDetection, Zone
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

_UPRIGHT_BASELINE_WINDOW_SEC = 3.0
_FALL_TRANSITION_TIMEOUT_SEC = 1.0

FallPhase = Literal["upright", "descending", "prone", "persistent", "recovered"]


@dataclass(frozen=True)
class UprightBaseline:
    """Median geometry for one person while confidently upright."""

    top: float
    height: float
    center_y: float
    width: float
    area: float


@dataclass(frozen=True)
class FallPerspectiveProfile:
    """Thresholds that may vary with one camera's viewing perspective."""

    center_velocity_ratio: float = 0.5
    area_growth_ratio: float = 0.5
    area_acceleration_ratio: float = 2.0
    foot_stability_ratio: float = 0.18
    upper_drop_ratio: float = 0.25
    optical_flow_ratio: float = 0.25
    confirmation_sec: float = 0.4
    persistence_grace_sec: float = 0.6


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
    fall_impact_bbox: BoundingBox | None = None
    fall_phase: FallPhase = "upright"
    fall_transition_since: float | None = None
    prone_candidate_since: float | None = None
    nonprone_since: float | None = None
    # First frame of an unbroken run that looks like standing back up; see
    # _evaluate_fall_indicators for why one such frame is not enough.
    recovery_candidate_since: float | None = None
    upright_samples: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)
    upright_baseline: UprightBaseline | None = None
    # Abandoned: when the object was last near a person, and where it settled.
    last_owner_seen: float | None = None
    stationary_since: float | None = None
    anchor: tuple[float, float] | None = None
    abandoned_fired: bool = False
    owner_track_id: str | None = None
    # Track stitching & fall indicators: (timestamp, y_top, height) per analyzed frame.
    # The centroid this used to carry was dead weight — stitching matches on `last_bbox`, not on
    # this history, and the fall rule only ever needs the box's top edge and its height.
    pos_history: list[tuple[float, float, float, float, float]] = field(default_factory=list)
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
        # Forward falls can increase box area while the top edge moves less than a
        # side-on fall, so these defaults give the depth-sensitive signals more weight.
        fall_velocity_ratio: float = 0.9,
        fall_height_drop_ratio: float = 0.35,
        fall_center_velocity_ratio: float = 0.35,
        fall_area_growth_ratio: float = 0.30,
        fall_area_acceleration_ratio: float = 1.2,
        fall_foot_stability_ratio: float = 0.25,
        fall_upper_drop_ratio: float = 0.18,
        fall_optical_flow_ratio: float = 0.15,
        fall_confirmation_sec: float = 0.4,
        fall_persistence_grace_sec: float = 0.6,
        fall_recovery_ratio: float = 0.3,
        fall_min_indicators: int = 2,
        abandon_sec: float = 2.0,
        owner_radius_ratio: float = 0.25,
        stationary_tolerance_ratio: float = 0.02,
        track_ttl_sec: float = 1.0,
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
        self.fall_center_velocity_ratio = fall_center_velocity_ratio
        self.fall_area_growth_ratio = fall_area_growth_ratio
        self.fall_area_acceleration_ratio = fall_area_acceleration_ratio
        self.fall_foot_stability_ratio = fall_foot_stability_ratio
        self.fall_upper_drop_ratio = fall_upper_drop_ratio
        self.fall_optical_flow_ratio = fall_optical_flow_ratio
        self.fall_confirmation_sec = fall_confirmation_sec
        self.fall_persistence_grace_sec = fall_persistence_grace_sec
        self.fall_recovery_ratio = fall_recovery_ratio
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
        self._last_alert: dict[tuple[str | None, str, str | None, str | None], float] = {}
        self._camera_fall_profiles: dict[str, FallPerspectiveProfile] = {}

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

    def note_external_event(
        self,
        label: str,
        timestamp: float,
        camera_id: str | None = None,
        track_id: str | None = None,
        zone_name: str | None = None,
    ) -> None:
        """Reserve the shared cooldown for an event emitted by a secondary detector."""
        if self.alert_cooldown_sec > 0:
            self._last_alert[(camera_id, label, track_id, zone_name)] = timestamp

    def set_camera_fall_profile(
        self,
        camera_id: str,
        profile: FallPerspectiveProfile | None,
    ) -> None:
        """Install or remove perspective thresholds for one camera."""
        if profile is None:
            self._camera_fall_profiles.pop(camera_id, None)
        else:
            self._camera_fall_profiles[camera_id] = profile

    def _fall_profile(self, camera_id: str | None) -> FallPerspectiveProfile:
        if camera_id is not None and camera_id in self._camera_fall_profiles:
            return self._camera_fall_profiles[camera_id]
        return FallPerspectiveProfile(
            center_velocity_ratio=self.fall_center_velocity_ratio,
            area_growth_ratio=self.fall_area_growth_ratio,
            area_acceleration_ratio=self.fall_area_acceleration_ratio,
            foot_stability_ratio=self.fall_foot_stability_ratio,
            upper_drop_ratio=self.fall_upper_drop_ratio,
            optical_flow_ratio=self.fall_optical_flow_ratio,
            confirmation_sec=self.fall_confirmation_sec,
            persistence_grace_sec=self.fall_persistence_grace_sec,
        )

    def optical_flow_candidates(
        self,
        persons: list[IncidentDetection],
        camera_id: str | None = None,
    ) -> set[str]:
        """Select falling-looking tracks before paying for sparse optical flow."""
        candidates: set[str] = set()
        profile = self._fall_profile(camera_id)
        for person in persons:
            if not person.track_id:
                continue
            state = self._tracks.get(self._track_key(camera_id, person.track_id))
            if state is None:
                continue
            if state.fall_phase != "upright":
                candidates.add(person.track_id)
                continue
            baseline = state.upright_baseline
            if baseline is None:
                continue
            width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
            height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
            center_y = (person.bbox.ymin + person.bbox.ymax) / 2.0
            top_drop = (person.bbox.ymin - baseline.top) / baseline.height
            center_drop = (center_y - baseline.center_y) / baseline.height
            height_drop = (baseline.height - height) / baseline.height
            area_growth = (width * height - baseline.area) / baseline.area
            if (
                top_drop >= profile.upper_drop_ratio * 0.5
                or center_drop >= profile.upper_drop_ratio * 0.5
                or height_drop >= self.fall_height_drop_ratio * 0.5
                or area_growth >= profile.area_growth_ratio * 0.5
                or width / height >= self.fall_aspect_ratio
            ):
                candidates.add(person.track_id)
        return candidates

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
        fall_motion_evidence: dict[str, FallMotionEvidence] | None = None,
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
        events.extend(
            self._check_falls(
                persons,
                timestamp,
                camera_id,
                fall_motion_evidence or {},
            )
        )
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
            key = (camera_id, event.label, event.track_id, event.zone_name)
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
        current_track_ids = {det.track_id for det in detections if det.track_id}
        for det in detections:
            if not det.track_id:
                continue
            key = self._track_key(camera_id, det.track_id)
            state = self._tracks.get(key)
            cx = (det.bbox.xmin + det.bbox.xmax) / 2.0
            cy = (det.bbox.ymin + det.bbox.ymax) / 2.0

            if state is None:
                state = self._stitch_or_create(
                    det,
                    key,
                    camera_id,
                    timestamp,
                    cx,
                    cy,
                    frame_width,
                    current_track_ids,
                )
            else:
                state.last_seen = timestamp
                state.last_bbox = det.bbox

            # Rolling window feeding the fall rule's V_y and ΔH indicators. Pruned to the
            # same span they are computed over, so the oldest sample is the window edge.
            width = max(1, det.bbox.xmax - det.bbox.xmin)
            height = max(1, det.bbox.ymax - det.bbox.ymin)
            center_y = (det.bbox.ymin + det.bbox.ymax) / 2.0
            state.pos_history.append(
                (timestamp, float(det.bbox.ymin), float(height), float(center_y), float(width * height))
            )
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
        current_track_ids: set[str],
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
        best_source: Literal["active", "expired"] | None = None
        best_dist = math.inf
        sources = (("active", self._tracks), ("expired", self._expired_tracks))
        new_area = max(1, det.bbox.xmax - det.bbox.xmin) * max(1, det.bbox.ymax - det.bbox.ymin)
        for source_name, source in sources:
            for candidate_key, candidate_state in list(source.items()):
                if candidate_key[0] != camera_id or candidate_state.label != det.label:
                    continue
                if candidate_state.track_id in current_track_ids:
                    continue
                gap = timestamp - candidate_state.last_seen
                if gap <= 0 or gap > self.track_stitch_window_sec:
                    continue
                old_area = max(1, candidate_state.last_bbox.xmax - candidate_state.last_bbox.xmin) * max(
                    1, candidate_state.last_bbox.ymax - candidate_state.last_bbox.ymin
                )
                area_ratio = new_area / old_area
                if area_ratio < 0.2 or area_ratio > 5.0:
                    continue
                old_cx = (candidate_state.last_bbox.xmin + candidate_state.last_bbox.xmax) / 2.0
                old_cy = (candidate_state.last_bbox.ymin + candidate_state.last_bbox.ymax) / 2.0
                dist = math.hypot(cx - old_cx, cy - old_cy)
                limit = radius
                if candidate_state.fall_phase in ("descending", "prone", "persistent"):
                    # A body going down moves its box centre by up to its own height — a
                    # standing person's centre is half a body above the floor, a lying
                    # person's is on it. The width-based radius (~38 px at 320 wide) is far
                    # smaller than that, so a tracker ID change at impact used to start a
                    # fresh "upright" track that could never arm, losing the fall entirely.
                    limit = max(radius, float(candidate_state.last_bbox.ymax - candidate_state.last_bbox.ymin))
                if dist < limit and dist < best_dist:
                    best_key, best_dist = candidate_key, dist
                    best_source = source_name

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

        if best_source == "active":
            stitched = self._tracks.pop(best_key)
        else:
            stitched = self._expired_tracks.pop(best_key)
        old_track_id = stitched.track_id
        stitched.aliased_from = stitched.track_id
        stitched.track_id = det.track_id
        stitched.last_seen = timestamp
        stitched.last_bbox = det.bbox
        # The dwell timers are the whole point of stitching, so fallen_since / stationary_since
        # / last_owner_seen ride along. A fired fall also remains fired until an observed
        # recovery; otherwise every brief ID change while someone is still prone emits a
        # duplicate incident. Abandoned-object state is also preserved until movement or the
        # owner returning explicitly re-arms it.
        # Position history is cleared because a stale sample would fake a velocity spike on the
        # first frame after the stitch.
        stitched.pos_history = []
        self._tracks[key] = stitched
        if stitched.label in PERSON_CLASSES:
            self._replace_owner_track_id(camera_id, old_track_id, det.track_id)
        logger.debug(
            f"Stitched {det.label} {stitched.aliased_from} -> {det.track_id} on {camera_id} "
            f"({best_dist:.0f}px, radius {radius:.0f}px)"
        )
        return stitched

    def _replace_owner_track_id(
        self,
        camera_id: str | None,
        old_track_id: str,
        new_track_id: str,
    ) -> None:
        """Keep bag-owner associations valid when the person tracker changes ID."""
        for key, state in self._tracks.items():
            if key[0] == camera_id and state.owner_track_id == old_track_id:
                state.owner_track_id = new_track_id
        for key, state in self._expired_tracks.items():
            if key[0] == camera_id and state.owner_track_id == old_track_id:
                state.owner_track_id = new_track_id

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
            used_lower_body_overlap = False
            if not hits:
                # A partially occluded person or umbrella can make the foot point unreliable.
                # Sample the lower body instead of using any-box overlap, which would alert
                # when a shoulder merely clips the zone boundary. Two of eight samples is a
                # 25% lower-body overlap approximation.
                width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
                height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
                sample_points = [
                    (person.bbox.xmin + width * x_ratio, person.bbox.ymin + height * y_ratio)
                    for y_ratio in (0.72, 0.92)
                    for x_ratio in (0.2, 0.4, 0.6, 0.8)
                ]
                sample_counts: dict[str, tuple[Zone, int]] = {}
                for sample_x, sample_y in sample_points:
                    for zone in self.zone_manager.zones_containing(
                        camera_id, sample_x, sample_y, frame_width, frame_height, now
                    ):
                        existing = sample_counts.get(zone.name)
                        sample_counts[zone.name] = (zone, (existing[1] if existing else 0) + 1)
                hits = [zone for zone, count in sample_counts.values() if count >= 2]
                used_lower_body_overlap = bool(hits)
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
                            f"{person.track_id} entered restricted zone '{zone.name}' during {window}; "
                            + (
                                "lower-body overlap confirmed by at least 2/8 samples"
                                if used_lower_body_overlap
                                else f"foot point=({foot_x / frame_width:.3f}, "
                                f"{foot_y / frame_height:.3f}) normalized"
                            )
                        ),
                    )
                )

            # Leaving the zone re-arms it, so a later re-entry alerts again.
            state.alerted_zones &= current

        return events

    def _fall_indicators(
        self,
        person: IncidentDetection,
        state: TrackState,
        timestamp: float,
        camera_id: str | None = None,
        motion_evidence: FallMotionEvidence | None = None,
    ) -> tuple[int, str]:
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
        profile = self._fall_profile(camera_id)

        current_center_y = (person.bbox.ymin + person.bbox.ymax) / 2.0
        current_area = max(1.0, float(width * height))
        oldest = (
            state.pos_history[0]
            if state.pos_history
            else (timestamp, float(person.bbox.ymin), float(height), current_center_y, current_area)
        )
        t0, y0, h0, center_y0, area0 = oldest
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
        center_v_str = "V_center=n/a"
        vel_hit = False
        center_vel_hit = False
        if dt >= _MIN_VELOCITY_DT_SEC:
            v_val = max(0.0, (float(person.bbox.ymin) - y0) / dt) / h_ref
            vel_hit = v_val >= self.fall_velocity_ratio
            v_str = f"V_y={v_val:.2f}H/s"
            center_v_val = max(0.0, (current_center_y - center_y0) / dt) / h_ref
            center_vel_hit = center_v_val >= profile.center_velocity_ratio
            center_v_str = f"V_center={center_v_val:.2f}H/s"

        dh_val = max(0.0, (h0 - height) / h_ref) if dt > 0 else 0.0
        dh_hit = dh_val >= self.fall_height_drop_ratio
        area_growth = max(0.0, (current_area - area0) / max(1.0, area0)) if dt > 0 else 0.0
        area_hit = area_growth >= profile.area_growth_ratio

        area_acceleration = 0.0
        if len(state.pos_history) >= 3:
            tm, _ym, _hm, _cm, area_m = state.pos_history[len(state.pos_history) // 2]
            before_dt = tm - t0
            after_dt = timestamp - tm
            if before_dt >= _MIN_VELOCITY_DT_SEC and after_dt >= _MIN_VELOCITY_DT_SEC:
                before_rate = (area_m - area0) / max(1.0, area0) / before_dt
                after_rate = (current_area - area_m) / max(1.0, area0) / after_dt
                area_acceleration = max(0.0, (after_rate - before_rate) / (before_dt + after_dt))
        area_accel_hit = area_acceleration >= profile.area_acceleration_ratio

        foot_stable_hit = False
        upper_drop = 0.0
        foot_error = float("inf")
        if state.upright_baseline is not None:
            baseline = state.upright_baseline
            upper_drop = max(0.0, (person.bbox.ymin - baseline.top) / baseline.height)
            baseline_bottom = baseline.top + baseline.height
            foot_error = abs(person.bbox.ymax - baseline_bottom) / baseline.height
            foot_stable_hit = (
                upper_drop >= profile.upper_drop_ratio
                and foot_error <= profile.foot_stability_ratio
            )

        flow_ratio = motion_evidence.downward_flow_ratio if motion_evidence else 0.0
        flow_hit = motion_evidence is not None and flow_ratio >= profile.optical_flow_ratio
        motion_hit = vel_hit or (center_vel_hit and area_hit) or foot_stable_hit or flow_hit
        shape_hit = dh_hit or area_hit or area_accel_hit
        ar_hit = ar >= self.fall_aspect_ratio

        breakdown = (
            f"{'+' if motion_hit else '-'}motion[{'+' if vel_hit else '-'}{v_str}"
            f"(>={self.fall_velocity_ratio}), {'+' if center_vel_hit else '-'}{center_v_str}"
            f"(>={profile.center_velocity_ratio}), "
            f"{'+' if foot_stable_hit else '-'}foot(drop={upper_drop:.2f}H,err={foot_error:.2f}H), "
            f"{'+' if flow_hit else '-'}flow={flow_ratio:.2f}H/s(>={profile.optical_flow_ratio})] "
            f"{'+' if ar_hit else '-'}AR={ar:.2f}(>={self.fall_aspect_ratio}) "
            f"{'+' if dh_hit else '-'}dH={int(dh_val * 100)}%(>={int(self.fall_height_drop_ratio * 100)}%) "
            f"{'+' if area_hit else '-'}dA={int(area_growth * 100)}%(>={int(profile.area_growth_ratio * 100)}%) "
            f"{'+' if area_accel_hit else '-'}aA={area_acceleration:.2f}/s2(>={profile.area_acceleration_ratio})"
        )

        # Measured before the upright test so a standing person's breakdown still carries the
        # numbers rather than a bare "upright". The pair is what makes a *missed* fall
        # diagnosable from the log alone, which is otherwise the one failure leaving no trace.
        hits = sum((motion_hit, ar_hit, shape_hit))
        if ar < self.fall_upright_aspect_ratio and hits < self.fall_min_indicators:
            return 0, f"{_UPRIGHT_PREFIX} (AR {ar:.2f}) [{breakdown}]"

        return hits, f"{hits}/3 indicators [{breakdown}]"

    def _is_confidently_upright(self, person: IncidentDetection) -> bool:
        width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
        height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
        return width / height < self.fall_upright_aspect_ratio

    def _record_upright_baseline(
        self, person: IncidentDetection, state: TrackState, timestamp: float
    ) -> None:
        """Update the person's standing reference without learning crouched poses."""
        width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
        height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
        top = float(person.bbox.ymin)
        center_y = (person.bbox.ymin + person.bbox.ymax) / 2.0
        area = width * height
        state.upright_samples.append((timestamp, top, height, center_y, width, area))
        state.upright_samples = [
            sample
            for sample in state.upright_samples
            if timestamp - sample[0] <= _UPRIGHT_BASELINE_WINDOW_SEC
        ]
        state.upright_baseline = UprightBaseline(
            top=median(sample[1] for sample in state.upright_samples),
            height=median(sample[2] for sample in state.upright_samples),
            center_y=median(sample[3] for sample in state.upright_samples),
            width=median(sample[4] for sample in state.upright_samples),
            area=median(sample[5] for sample in state.upright_samples),
        )

    def _is_prone_relative_to_baseline(
        self,
        person: IncidentDetection,
        state: TrackState,
        camera_id: str | None = None,
    ) -> bool:
        """Whether the post-transition box still looks down relative to standing."""
        baseline = state.upright_baseline
        if baseline is None:
            return False
        profile = self._fall_profile(camera_id)

        width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
        height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
        center_y = (person.bbox.ymin + person.bbox.ymax) / 2.0
        aspect_ratio = width / height
        height_drop = max(0.0, (baseline.height - height) / baseline.height)
        center_drop = max(0.0, (center_y - baseline.center_y) / baseline.height)
        area_growth = max(0.0, (width * height - baseline.area) / baseline.area)

        # A narrow box that has not moved downward is more likely a partial detection or
        # tracker handoff than a prone person. Keep narrow forward falls eligible when the
        # center really descends, but do not let height loss alone validate a tiny box.
        if aspect_ratio < self.fall_upright_aspect_ratio and center_drop < profile.upper_drop_ratio:
            return False

        return (
            aspect_ratio >= self.fall_aspect_ratio
            or height_drop >= self.fall_height_drop_ratio
            or (center_drop >= profile.upper_drop_ratio and area_growth >= profile.area_growth_ratio)
        )

    @staticmethod
    def _clear_fall_transition(state: TrackState, phase: FallPhase = "upright") -> None:
        state.fall_phase = phase
        state.fall_transition_since = None
        state.prone_candidate_since = None
        state.nonprone_since = None
        state.recovery_candidate_since = None
        state.fallen_since = None
        state.fall_fired = False
        state.fall_latched = False
        state.fall_reason = ""
        state.fall_impact_bbox = None

    def _is_recovery_transition(self, person: IncidentDetection, state: TrackState) -> bool:
        """Require upright shape, scale, and position before clearing a fall."""
        baseline = state.upright_baseline
        if baseline is None or not self._is_confidently_upright(person):
            return False

        height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
        top = float(person.bbox.ymin)
        center_y = (person.bbox.ymin + person.bbox.ymax) / 2.0
        reference = max(1.0, baseline.height)
        top_error = abs(top - baseline.top) / reference
        center_error = abs(center_y - baseline.center_y) / reference
        height_error = abs(height - baseline.height) / reference
        position_error = min(top_error, center_error)
        return (
            height_error <= self.fall_recovery_ratio
            and position_error <= self.fall_recovery_ratio
        )

    def _evaluate_fall_indicators(
        self,
        person: IncidentDetection,
        state: TrackState,
        timestamp: float,
        camera_id: str | None = None,
        motion_evidence: FallMotionEvidence | None = None,
    ) -> bool:
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
        hits, breakdown = self._fall_indicators(
            person,
            state,
            timestamp,
            camera_id,
            motion_evidence,
        )
        profile = self._fall_profile(camera_id)
        # A zero dwell is used by the indicator-only tests to inspect whether a single
        # transition arms the rule. Production paths have a positive dwell and therefore
        # receive the extra prone confirmation interval.
        confirmation_sec = profile.confirmation_sec if self.fall_dwell_sec > 0 else 0.0
        upright = self._is_confidently_upright(person) and hits < self.fall_min_indicators

        if state.fall_phase == "recovered":
            if upright:
                self._clear_fall_transition(state)
                self._record_upright_baseline(person, state, timestamp)
            return False

        if state.fall_phase != "upright":
            # Recovery must hold, not just appear. The indicators arm on the first frame of a
            # descent, while the box is still tall and narrow — so the very next frame used to
            # pass this check and latch "recovered" mid-fall. "Recovered" only clears on a calm
            # upright frame, which a fall in progress never produces, so the real landing a
            # moment later could not re-arm (UR Fall fall-01).
            if self._is_recovery_transition(person, state):
                if state.recovery_candidate_since is None:
                    state.recovery_candidate_since = timestamp
                # Epsilon: frame timestamps are float multiples of 1/fps, so three 0.2 s steps
                # come to 0.5999... and would otherwise need a fourth frame.
                if timestamp - state.recovery_candidate_since >= profile.persistence_grace_sec - 1e-6:
                    self._clear_fall_transition(state, phase="recovered")
                    return False
            else:
                state.recovery_candidate_since = None

        if state.fall_phase == "upright":
            if upright:
                self._record_upright_baseline(person, state, timestamp)
            if hits < self.fall_min_indicators or state.upright_baseline is None:
                return False

            state.fall_phase = "descending"
            state.fall_transition_since = timestamp
            state.prone_candidate_since = None
            state.fall_latched = True
            state.fall_impact_bbox = person.bbox
            baseline = state.upright_baseline
            state.fall_reason = (
                f"state upright->descending; baseline H={baseline.height:.0f}px "
                f"AR={baseline.width / baseline.height:.2f}; {breakdown}"
            )

        if state.fall_phase == "descending":
            prone_now = self._is_prone_relative_to_baseline(person, state, camera_id)
            # Once a down transition has been observed, the 0.7-1.2 aspect band is
            # intentionally neutral: tracker jitter through that band must not cancel the
            # confirmation interval before the person is clearly upright again.
            if not prone_now and state.prone_candidate_since is not None:
                width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
                height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
                prone_now = width / height >= self.fall_upright_aspect_ratio
            if prone_now:
                if state.prone_candidate_since is None:
                    state.prone_candidate_since = timestamp
                if timestamp - state.prone_candidate_since >= confirmation_sec:
                    state.fall_phase = "prone"
                    state.fallen_since = state.prone_candidate_since
            else:
                state.prone_candidate_since = None

            if state.fall_phase == "descending" and (
                state.fall_transition_since is not None
                and timestamp - state.fall_transition_since
                > max(_FALL_TRANSITION_TIMEOUT_SEC, confirmation_sec * 2.0)
            ):
                self._clear_fall_transition(state)
                return False

        if state.fall_phase == "prone":
            prone_now = self._is_prone_relative_to_baseline(person, state, camera_id)
            if not prone_now:
                baseline = state.upright_baseline
                width = max(1.0, float(person.bbox.xmax - person.bbox.xmin))
                height = max(1.0, float(person.bbox.ymax - person.bbox.ymin))
                center_y = (person.bbox.ymin + person.bbox.ymax) / 2.0
                prone_now = bool(
                    baseline
                    and width / height >= self.fall_upright_aspect_ratio
                    and (
                        (baseline.height - height) / baseline.height >= 0.2
                        or (center_y - baseline.center_y) / baseline.height
                        >= profile.upper_drop_ratio * 0.5
                    )
                )
            if prone_now:
                state.nonprone_since = None
            else:
                if state.nonprone_since is None:
                    state.nonprone_since = timestamp
                # Do not let the dwell mature while the track is no longer down. If the
                # prone shape returns inside the grace window, confirmation starts again
                # from the last non-prone frame instead of firing immediately on stale time.
                state.fallen_since = timestamp
                if timestamp - state.nonprone_since >= profile.persistence_grace_sec:
                    self._clear_fall_transition(state)
                return False

        return state.fall_phase in {"prone", "persistent"}

    def _check_falls(
        self,
        persons: list[IncidentDetection],
        timestamp: float,
        camera_id: str | None,
        motion_evidence: dict[str, FallMotionEvidence],
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

            if not self._evaluate_fall_indicators(
                person,
                state,
                timestamp,
                camera_id,
                motion_evidence.get(person.track_id),
            ):
                continue

            if state.fallen_since is None:
                state.fallen_since = timestamp

            elapsed = timestamp - state.fallen_since
            if elapsed >= self.fall_dwell_sec and not state.fall_fired:
                state.fall_fired = True
                state.fall_phase = "persistent"
                events.append(
                    IncidentDetection(
                        label=LABEL_FALL,
                        track_id=person.track_id,
                        confidence=person.confidence,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        bbox=person.bbox,
                        reason=(
                            f"{person.track_id} transitioned upright->prone and stayed down {elapsed:.1f}s "
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

            if state.anchor is None:
                state.anchor = (cx, cy)
                state.stationary_since = timestamp
            elif math.dist(state.anchor, (cx, cy)) > frame_width * self.stationary_tolerance_ratio:
                # A bag that is still being carried is not abandoned. Movement starts a new
                # stationary interval and also permits a future alert after it is put down.
                state.anchor = (cx, cy)
                state.stationary_since = timestamp
                state.abandoned_fired = False

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
                    state.owner_track_id = min(
                        close_owners,
                        key=lambda p: math.dist(
                            ((p.bbox.xmin + p.bbox.xmax) / 2.0, (p.bbox.ymin + p.bbox.ymax) / 2.0),
                            (cx, cy),
                        ),
                    ).track_id

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
            anchors = [value for value in (state.last_owner_seen, state.stationary_since) if value is not None]
            unattended_since = max(anchors, default=timestamp)
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
