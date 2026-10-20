"""Abandoned-object rule: a bag that sits still after the people who brought it have left.

State is kept per bag, not per detector track. On a busy platform one suitcase is a
"suitcase", then a "handbag", then hidden behind a passer-by for ten seconds, and every one of
those restarts its track; keyed on the track, the bag also forgot who its owner was and bound
whoever happened to be nearest next. A bag here is a place where a carriable object sits: a
detection joins the bag whose track it continues, or failing that the bag it lies on, and a bag
is only forgotten after `memory_sec` without being seen.

Owners are everyone within the owner radius while the bag is carried or within
`owner_bind_sec` of it coming to rest - not just the nearest person, which in a crowd is as
often a passer-by as the owner (AVSS 2007 MEDIUM, 2026-10-19: the man beside his suitcase was
passed over for someone who then walked off, and the suitcase was reported four times while
he stood there). The bag is attended while any owner is within the radius, recognised by
track ID, by a new track standing where an owner was a moment ago (the tracker renumbering
someone in place), or, with appearance signatures, by looking like one of them (see
appearance.py). A bag inside a person's box is in their hands, or hidden behind them: it is
not reported on that frame, but its clock is not reset either, since in a crowd people walk
in front of a left bag all the time.

With `require_owner`, a bag nobody was ever seen with is not reported: it was already there
when the camera started, or the detector only picked it up after its owner had gone. On ABODA
the first case is most of the false alarms and the second is rare, since the owner is usually
still beside the bag when it is first detected.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from src.vision.appearance import appearance_similarity
from src.vision.schemas import BoundingBox, IncidentDetection

logger = logging.getLogger(__name__)

LABEL_ABANDONED = "abandoned_object"
# Signatures kept per owner: enough to cover them turning around or being half hidden.
_SAMPLES_PER_OWNER = 8
# A new track overlapping an owner's last box by this much, this soon after it was seen, is
# that owner renumbered. Colour alone misses this: on AVSS 2007 EASY the owner's similarity to
# his own earlier samples swung between 0.47 and 0.65 as he turned on the spot.
_HANDOFF_SEC = 2.0
_HANDOFF_IOU = 0.3


def _center(det: IncidentDetection) -> tuple[float, float]:
    return ((det.bbox.xmin + det.bbox.xmax) / 2.0, (det.bbox.ymin + det.bbox.ymax) / 2.0)


def _diagonal(det: IncidentDetection) -> float:
    return math.hypot(det.bbox.xmax - det.bbox.xmin, det.bbox.ymax - det.bbox.ymin)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    w = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    h = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if w <= 0 or h <= 0:
        return 0.0
    union = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - w * h
    return w * h / union if union > 0 else 0.0


def _inside(point: tuple[float, float], box: BoundingBox) -> bool:
    return box.xmin <= point[0] <= box.xmax and box.ymin <= point[1] <= box.ymax


@dataclass
class Owner:
    """One person a bag belongs to, under one track ID."""

    samples: list[np.ndarray] = field(default_factory=list)  # recent appearance signatures
    box: BoundingBox | None = None  # where they last stood near the bag
    seen: float = -math.inf


@dataclass
class BagState:
    """One carriable object at one spot, across however many detector tracks it takes."""

    anchor: tuple[float, float]
    first_seen: float
    last_seen: float
    stationary_since: float
    track_ids: set[str] = field(default_factory=set)
    owners: dict[str, Owner] = field(default_factory=dict)  # by person track ID
    last_owner_seen: float | None = None
    fired: bool = False
    seen_at: float = -math.inf  # last frame processed, so duplicate boxes count once


class AbandonedObjectTracker:
    """Per-camera bag state and the abandoned-object rule.

    Args:
        abandon_sec: Seconds a bag must sit with no owner near before it is reported.
        owner_radius_ratio: Owner radius as a fraction of frame width (PRD's radius R).
        stationary_tolerance_ratio: A track moving less than this fraction of frame width
            from where it settled counts as still.
        require_owner: Never report a bag that no person was seen with.
        owner_bind_sec: How long after a bag comes to rest people near it become owners.
        owner_match_similarity: Appearance similarity (0-1) at which a person near the bag is
            taken for one of its owners under a new track ID. Only used with signatures.
        memory_sec: Forget a bag not seen for this long.
    """

    def __init__(
        self,
        abandon_sec: float = 2.0,
        owner_radius_ratio: float = 0.25,
        stationary_tolerance_ratio: float = 0.02,
        require_owner: bool = True,
        owner_bind_sec: float = 3.0,
        owner_match_similarity: float = 0.85,
        memory_sec: float = 30.0,
    ):
        self.abandon_sec = abandon_sec
        self.owner_radius_ratio = owner_radius_ratio
        self.stationary_tolerance_ratio = stationary_tolerance_ratio
        self.require_owner = require_owner
        self.owner_bind_sec = owner_bind_sec
        self.owner_match_similarity = owner_match_similarity
        self.memory_sec = memory_sec
        self._bags: dict[str | None, list[BagState]] = {}

    def reset(self) -> None:
        self._bags.clear()

    def bags(self, camera_id: str | None) -> list[BagState]:
        return self._bags.get(camera_id, [])

    def fired_track_ids(self, camera_id: str | None) -> set[str]:
        return {track_id for bag in self.bags(camera_id) if bag.fired for track_id in bag.track_ids}

    def persons_near_bags(
        self, objects: list[IncidentDetection], persons: list[IncidentDetection], frame_width: int
    ) -> set[str]:
        """Persons within the owner radius of a carriable object: the only ones worth describing."""
        radius = frame_width * self.owner_radius_ratio
        centers = [_center(obj) for obj in objects]
        return {
            p.track_id
            for p in persons
            if p.track_id and any(math.dist(_center(p), c) <= radius for c in centers)
        }

    def rename_person(self, camera_id: str | None, old_track_id: str, new_track_id: str) -> None:
        """A person track was stitched to a new ID: the new ID is an owner wherever the old one was.

        The old ID stays an owner too. A stitch is a guess by position, and in a crowd the
        tracker sometimes recovers the original ID a moment later.
        """
        for bag in self.bags(camera_id):
            if old_track_id in bag.owners and new_track_id not in bag.owners:
                old = bag.owners[old_track_id]
                bag.owners[new_track_id] = Owner(list(old.samples), old.box, old.seen)

    def update(
        self,
        objects: list[IncidentDetection],
        persons: list[IncidentDetection],
        frame_width: int,
        timestamp: float,
        camera_id: str | None,
        appearance: dict[str, np.ndarray] | None = None,
    ) -> list[IncidentDetection]:
        appearance = appearance or {}
        radius = frame_width * self.owner_radius_ratio
        tolerance = frame_width * self.stationary_tolerance_ratio
        bags = self._bags.setdefault(camera_id, [])
        events = []

        for obj in objects:
            if not obj.track_id:
                continue
            center = _center(obj)
            bag, moved = self._bag_for(bags, obj, center, tolerance, timestamp)
            if bag.seen_at == timestamp:
                continue  # a second box on the same bag this frame
            bag.seen_at = timestamp

            near = [(p, math.dist(_center(p), center)) for p in persons if p.track_id]
            near = [p for p, d in sorted(near, key=lambda item: item[1]) if d <= radius]
            if moved or timestamp - bag.stationary_since <= self.owner_bind_sec:
                for person in near:
                    bag.owners.setdefault(person.track_id, Owner())
            if self._owner_present(bag, near, appearance, timestamp, {p.track_id for p in persons}):
                bag.last_owner_seen = timestamp
                bag.fired = False
                continue
            if not bag.owners:
                if self.require_owner:
                    continue
                if near:
                    # Without an owner to look for, anyone beside the bag attends it.
                    bag.last_owner_seen = timestamp
                    bag.fired = False
                    continue

            if any(_inside(center, p.bbox) for p in persons):
                continue  # held, or hidden behind someone
            anchors = [v for v in (bag.last_owner_seen, bag.stationary_since) if v is not None]
            elapsed = timestamp - max(anchors)
            if elapsed >= self.abandon_sec and not bag.fired:
                bag.fired = True
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
                            f"with none of its {len(bag.owners)} owner(s) within {radius:.0f}px"
                        ),
                    )
                )

        self._bags[camera_id] = [bag for bag in bags if timestamp - bag.last_seen <= self.memory_sec]
        return events

    def _bag_for(
        self,
        bags: list[BagState],
        obj: IncidentDetection,
        center: tuple[float, float],
        tolerance: float,
        timestamp: float,
    ) -> tuple[BagState, bool]:
        """The bag this detection belongs to, creating one if needed, and whether it moved.

        Only a track that carries on from the bag can move it: that is the bag being carried.
        A new track on the same spot just joins - detector jitter or a relabel, not movement.
        """
        for bag in bags:
            if obj.track_id in bag.track_ids:
                bag.last_seen = timestamp
                if math.dist(bag.anchor, center) > tolerance:
                    bag.anchor = center
                    bag.stationary_since = timestamp
                    bag.fired = False
                    return bag, True
                return bag, False
        join_radius = max(tolerance, 0.5 * _diagonal(obj))
        here = [bag for bag in bags if math.dist(bag.anchor, center) <= join_radius]
        if here:
            bag = min(here, key=lambda b: math.dist(b.anchor, center))
            bag.track_ids.add(obj.track_id)
            bag.last_seen = timestamp
            return bag, False
        bag = BagState(anchor=center, first_seen=timestamp, last_seen=timestamp, stationary_since=timestamp)
        bag.track_ids.add(obj.track_id)
        bags.append(bag)
        return bag, False

    def _owner_present(
        self,
        bag: BagState,
        near: list[IncidentDetection],
        appearance: dict[str, np.ndarray],
        timestamp: float,
        in_frame: set[str],
    ) -> bool:
        """Whether an owner is within the radius: by track ID, by taking over an owner's spot,
        or by looking like one."""
        present = False
        for person in near:
            signature = appearance.get(person.track_id)
            if person.track_id in bag.owners:
                pass
            elif self._took_over(bag, person, timestamp, in_frame) or (
                signature is not None and self._looks_like_owner(bag, signature)
            ):
                logger.debug(f"{person.track_id} recognised as an owner of a bag at {bag.anchor}")
                bag.owners[person.track_id] = Owner()
            else:
                continue
            present = True
            owner = bag.owners[person.track_id]
            owner.box, owner.seen = person.bbox, timestamp
            if signature is not None:
                owner.samples.append(signature)
                del owner.samples[:-_SAMPLES_PER_OWNER]
        return present

    @staticmethod
    def _took_over(bag: BagState, person: IncidentDetection, timestamp: float, in_frame: set[str]) -> bool:
        # An owner still in the frame was not renumbered, whoever is standing next to them.
        return any(
            track_id not in in_frame
            and owner.box is not None
            and 0 < timestamp - owner.seen <= _HANDOFF_SEC
            and _iou(owner.box, person.bbox) >= _HANDOFF_IOU
            for track_id, owner in bag.owners.items()
        )

    def _looks_like_owner(self, bag: BagState, signature: np.ndarray) -> bool:
        return any(
            appearance_similarity(signature, sample) >= self.owner_match_similarity
            for owner in bag.owners.values()
            for sample in owner.samples
        )
