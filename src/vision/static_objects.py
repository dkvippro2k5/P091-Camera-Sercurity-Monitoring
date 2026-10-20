"""Background model that finds objects newly left in the scene, whether or not the detector knows them.

On ABODA, AVSS 2007 and CAVIAR (2026-10-20) every abandonment the rule missed was a bag YOLO
never boxed: small, far away, filmed in infrared, or not a COCO class at all (a shopping bag
on a table, a cardboard box). The bag is still plain to see as a change against the empty
scene, so this module keeps a picture of the scene without its passers-by and reports the
places that have differed from it, without a break, for `static_sec`.

The background is updated only where the frame matches it, so a bag - or a person standing
still - is never learnt into it, and a person who stood still for a minute leaves no ghost
behind when they walk on. A region is absorbed after `absorb_sec`, so a permanent change (a
chair moved, a door left open) is not a region forever. A frame in which most of the picture
changed at once is a lighting change, not a hundred objects: the background is relearnt from
it. There is deliberately no hysteresis on "differs": where the background had drifted a
little, a person walking past pushed pixels over the threshold and hysteresis then held them
there, which on AVSS 2007 and the ABODA night lobby made a "region" of the floor behind every
passer-by.

Each region carries `since`, when its pixels started to differ: the moment the object was
put down, which is when the abandoned rule looks for who brought it.

Three things that differ from the background are not left objects, and are dropped:

  - People. Pixels under a person's box, grown by `person_margin` and kept for
    `person_memory_sec` (the detector loses people for a frame or two), are never part of a
    region.
  - Ghosts: something that was in the background picture and has gone (the person in the
    first frame, a train that pulled out). An object put down brings its own outline; a
    ghost's outline is in the background and the frame there is plain floor. A region needs
    `min_new_edge` of its outline to be edges the background did not have.
  - Lighting: a shadow or a change of exposure moves brightness, not outlines, so it fails
    the same test.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.schemas import BoundingBox, IncidentDetection

LABEL_STATIC = "static_object"
# Sobel magnitude (on the blurred 0-255 image) above which a pixel is an edge.
_EDGE_THRESHOLD = 60.0
_CHANNEL_MEAN = np.full((1, 3), 1 / 3, dtype=np.float32)


@dataclass
class StaticRegion:
    """A place that has differed from the background since `since`."""

    detection: IncidentDetection  # label LABEL_STATIC, track_id kept while the region lasts
    since: float


class StaticObjectDetector:
    """Per-camera background model reporting newly static regions.

    Stateful, so one instance per stream, like the motion gate.

    Args:
        static_sec: How long a pixel must differ from the background before it counts.
        absorb_sec: A pixel differing this long is learnt into the background.
        diff_threshold: Per-channel difference (0-255) at which a pixel differs.
        background_rate: Running-average rate for pixels that match the background.
        relearn_fraction: Fraction of differing pixels at which the frame is taken for a
            lighting change and becomes the new background.
        min_area_ratio, max_area_ratio: Region size bounds, as fractions of the frame area.
            The largest bag in the benchmarks (a holdall on a classroom table) is ~1.4%; a
            train at the platform is several times that.
        work_width: Frames are downscaled to this width (never upscaled). CAVIAR's bags are
            ~11x9 px at 384x288, so going much below 384 loses them.
        person_margin: Person boxes are grown by this fraction of their size before being cut
            out: legs, bags in hand and shadows spill outside the box.
        person_memory_sec: How long a person's box keeps cutting out pixels after it was last
            seen there.
        min_new_edge: Fraction of a region's outline that must be new edges (and at least
            twice the fraction of background edges that have gone).
    """

    def __init__(
        self,
        static_sec: float = 8.0,
        absorb_sec: float = 300.0,
        diff_threshold: int = 30,
        background_rate: float = 0.02,
        relearn_fraction: float = 0.4,
        min_area_ratio: float = 0.0004,
        max_area_ratio: float = 0.02,
        work_width: int = 384,
        person_margin: float = 0.25,
        person_memory_sec: float = 2.0,
        min_new_edge: float = 0.4,
    ):
        self.static_sec = static_sec
        self.absorb_sec = absorb_sec
        self.diff_threshold = diff_threshold
        self.background_rate = background_rate
        self.relearn_fraction = relearn_fraction
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.work_width = work_width
        self.person_margin = person_margin
        self.person_memory_sec = person_memory_sec
        self.min_new_edge = min_new_edge
        self.reset()

    def reset(self) -> None:
        self._background: np.ndarray | None = None
        self._since: np.ndarray | None = None  # per pixel: when it started differing, NaN if it matches
        self._people: list[tuple[float, BoundingBox]] = []
        self._regions: dict[str, tuple[BoundingBox, float]] = {}
        self._next_id = 0

    def measure(
        self,
        frame: np.ndarray,
        timestamp: float,
        persons: list[IncidentDetection],
        camera_id: str | None = None,
    ) -> list[StaticRegion]:
        """Update the background with `frame` and return the static regions in it."""
        height, width = frame.shape[:2]
        scale = min(1.0, self.work_width / width)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
        self._people = [(seen, box) for seen, box in self._people if timestamp - seen <= self.person_memory_sec]
        self._people += [(timestamp, p.bbox) for p in persons]
        if self._background is None or self._background.shape != small.shape:
            self._relearn(small)
            return []

        blue, green, red = cv2.split(cv2.absdiff(small, self._background))
        different = cv2.max(cv2.max(blue, green), red) > self.diff_threshold
        if np.count_nonzero(different) > self.relearn_fraction * different.size:
            self._relearn(small)
            return []

        self._since = np.where(different, np.fmin(self._since, timestamp), np.nan)
        cv2.accumulateWeighted(small, self._background, self.background_rate, mask=(~different).astype(np.uint8))
        with np.errstate(invalid="ignore"):
            age = timestamp - self._since  # NaN where the pixel matches
            absorbed = age >= self.absorb_sec
            static = (age >= self.static_sec) & ~absorbed
        if absorbed.any():
            self._background[absorbed] = small[absorbed]
            self._since[absorbed] = np.nan

        mask = static.astype(np.uint8)
        for _, box in self._people:
            mx = (box.xmax - box.xmin) * self.person_margin
            my = (box.ymax - box.ymin) * self.person_margin
            x0, y0 = max(0, int((box.xmin - mx) * scale)), max(0, int((box.ymin - my) * scale))
            x1, y1 = int(math.ceil((box.xmax + mx) * scale)), int(math.ceil((box.ymax + my) * scale))
            mask[y0:y1, x0:x1] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        area = mask.shape[0] * mask.shape[1]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        found = []
        edges = None
        for i in range(1, count):
            x, y, w, h, pixels = stats[i]
            if not self.min_area_ratio * area <= pixels <= self.max_area_ratio * area:
                continue
            if pixels < 0.3 * w * h:
                continue  # a thin sliver along an edge, not an object
            component = labels == i
            if edges is None:
                edges = _edges(small), _edges(self._background)
            if not self._has_own_outline(component, *edges):
                continue
            box = BoundingBox(
                xmin=int(x / scale), ymin=int(y / scale), xmax=int((x + w) / scale), ymax=int((y + h) / scale)
            )
            found.append((box, float(np.nanmedian(self._since[component & static]))))
        return self._assign_ids(found, timestamp, camera_id)

    def background_image(self, width: int, height: int) -> np.ndarray | None:
        """The background picture at `width` x `height`, or None before the first frame.

        At the working resolution, scaled up: enough to show a person what the spot looked
        like before something was left there, not a sharp image.
        """
        if self._background is None:
            return None
        return cv2.resize(self._background.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

    def _relearn(self, small: np.ndarray) -> None:
        self._background = small.copy()
        self._since = np.full(small.shape[:2], np.nan)

    def _has_own_outline(self, component: np.ndarray, frame_edges: np.ndarray, background_edges: np.ndarray) -> bool:
        """Whether the region's outline is mostly edges of the frame the background did not have."""
        kernel = np.ones((3, 3), np.uint8)
        pixels = component.astype(np.uint8)
        ring = (cv2.dilate(pixels, kernel) - cv2.erode(pixels, kernel)).astype(bool)
        here, before = frame_edges[ring], background_edges[ring]
        new = float((here > np.maximum(_EDGE_THRESHOLD, 1.5 * before)).mean())
        gone = float((before > np.maximum(_EDGE_THRESHOLD, 1.5 * here)).mean())
        return new >= self.min_new_edge and new > 2 * gone

    def _assign_ids(
        self, found: list[tuple[BoundingBox, float]], timestamp: float, camera_id: str | None
    ) -> list[StaticRegion]:
        """Keep a region's ID while it stays in place, so the abandoned rule sees one object."""
        previous, self._regions = self._regions, {}
        regions = []
        for box, since in found:
            match = max(previous, key=lambda key: _iou(previous[key][0], box), default=None)
            if match is not None and _iou(previous[match][0], box) >= 0.3:
                region_id = match
                since = min(since, previous.pop(match)[1])
            else:
                self._next_id += 1
                region_id = f"static_{self._next_id}"
            self._regions[region_id] = (box, since)
            detection = IncidentDetection(
                label=LABEL_STATIC,
                track_id=region_id,
                confidence=1.0,
                timestamp=timestamp,
                camera_id=camera_id,
                bbox=box,
            )
            regions.append(StaticRegion(detection, since))
        return regions


def _edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.transform(image, _CHANNEL_MEAN)
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    w = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    h = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if w <= 0 or h <= 0:
        return 0.0
    union = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - w * h
    return w * h / union if union > 0 else 0.0
