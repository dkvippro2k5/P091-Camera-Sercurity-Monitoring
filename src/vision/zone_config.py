"""Polygon zone configuration: loading, time windows, and point-in-polygon tests.

Zones are stored in normalized [0, 1] coordinates so one config survives resolution
changes. Pixel conversion happens only at query and draw time.
"""

import json
import logging
import os
from datetime import datetime, time
from functools import lru_cache

from src.vision.schemas import CameraZoneConfig, Zone

logger = logging.getLogger(__name__)

# Applied to any camera without its own entry, so a new camera is covered by default.
WILDCARD_CAMERA = "*"


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test. Coordinates must share the same space.

    Args:
        x: X coordinate of the query point.
        y: Y coordinate of the query point.
        polygon: Vertices as (x, y) pairs.

    Returns:
        True when the point lies inside the polygon.
    """
    if len(polygon) < 3:
        return False

    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Does the horizontal ray from (x, y) cross the edge (i, j)?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@lru_cache(maxsize=256)
def parse_active_hours(active_hours: str | None) -> tuple[time, time] | None:
    """Parse an 'HH:MM-HH:MM' window into start/end times. Returns None if always active.

    Cached: this runs per zone per frame, and without it a malformed window would emit a
    warning on every frame of every stream.
    """
    if not active_hours:
        return None
    try:
        start_str, end_str = active_hours.split("-")
        start_h, start_m = (int(part) for part in start_str.strip().split(":"))
        end_h, end_m = (int(part) for part in end_str.strip().split(":"))
        return time(start_h, start_m), time(end_h, end_m)
    except (ValueError, TypeError):
        logger.warning(f"Invalid active_hours '{active_hours}'. Treating zone as always active.")
        return None


def is_zone_active(zone: Zone, now: datetime) -> bool:
    """Whether a zone's forbidden window covers `now`. Windows may wrap past midnight."""
    window = parse_active_hours(zone.active_hours)
    if window is None:
        return True

    start, end = window
    current = now.time()
    if start <= end:
        return start <= current <= end
    # Wraps midnight, e.g. 22:00-05:00
    return current >= start or current <= end


class ZoneManager:
    """Resolves which polygon zones a point falls inside, per camera and per time of day."""

    def __init__(self, configs: list[CameraZoneConfig] | None = None):
        self._by_camera: dict[str, list[Zone]] = {}
        for config in configs or []:
            self._by_camera[config.camera_id] = list(config.zones)

    @classmethod
    def from_file(cls, path: str | None) -> "ZoneManager":
        """Load zones from a JSON file. A missing file yields an empty manager.

        Expected shape (either form works)::

            {"cam_hallway_01": [{"name": "...", "polygon": [[0.1, 0.2], ...]}]}
            {"cameras": [{"camera_id": "cam_hallway_01", "zones": [...]}]}
        """
        if not path or not os.path.exists(path):
            if path:
                logger.info(f"Zone config '{path}' not found. Intrusion rule stays idle until zones exist.")
            return cls()

        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read zone config '{path}': {e}. Continuing with no zones.")
            return cls()

        configs: list[CameraZoneConfig] = []
        try:
            if isinstance(raw, dict) and "cameras" in raw:
                configs = [CameraZoneConfig(**entry) for entry in raw["cameras"]]
            elif isinstance(raw, dict):
                configs = [
                    CameraZoneConfig(camera_id=camera_id, zones=[Zone(**z) for z in zones])
                    for camera_id, zones in raw.items()
                ]
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid zone config in '{path}': {e}. Continuing with no zones.")
            return cls()

        manager = cls(configs)
        logger.info(f"Loaded {manager.zone_count} zone(s) for {len(configs)} camera(s) from '{path}'")
        return manager

    @property
    def zone_count(self) -> int:
        return sum(len(zones) for zones in self._by_camera.values())

    def zones_for(self, camera_id: str | None) -> list[Zone]:
        """Zones for a camera, falling back to the wildcard entry when unconfigured."""
        if camera_id and camera_id in self._by_camera:
            return self._by_camera[camera_id]
        return self._by_camera.get(WILDCARD_CAMERA, [])

    def zones_containing(
        self,
        camera_id: str | None,
        x: float,
        y: float,
        frame_width: int,
        frame_height: int,
        now: datetime,
        active_only: bool = True,
    ) -> list[Zone]:
        """Zones whose polygon contains the pixel point (x, y) and are active at `now`."""
        if frame_width <= 0 or frame_height <= 0:
            return []

        nx, ny = x / frame_width, y / frame_height
        hits = []
        for zone in self.zones_for(camera_id):
            if active_only and not is_zone_active(zone, now):
                continue
            if point_in_polygon(nx, ny, zone.polygon):
                hits.append(zone)
        return hits
