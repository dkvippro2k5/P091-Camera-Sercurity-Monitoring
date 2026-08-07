"""Decide whether a set of pose keypoints describes a person on the ground.

Pure geometry, no training. This replaces the bounding-box aspect-ratio test as the
*primary* fall signal because that test cannot separate the cases that matter:

    lying down      torso horizontal, hips low     -> fall
    crouching       torso vertical,   hips low     -> not a fall  (aspect ratio says fall)
    sitting         torso vertical,   hips mid     -> not a fall
    near the lens   torso vertical,   hips low     -> not a fall  (aspect ratio says fall)

DeepCamera solves the same problem with an ST-GCN action classifier over a 30-frame
keypoint window. That was not ported: it needs a labelled fall dataset the team does not
have, its temporal window duplicates the dwell timer RuleEngine already owns, and a
learned scalar cannot explain itself. Geometry yields an auditable `reason` string, which
matters because the PRD feeds `reason` into an LLM for false-positive filtering.
"""

import math

from src.vision.pose import (
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
)

Keypoints = list[tuple[float, float, float]]

# A torso this far off vertical is lying/collapsed rather than bent over. 55 deg leaves
# room for someone leaning to pick something up (typically 30-45 deg) without firing.
DEFAULT_TORSO_ANGLE_DEG = 55.0

# Mean confidence of the four torso keypoints. Below this the geometry is guesswork, so
# the classifier returns None (undecided) and the caller keeps its previous behaviour.
DEFAULT_MIN_KP_CONFIDENCE = 0.3


class FallPoseVerdict:
    """Outcome of one pose evaluation, carrying the evidence for the alert text."""

    def __init__(
        self,
        is_prone: bool,
        torso_angle_deg: float,
        reason: str,
        downward_velocity: float | None = None,
    ):
        self.is_prone = is_prone
        self.torso_angle_deg = torso_angle_deg
        self.reason = reason
        self.downward_velocity = downward_velocity

    def __repr__(self) -> str:
        vel_str = f", vel={self.downward_velocity:.1f}px/s" if self.downward_velocity is not None else ""
        return f"FallPoseVerdict(is_prone={self.is_prone}, torso_angle_deg={self.torso_angle_deg:.0f}{vel_str})"


def _midpoint(keypoints: Keypoints, left: int, right: int, min_conf: float) -> tuple[float, float] | None:
    """Average a symmetric keypoint pair, tolerating one side being occluded."""
    lx, ly, lc = keypoints[left]
    rx, ry, rc = keypoints[right]

    usable = [(x, y) for x, y, c in ((lx, ly, lc), (rx, ry, rc)) if c >= min_conf]
    if not usable:
        return None
    return (
        sum(p[0] for p in usable) / len(usable),
        sum(p[1] for p in usable) / len(usable),
    )


def classify_pose(
    keypoints: Keypoints | None,
    torso_angle_deg: float = DEFAULT_TORSO_ANGLE_DEG,
    min_confidence: float = DEFAULT_MIN_KP_CONFIDENCE,
    downward_velocity: float | None = None,
) -> FallPoseVerdict | None:
    """Judge one person's pose.

    Returns None when the keypoints cannot support a decision — missing, too few, or
    too low-confidence. None means *undecided*, which callers must not read as "upright";
    the fall rule falls back to its bounding-box heuristic in that case.
    """
    if not keypoints or len(keypoints) <= KP_RIGHT_HIP:
        return None

    shoulders = _midpoint(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, min_confidence)
    hips = _midpoint(keypoints, KP_LEFT_HIP, KP_RIGHT_HIP, min_confidence)
    if shoulders is None or hips is None:
        return None

    dx = hips[0] - shoulders[0]
    dy = hips[1] - shoulders[1]
    if math.hypot(dx, dy) < 1.0:
        # Torso collapsed to a point: either a straight-down camera or garbage keypoints.
        # Either way the angle is meaningless, so decline rather than guess.
        return None

    # Angle away from the image's vertical axis. atan2 on (|dx|, |dy|) gives 0 deg for a
    # perfectly upright torso and 90 deg for a perfectly horizontal one, regardless of
    # whether the person is head-up or head-down.
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))

    # Secondary signal: the head at or below hip level. Catches a person collapsed
    # directly toward the camera, where the torso foreshortens and the angle stays small.
    nose_x, nose_y, nose_c = keypoints[KP_NOSE]
    head_below_hips = nose_c >= min_confidence and nose_y >= hips[1]

    vel_suffix = f" (downward vel {downward_velocity:.1f}px/s)" if downward_velocity is not None else ""

    if angle >= torso_angle_deg:
        return FallPoseVerdict(
            True,
            angle,
            f"torso {angle:.0f}deg from vertical (threshold {torso_angle_deg:.0f}deg){vel_suffix}",
            downward_velocity,
        )
    if head_below_hips:
        return FallPoseVerdict(
            True,
            angle,
            f"head at/below hip level (torso {angle:.0f}deg){vel_suffix}",
            downward_velocity,
        )

    return FallPoseVerdict(
        False,
        angle,
        f"torso upright at {angle:.0f}deg from vertical{vel_suffix}",
        downward_velocity,
    )
