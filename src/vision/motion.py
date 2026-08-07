"""Frame-differencing gate that skips inference when nothing in view changed.

Ported in intent from DeepCamera's `src/camera/detector/motion.js`, which computes a
per-pixel channel-average difference and thresholds it at 0x15 (21/255). Note that in
DeepCamera the check is vestigial — `blenderRegion` is an empty function, the call is
commented out, and `has_motion` is hardcoded true; what actually throttles its detector is
a 200ms debounce. So the algorithm is reimplemented here rather than copied, and the hard
part is one DeepCamera never had to solve.

**Skipping a frame is not free in P-091, because the rules are stateful.** Three ways a
naive gate silently breaks things:

  1. Abandoned objects are stationary *by definition*. A bag left alone generates zero
     motion; gate on motion alone and the bag drops out of detections, its track dies,
     track_ttl_sec evicts it, and `abandoned_object` can never fire. A whole PRD label
     disappears with no error.
  2. A fall dwell mid-count needs frames to keep accumulating toward the PRD's 3 seconds.
  3. `model.track(persist=True)` degrades when starved of frames, and every rule keys on
     the track IDs it produces.

Hence a skip requires *three* conditions, not one: no motion, no live track state for the
camera, and no elapsed heartbeat. The gate is therefore only ever active on a genuinely
idle scene — which is the case it was introduced for, a corridor at 3am.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# DeepCamera's threshold(value) { return (value > 0x15) ? 0xFF : 0 } — motion.js:56-58.
# 21/255 is high enough to ignore sensor noise and compression mosquito artifacts on a
# static scene, low enough to catch a person entering at the far end of a corridor.
DEFAULT_PIXEL_THRESHOLD = 21

# Fraction of pixels that must change before the frame counts as moving. 0.2% of a
# 160x120 grid is ~38 pixels: smaller than any real person at any usable distance, so
# this errs strongly toward "there was motion" — the safe direction for a security system.
DEFAULT_MIN_CHANGED_FRACTION = 0.002

# Difference is computed at this size regardless of input resolution. Downscaling is what
# makes the gate cheap enough to be worth having (a 1080p absdiff is not free), and it
# doubles as a low-pass filter against single-pixel noise.
DEFAULT_DIFF_SIZE = (160, 120)


class MotionGate:
    """Decides whether a frame is worth running detection on.

    Stateful (holds the previous frame), so one instance per stream — never share across
    cameras, for the same reason detectors and rule engines are per-worker.
    """

    def __init__(
        self,
        enabled: bool = False,
        pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
        min_changed_fraction: float = DEFAULT_MIN_CHANGED_FRACTION,
        heartbeat_sec: float = 1.0,
        diff_size: tuple[int, int] = DEFAULT_DIFF_SIZE,
    ):
        # Defaults off: the capacity numbers recorded in MultiStreamManager's docstring
        # were measured without gating, and enabling it by default would make them
        # un-reproducible and the A/B benchmark meaningless.
        self.enabled = enabled
        self.pixel_threshold = pixel_threshold
        self.min_changed_fraction = min_changed_fraction
        self.heartbeat_sec = heartbeat_sec
        self.diff_size = diff_size

        self._previous: np.ndarray | None = None
        self._last_inference_time: float | None = None
        self.skipped_frames = 0
        self.processed_frames = 0

    def reset(self) -> None:
        """Forget the reference frame, e.g. between videos or after a reconnect."""
        self._previous = None
        self._last_inference_time = None
        self.skipped_frames = 0
        self.processed_frames = 0

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, self.diff_size, interpolation=cv2.INTER_AREA)

    def has_motion(self, frame: np.ndarray) -> bool:
        """Whether this frame differs from the previous one beyond the noise threshold.

        The first frame always counts as motion: with no reference there is nothing to
        compare against, and defaulting to "still" would skip the opening of every stream.
        """
        small = self._downscale(frame)
        previous, self._previous = self._previous, small

        if previous is None or previous.shape != small.shape:
            return True

        diff = cv2.absdiff(small, previous)
        changed = int(np.count_nonzero(diff > self.pixel_threshold))
        return changed >= max(1, int(small.size * self.min_changed_fraction))

    def should_process(self, frame: np.ndarray, timestamp: float, has_active_tracks: bool = False) -> bool:
        """Whether detection should run on this frame.

        Args:
            frame: Input BGR frame.
            timestamp: Stream time in seconds, the same clock the dwell timers use, so the
                heartbeat means the same thing for a live camera and a decoded file.
            has_active_tracks: Whether the rule engine holds live state for this camera.
                True forces processing — see the module docstring.

        Returns:
            True to run detection. Always True when the gate is disabled.
        """
        if not self.enabled:
            self.processed_frames += 1
            self._last_inference_time = timestamp
            return True

        motion = self.has_motion(frame)

        # Heartbeat: run at least this often no matter how still the scene is, so the
        # tracker stays warm and stationary-object timers keep advancing.
        stale = self._last_inference_time is None or (timestamp - self._last_inference_time) >= self.heartbeat_sec

        if motion or has_active_tracks or stale:
            self.processed_frames += 1
            self._last_inference_time = timestamp
            return True

        self.skipped_frames += 1
        return False

    @property
    def skip_ratio(self) -> float:
        """Fraction of frames skipped so far — the number the A/B benchmark reports."""
        total = self.skipped_frames + self.processed_frames
        return round(self.skipped_frames / total, 4) if total else 0.0
