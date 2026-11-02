"""Tests for the motion gate.

The interesting cases are not "does it detect motion" — that is three lines of OpenCV —
but the three ways a naive gate silently breaks P-091's stateful rules. See motion.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.motion import MotionGate
from src.vision.pipeline import VisionPipeline
from src.vision.rule_engine import LABEL_ABANDONED, RuleEngine
from src.vision.schemas import BoundingBox, IncidentDetection
from src.vision.zone_config import ZoneManager

FRAME_W, FRAME_H = 320, 240


def _blank() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _noisy(seed: int) -> np.ndarray:
    """A frame that differs from a blank one across most of its area."""
    rng = np.random.default_rng(seed)
    return rng.integers(120, 255, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)


class StaticBagDetector:
    """Detector that always reports one motionless backpack and no people.

    Models the exact scenario the gate must not break: an abandoned bag generates zero
    motion forever, so a gate keyed only on pixel change would starve the rule that
    exists to find it.
    """

    def __init__(self):
        self.calls = 0

    def detect(self, frame, frame_id=0, camera_id=None, timestamp=None):
        self.calls += 1
        return [
            IncidentDetection(
                label="backpack",
                track_id="track_bag",
                confidence=0.9,
                timestamp=timestamp or 0.0,
                camera_id=camera_id,
                bbox=BoundingBox(xmin=50, ymin=50, xmax=80, ymax=90),
            )
        ]


# --- gate mechanics ---------------------------------------------------------------


def test_disabled_gate_always_processes():
    gate = MotionGate(enabled=False)
    frame = _blank()
    assert all(gate.should_process(frame, timestamp=t) for t in range(5))
    assert gate.skipped_frames == 0


def test_first_frame_always_processes():
    """With no reference frame there is nothing to compare, so never skip the opener."""
    gate = MotionGate(enabled=True)
    assert gate.should_process(_blank(), timestamp=0.0) is True


def test_identical_frames_are_skipped_once_the_scene_is_idle():
    gate = MotionGate(enabled=True, heartbeat_sec=1000.0)
    frame = _blank()

    gate.should_process(frame, timestamp=0.0)  # primes the reference frame
    assert gate.should_process(frame, timestamp=0.1, has_active_tracks=False) is False
    assert gate.skipped_frames == 1


def test_changed_frame_is_processed():
    gate = MotionGate(enabled=True, heartbeat_sec=1000.0)
    gate.should_process(_blank(), timestamp=0.0)
    assert gate.should_process(_noisy(1), timestamp=0.1, has_active_tracks=False) is True


def test_active_tracks_force_processing_despite_a_frozen_frame():
    """A dwell timer mid-count must keep receiving frames or it can never reach 3s."""
    gate = MotionGate(enabled=True, heartbeat_sec=1000.0)
    frame = _blank()

    gate.should_process(frame, timestamp=0.0)
    assert gate.should_process(frame, timestamp=0.1, has_active_tracks=True) is True
    assert gate.skipped_frames == 0


def test_heartbeat_forces_processing_on_a_completely_still_scene():
    gate = MotionGate(enabled=True, heartbeat_sec=1.0)
    frame = _blank()

    gate.should_process(frame, timestamp=0.0)
    assert gate.should_process(frame, timestamp=0.5, has_active_tracks=False) is False
    # Heartbeat elapsed: run anyway, so the tracker and stationary timers stay warm.
    assert gate.should_process(frame, timestamp=1.0, has_active_tracks=False) is True


def test_reset_clears_the_reference_frame():
    gate = MotionGate(enabled=True, heartbeat_sec=1000.0)
    frame = _blank()
    gate.should_process(frame, timestamp=0.0)
    assert gate.should_process(frame, timestamp=0.1) is False

    gate.reset()
    # After a reset there is no reference again, so the next frame must be processed.
    assert gate.should_process(frame, timestamp=0.2) is True
    assert gate.skipped_frames == 0


def test_skip_ratio_reports_what_was_dropped():
    gate = MotionGate(enabled=True, heartbeat_sec=1000.0)
    frame = _blank()
    for t in range(10):
        gate.should_process(frame, timestamp=t * 0.01)

    assert gate.skipped_frames == 9
    assert gate.processed_frames == 1
    assert gate.skip_ratio == pytest.approx(0.9)


# --- the regression the gate was designed around ----------------------------------


def test_abandoned_object_still_fires_with_gating_enabled():
    """A stationary bag produces zero motion; abandonment must still be detected.

    This is the failure mode a pixel-diff gate introduces if it skips on motion alone:
    the bag leaves the detection list, its track is evicted, and a whole PRD label
    silently stops working.
    """
    detector = StaticBagDetector()
    pipeline = VisionPipeline(
        detector=detector,
        # No person ever appears, so the bag has no owner to require.
        rule_engine=RuleEngine(
            zone_manager=ZoneManager(), abandon_sec=2.0, track_ttl_sec=1000.0, abandon_require_owner=False
        ),
        zones_path=None,
        draw_overlays=False,
        motion_gate=MotionGate(enabled=True, heartbeat_sec=0.5),
    )

    frame = _blank()
    fired = []
    for i in range(40):
        _, result = pipeline.process_frame(frame, frame_id=i, camera_id="cam_01", timestamp=i * 0.25)
        fired.extend(result.events)

    assert any(e.label == LABEL_ABANDONED for e in fired), (
        "motion gating suppressed abandoned_object — the bag's track was starved"
    )


def test_gate_skips_frames_on_an_empty_idle_scene():
    """The gate must actually save work, or it is pure risk for no benefit."""

    class EmptyDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame, frame_id=0, camera_id=None, timestamp=None):
            self.calls += 1
            return []

    detector = EmptyDetector()
    pipeline = VisionPipeline(
        detector=detector,
        rule_engine=RuleEngine(zone_manager=ZoneManager()),
        zones_path=None,
        draw_overlays=False,
        motion_gate=MotionGate(enabled=True, heartbeat_sec=1000.0),
    )

    frame = _blank()
    for i in range(20):
        pipeline.process_frame(frame, frame_id=i, camera_id="cam_01", timestamp=i * 0.05)

    assert detector.calls == 1, f"expected 1 inference on an idle scene, got {detector.calls}"
    assert pipeline.motion_gate.skipped_frames == 19


def test_a_skipped_frame_still_comes_back_for_the_preview():
    """Skipping is a throughput decision about inference, not about output.

    The frame still reaches previews and recorders on every tick, so the skip path must
    return a usable array rather than None — a preview that goes blank on an idle scene
    looks exactly like a camera that died.
    """

    class EmptyDetector:
        def detect(self, frame, frame_id=0, camera_id=None, timestamp=None):
            return []

    pipeline = VisionPipeline(
        detector=EmptyDetector(),
        rule_engine=RuleEngine(zone_manager=ZoneManager()),
        zones_path=None,
        draw_overlays=False,
        motion_gate=MotionGate(enabled=True, heartbeat_sec=1000.0),
    )

    frame = _blank()
    for i in range(5):
        output, _ = pipeline.process_frame(frame, frame_id=i, camera_id="cam_01", timestamp=i * 0.05)
        assert output is not None
        assert output.shape == frame.shape

    # An idle scene with no live tracks is the only state that actually skips — with a
    # track alive the gate refuses to, which is what keeps `abandoned_object` working.
    assert pipeline.motion_gate.skipped_frames == 4


def test_process_frame_result_is_empty_but_valid_when_skipped():
    pipeline = VisionPipeline(
        detector=StaticBagDetector(),
        rule_engine=RuleEngine(zone_manager=ZoneManager()),
        zones_path=None,
        draw_overlays=False,
        motion_gate=MotionGate(enabled=True, heartbeat_sec=1000.0),
    )
    frame = _blank()

    pipeline.process_frame(frame, frame_id=0, camera_id="cam_01", timestamp=0.0)
    out, result = pipeline.process_frame(frame, frame_id=1, camera_id="cam_01", timestamp=0.01)

    assert result.frame_id == 1
    assert result.camera_id == "cam_01"
    assert result.events == []
    assert out.shape == frame.shape


def test_gate_state_is_per_instance_not_shared():
    """Two cameras sharing one gate would diff camera A against camera B."""
    a, b = MotionGate(enabled=True), MotionGate(enabled=True)
    a.should_process(_blank(), timestamp=0.0)
    # b has never seen a frame, so its first call must still process.
    assert b.should_process(_noisy(7), timestamp=0.0) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
