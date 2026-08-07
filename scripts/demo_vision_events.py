"""Ad-hoc end-to-end check: run the full pipeline over a video and report events."""

import logging
import sys
from collections import Counter
from pathlib import Path

# The project is not installed as a package, so `python scripts/demo_vision_events.py`
# puts scripts/ on sys.path rather than the repo root and `src` fails to import. pytest
# only works because it inserts its rootdir for us.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision import VisionPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

video = sys.argv[1] if len(sys.argv) > 1 else "data/vision/test/input_test.mp4"
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 600

pipeline = VisionPipeline()

print(f"detector: weights at '{pipeline.detector.model_path}'")
print(f"zones loaded: {pipeline.rule_engine.zone_manager.zone_count}")

summary = pipeline.process_video(
    video_path=video,
    output_json_path="data/vision/test/handover.json",
    output_video_path="data/vision/test/output_anonymized.mp4",
    max_frames=max_frames,
    camera_id="cam_hallway_01",
)

counts = Counter(e.label for fr in summary.frame_results for e in fr.events)
print(f"\nprocessed {summary.processed_frames} frames @ {summary.average_fps} fps")
print(f"total events: {summary.total_events} -> {dict(counts)}")

for fr in summary.frame_results:
    for e in fr.events:
        print(f"  frame {fr.frame_id:4d}  {e.label:18s} {e.reason}")
