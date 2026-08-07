"""Visual check for pose-confirmed fall detection.

Runs the real pipeline over a video and shows the annotated output, so the COCO-17
skeleton `draw_pose` renders can be eyeballed against the footage that produced it.

Pose only runs on detections the cheap aspect-ratio pre-filter flags as fall candidates,
which is the whole point of the cascade — so on footage of upright people you will
correctly see no skeletons at all. `--force-pose` bypasses that gate to prove the pose
model itself works, at the cost of running inference on every person in frame.

    python scripts/demo_pose_overlay.py data/vision/test/fall.mp4 --show
    python scripts/demo_pose_overlay.py data/vision/test/fall.mp4 --force-pose --output out.mp4
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision import VisionPipeline  # noqa: E402
from src.vision.pose import PoseEstimator  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render pose skeletons over a video")
    parser.add_argument("source", nargs="?", default="data/vision/test/fall.mp4")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--show", action="store_true", help="Open a live preview window")
    parser.add_argument("--output", type=str, default=None, help="Write annotated MP4 here")
    parser.add_argument(
        "--force-pose",
        action="store_true",
        help="Run pose on every person, not just gated fall candidates",
    )
    args = parser.parse_args()

    pipeline = VisionPipeline()
    pose: PoseEstimator = pipeline.pose_estimator
    print(f"detector : {pipeline.detector.model_path}")
    print(f"pose     : {pose.model_path} (available={pose.available})")
    if not pose.available:
        print("Pose weights missing — no skeletons will render. Put yolo11n-pose.pt in src/vision/weights/.")
        return 1

    if args.force_pose:
        # is_fall_candidate is the cascade's pre-filter. Forcing it True makes every person
        # a candidate, which is what you want when proving the pose model works rather than
        # measuring the cascade's cost.
        pipeline.rule_engine.is_fall_candidate = lambda det: True
        print("force-pose  : ON — pose runs on every person")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Could not open '{args.source}'")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    frames = posed = 0
    events: list[str] = []

    try:
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            # Video-relative time, so the 3-second fall dwell means 3 seconds of footage
            # rather than 3 seconds of wall-clock spent decoding it.
            annotated, result = pipeline.process_frame(
                frame,
                frame_id=frames,
                camera_id="pose_demo",
                timestamp=frames / fps,
            )
            frames += 1
            posed += sum(1 for d in result.detections if d.keypoints)
            events.extend(f"frame {frames}: {e.label} {e.track_id or ''}".strip() for e in result.events)

            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(annotated)

            if args.show:
                cv2.imshow("pose overlay - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\nframes processed     : {frames}")
    print(f"detections with pose : {posed}")
    print(f"incidents fired      : {len(events)}")
    for line in events[:10]:
        print(f"  {line}")
    if args.output:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
