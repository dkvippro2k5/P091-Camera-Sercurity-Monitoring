"""Turn a raw video clip into a benchmark case: auto-find the trigger moment, extract
BEFORE/AFTER frames around it, save as JPEG.

Reuses the real `VisionPipeline`/`RuleEngine` to detect *when* an event fires, the same
way `MonitoringService` does in production -- this script does not guess timestamps by
eye, it runs the actual detector over the footage.

Usage:
    python scripts/extract_benchmark_frames.py data/vision/test/my_clip.mov \\
        --case-id fall_02_vertical_drop --camera-id cam_lobby_01

Output: JPEGs at tests/fixtures/benchmark/images/<case_id>_before{1..N}.jpg and
_after{1..N}.jpg, plus a printed summary (label, reason, timestamp) -- ground truth
(`expected_verdict`, the human `note`) is a judgment call this script cannot make, so it
prints what's needed to hand-write the `cases.jsonl` entry rather than guessing one.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision import VisionPipeline  # noqa: E402
from src.vision.rule_engine import RuleEngine  # noqa: E402
from src.vision.schemas import CameraZoneConfig, Zone  # noqa: E402
from src.vision.zone_config import ZoneManager  # noqa: E402


def _full_frame_pipeline(camera_id: str) -> VisionPipeline:
    """Placeholder geofence covering almost the whole frame.

    Intrusion has no default zone (no `data/vision/zones.json` in this repo), so without
    this the rule never fires and the auto-detector reports "no event" for any intrusion
    clip -- not a bug, just an unconfigured zone. This is NOT a real geofence: it lets any
    person passing through trigger the rule, purely so the trigger frame can be located
    for benchmark extraction.
    """
    zone = Zone(name="full_frame_placeholder", polygon=[(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)])
    zone_manager = ZoneManager([CameraZoneConfig(camera_id=camera_id, zones=[zone])])
    return VisionPipeline(rule_engine=RuleEngine(zone_manager=zone_manager))


def find_trigger_frame(
    video_path: str, camera_id: str, zone_full_frame: bool = False, sample_fps: float = 8.0
) -> tuple[int, float, str, str] | None:
    """First pass: run the real pipeline frame-by-frame, stop at the first fired event.

    Runs inference at `sample_fps`, not the file's native fps -- production never
    processes faster than `camera_target_fps` (8.0 by default, see config.py) either, so
    running full 60fps inference here would burn CPU time modeling a sampling rate no
    live camera actually uses, without making the trigger detection any more faithful.

    Returns (frame_idx, fps, label, reason) or None if nothing fired in the whole clip.
    """
    pipeline = _full_frame_pipeline(camera_id) if zone_full_frame else VisionPipeline()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not decode '{video_path}' -- corrupt, truncated, or unsupported codec.")

    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = reported_fps if reported_fps and math.isfinite(reported_fps) and reported_fps > 0 else 30.0
    stride = max(1, round(fps / sample_fps)) if sample_fps > 0 else 1

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                _, frame_res = pipeline.process_frame(
                    frame, frame_id=frame_idx, camera_id=camera_id, timestamp=frame_idx / fps
                )
                if frame_res.events:
                    event = frame_res.events[0]
                    return frame_idx, fps, event.label, event.reason or ""
            frame_idx += 1
    finally:
        cap.release()
    return None


def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not decode '{video_path}'.")
    try:
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        return reported_fps if reported_fps and math.isfinite(reported_fps) and reported_fps > 0 else 30.0
    finally:
        cap.release()


def extract_frames(
    video_path: str,
    trigger_idx: int,
    fps: float,
    pre_count: int,
    post_count: int,
    spacing_sec: float,
    max_edge: int,
) -> tuple[list, list]:
    """Second pass: seek to specific offsets around the trigger and grab those frames.

    Spaced by wall-clock time (not consecutive frames) -- at 60fps, consecutive frames are
    ~17ms apart and look nearly identical, giving a VLM nothing to read a transition from.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not decode '{video_path}' on the second pass.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, round(spacing_sec * fps))
    pre_idxs = [trigger_idx - step * k for k in range(pre_count, 0, -1)]
    post_idxs = [trigger_idx + step * k for k in range(1, post_count + 1)]

    def grab(idx: int):
        idx = max(0, min(total_frames - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {idx} (total_frames={total_frames}).")
        h, w = frame.shape[:2]
        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    try:
        pre_frames = [grab(i) for i in pre_idxs]
        post_frames = [grab(i) for i in post_idxs]
    finally:
        cap.release()
    return pre_frames, post_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--case-id", required=True, help="Used as the output filename prefix.")
    parser.add_argument("--camera-id", default="cam_test")
    parser.add_argument("--pre", type=int, default=4)
    parser.add_argument("--post", type=int, default=2)
    parser.add_argument("--spacing-sec", type=float, default=0.4)
    parser.add_argument("--max-edge", type=int, default=1280, help="Resize longest edge down from 4K.")
    parser.add_argument("--out-dir", default="tests/fixtures/benchmark/images")
    parser.add_argument(
        "--zone-full-frame",
        action="store_true",
        help="Use a placeholder almost-full-frame zone so intrusion clips can trigger without a real zones.json.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=8.0,
        help="Inference sampling rate for pass 1 (matches camera_target_fps default; 0 = every frame).",
    )
    parser.add_argument(
        "--trigger-time-sec",
        type=float,
        default=None,
        help="Skip RuleEngine auto-detection; anchor extraction on this known timestamp instead "
        "(e.g. a human-labeled event_start). Use when auto-detect can't be trusted -- no real zone "
        "config for intrusion, or a case the detector misses/mistimes.",
    )
    args = parser.parse_args()

    if args.trigger_time_sec is not None:
        fps = get_fps(args.video_path)
        trigger_idx = round(args.trigger_time_sec * fps)
        label, reason = None, None
        print(f"Using manual trigger time t={args.trigger_time_sec:.2f}s (frame {trigger_idx}, fps={fps:.2f}).")
    else:
        print(f"Pass 1/2: running RuleEngine over '{args.video_path}' to find the trigger frame...")
        found = find_trigger_frame(
            args.video_path, args.camera_id, zone_full_frame=args.zone_full_frame, sample_fps=args.sample_fps
        )
        if found is None:
            print("No event fired in this clip -- nothing to extract. Check the footage or rule thresholds.")
            return
        trigger_idx, fps, label, reason = found
        print(f"  Trigger: frame {trigger_idx} (t={trigger_idx / fps:.2f}s), label={label!r}")
        print(f"  reason: {reason}")

    print(f"Pass 2/2: extracting {args.pre} before + {args.post} after (spacing {args.spacing_sec}s)...")
    pre_frames, post_frames = extract_frames(
        args.video_path, trigger_idx, fps, args.pre, args.post, args.spacing_sec, args.max_edge
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, frame in enumerate(pre_frames, start=1):
        path = out_dir / f"{args.case_id}_before{i}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(path)
    for i, frame in enumerate(post_frames, start=1):
        path = out_dir / f"{args.case_id}_after{i}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(path)

    print(f"\nSaved {len(saved)} frames:")
    for path in saved:
        print(f"  {path}")

    print(
        "\nGround truth is a human call -- add this to tests/fixtures/benchmark/cases.jsonl "
        "yourself (or tell me the expected_verdict + a short note and I will):\n"
        "{\n"
        f'  "case_id": "{args.case_id}",\n'
        f'  "label": "{label}",\n'
        '  "expected_verdict": "<real | false_positive | uncertain -- your judgment>",\n'
        f'  "image_files": {[p.name for p in saved]},\n'.replace("'", '"')
        + f'  "pre_count": {len(pre_frames)},\n'
        f'  "reason": {reason!r},\n'
        '  "zone_name": null,\n'
        f'  "camera_name": "{args.camera_id}",\n'
        '  "note": "<what makes this case interesting -- why it should pass/fail>"\n'
        "}"
    )


if __name__ == "__main__":
    main()
