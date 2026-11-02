"""Baseline vs. after-optimization benchmark for multi-camera streaming capacity.

See docs/plans/multi-camera-streaming-scale.md for the architecture assessment this measures
against. That doc's recommended changes (splitting the display/AI streams, batching + TensorRT
on GPU) are only worth landing if they move these numbers — this script is what "moved" gets
checked against, run once now as `--label baseline` and again after each change.

It drives `MultiStreamManager` directly (no FastAPI, no DB) against the demo clips in
`data/vision/test/`, cycling through them to synthesize however many camera feeds are asked
for and looping each at EOF so a run does not stall mid-measurement.

Two numbers matter, and they answer different questions:
  - achieved_fps:   frames actually processed per second, aggregate across all cameras in the
                     run. This is what a real deployment gets — capped by target_fps and by
                     whatever `get_status()`'s own docstring warns about (source delivery,
                     inference contention). This is the number an optimization should raise.
  - capacity_fps:   `average_inference_capacity_fps` from `get_status()` — 1/latency headroom,
                     NOT the achieved rate. Reported alongside because a change that raises
                     capacity but not achieved_fps means something else (target_fps, source
                     FPS) is now the bottleneck, which is itself worth knowing.

Usage:
    python -m src.benchmark.run_multicam_scale --label baseline
    python -m src.benchmark.run_multicam_scale --label after-stream-split
    python -m src.benchmark.run_multicam_scale --compare eval/results/multicam-scale-baseline.json eval/results/multicam-scale-after-stream-split.json
"""

import argparse
import json
import os
import sys
import time

# Several source clips have Vietnamese filenames; Windows' default console codepage (cp1252)
# cannot encode them, which crashes a plain print() rather than just mangling the text.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VISION_DATA_DIR = REPO_ROOT / "data" / "vision"
# Named explicitly rather than globbing every subdirectory of data/vision: `uploads/` holds
# transient operator-uploaded files (not version-controlled, not the same across machines),
# so including it would make a benchmark run non-reproducible between people. These four are
# the label-scenario clips the team already shot for Gate 2 accuracy testing (34 clips across
# 3 labels, real motion, real camera angles) plus the small demo set from `test/` — using them
# instead of 4 repeated files means N simulated cameras get N genuinely different feeds instead
# of the same footage playing twice, closer to what independent real cameras look like.
_CLIP_DIRS = ("test", "abandoned_object", "falling", "instrusion")
RESULTS_DIR = REPO_ROOT / "eval" / "results"
_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def _demo_clips() -> list[str]:
    """One clip per label directory, not all of them.

    Each directory holds 10-13 clips meant for accuracy eval, some multiple minutes and one
    resolution as large as 2160x1214 — every added clip is another full model load (each
    simulated camera builds its own IncidentDetector) plus its own per-frame decode cost, so
    scanning all ~38 turns a benchmark run into a multi-minute-per-config wait for no benefit
    a throughput/latency measurement needs: content variety across labels is enough, more
    clips of the same label does not change how fast the pipeline processes frames.
    """
    clips = []
    for name in _CLIP_DIRS:
        clip_dir = VISION_DATA_DIR / name
        if not clip_dir.is_dir():
            continue
        candidates = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in _VIDEO_EXT)
        if candidates:
            clips.append(str(candidates[0]))
    return clips


def _device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return f"cpu:{os.cpu_count()}-cores"


def _run_one(camera_count: int, clips: list[str], duration: float, warmup: float, target_fps: float) -> dict:
    from src.vision.stream_manager import MultiStreamManager

    manager = MultiStreamManager()
    for i in range(camera_count):
        manager.add_camera(
            camera_id=f"bench_{i}",
            rtsp_url=clips[i % len(clips)],
            target_fps=target_fps,
            autostart=True,
            loop_finite_sources=True,
        )

    try:
        time.sleep(warmup)
        processed_before = manager.get_status()["total_processed_frames"]
        time.sleep(duration)
        status = manager.get_status()
        processed_after = status["total_processed_frames"]

        camera_stats = status["cameras"]
        running = [c for c in camera_stats if c["running"]]
        avg = lambda key: (sum(c[key] for c in running) / len(running)) if running else 0.0  # noqa: E731

        return {
            "cameras": camera_count,
            "achieved_fps": round((processed_after - processed_before) / duration, 2),
            "capacity_fps": status["average_inference_capacity_fps"],
            "latency_p50_ms": round(avg("latency_p50_ms"), 1),
            "latency_p95_ms": round(avg("latency_p95_ms"), 1),
            "latency_p99_ms": round(avg("latency_p99_ms"), 1),
            "processing_errors": status["processing_errors"],
            "active_cameras": status["active_cameras"],
        }
    finally:
        manager.stop_all()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def run(camera_counts: list[int], duration: float, warmup: float, target_fps: float, label: str, repeats: int) -> dict:
    clips = _demo_clips()
    if not clips:
        print(f"No demo clips found under {VISION_DATA_DIR} in {_CLIP_DIRS}.", file=sys.stderr)
        sys.exit(1)

    device = _device_label()
    print(f"Device: {device}")
    print(f"Clips: {', '.join(f'{Path(c).parent.name}/{Path(c).name}' for c in clips)}")
    print(f"{'cameras':>7}  {'achieved_fps (mean+/-sd)':>24}  {'capacity_fps':>12}  {'p95_ms (mean+/-sd)':>20}  {'errors':>6}")

    rows = []
    for count in camera_counts:
        # A single sample per config cannot separate a real effect from run-to-run noise —
        # background OS load, thread scheduling, and (on a laptop) thermal throttling all
        # move these numbers 30-80% between two runs of *identical* code, as observed while
        # building this script. Repeats + reported stdev is what makes a before/after
        # comparison mean anything instead of chasing noise.
        samples = [_run_one(count, clips, duration, warmup, target_fps) for _ in range(repeats)]
        fps_values = [s["achieved_fps"] for s in samples]
        p95_values = [s["latency_p95_ms"] for s in samples]
        fps_mean = _mean(fps_values)
        p95_mean = _mean(p95_values)

        row = {
            "cameras": count,
            "achieved_fps_mean": round(fps_mean, 2),
            "achieved_fps_stdev": round(_stdev(fps_values, fps_mean), 2),
            "capacity_fps_mean": round(_mean([s["capacity_fps"] for s in samples]), 2),
            "latency_p95_ms_mean": round(p95_mean, 1),
            "latency_p95_ms_stdev": round(_stdev(p95_values, p95_mean), 1),
            "processing_errors_total": sum(s["processing_errors"] for s in samples),
            "samples": samples,
        }
        rows.append(row)
        print(
            f"{row['cameras']:>7}  {row['achieved_fps_mean']:>10} +/- {row['achieved_fps_stdev']:<9}  "
            f"{row['capacity_fps_mean']:>12}  {row['latency_p95_ms_mean']:>8} +/- {row['latency_p95_ms_stdev']:<8}  "
            f"{row['processing_errors_total']:>6}"
        )

    result = {
        "label": label,
        "recorded_at": datetime.now(UTC).isoformat(),
        "device": device,
        "duration_sec": duration,
        "warmup_sec": warmup,
        "target_fps": target_fps,
        "repeats": repeats,
        "rows": rows,
    }
    return result


def compare(path_a: str, path_b: str) -> None:
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)

    rows_a = {r["cameras"]: r for r in a["rows"]}
    rows_b = {r["cameras"]: r for r in b["rows"]}

    print(f"A = {a['label']} ({a['recorded_at']}, {a['device']})")
    print(f"B = {b['label']} ({b['recorded_at']}, {b['device']})")
    if a["device"] != b["device"]:
        print("WARNING: different devices — the delta below conflates the hardware change with the code change.")
    print(f"\n{'cameras':>7}  {'achieved_fps A->B':>26}  {'p95_ms A->B':>26}")

    for count in sorted(set(rows_a) | set(rows_b)):
        ra, rb = rows_a.get(count), rows_b.get(count)
        if ra is None or rb is None:
            print(f"{count:>7}  (missing in {'A' if ra is None else 'B'})")
            continue
        fps_a, fps_b = ra["achieved_fps_mean"], rb["achieved_fps_mean"]
        lat_a, lat_b = ra["latency_p95_ms_mean"], rb["latency_p95_ms_mean"]
        fps_delta_pct = ((fps_b - fps_a) / fps_a * 100) if fps_a else 0.0
        lat_delta_pct = ((lat_b - lat_a) / lat_a * 100) if lat_a else 0.0
        # A delta smaller than either side's own run-to-run spread is not distinguishable
        # from noise — flagging it here is what stops "-30% at 1 camera" from being read as
        # a real regression when it is actually two samples of the same distribution.
        noisy = abs(fps_b - fps_a) < (ra["achieved_fps_stdev"] + rb["achieved_fps_stdev"])
        note = "  (within noise — increase --repeats)" if noisy else ""
        print(
            f"{count:>7}  {fps_a:>8} -> {fps_b:<8} ({fps_delta_pct:+.0f}%)  "
            f"{lat_a:>8} -> {lat_b:<8} ({lat_delta_pct:+.0f}%){note}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", default="1,2,4", help="Comma-separated camera counts to test.")
    parser.add_argument("--duration", type=float, default=30.0, help="Measurement window per camera count, seconds.")
    parser.add_argument("--warmup", type=float, default=8.0, help="Discarded settle time before measuring, seconds.")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Per-camera target_fps passed to add_camera.")
    parser.add_argument("--repeats", type=int, default=3, help="Samples per camera count, to separate a real effect from run-to-run noise.")
    parser.add_argument("--label", default="run", help="Tag for this result set, e.g. 'baseline'.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: eval/results/multicam-scale-<label>.json).")
    parser.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"), help="Skip benchmarking; print a delta table between two saved results.")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    camera_counts = [int(c.strip()) for c in args.cameras.split(",") if c.strip()]
    result = run(camera_counts, args.duration, args.warmup, args.target_fps, args.label, args.repeats)

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"multicam-scale-{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
