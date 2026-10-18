"""Clip-level fall-detection benchmark on public labelled datasets (UR Fall, GMDCSA24).

`src/services/eval_report.py` measures precision from guard verdicts but deliberately refuses
to print recall: a missed fall never became an alert, so it is not in the history. This script
is the other half `docs/plans/benchmark-criteria.md` (Group 1) asks for — a labelled clip set
where every true fall is marked whether or not the system fired, run through the same
detector + RuleEngine configuration the live service builds.

Dataset: UR Fall Detection Dataset (Kwolek & Kepski, 2014), http://fenix.ur.edu.pl/~mkepski/ds/uf.html
  - 30 fall sequences, 40 ADL (activities of daily living) sequences, cam0, 30 fps.
  - `urfall-cam0-falls.csv` labels every frame -1 (not lying), 0 (falling), 1 (lying).
  - The `*-cam0.mp4` files place depth (left) and RGB (right) side by side at 320x240 each;
    only the RGB half is fed to the pipeline. That is a quarter of the 640x480 the RGB PNG
    archives offer (~4 GB in total), which is a stated limitation of these numbers.

Dataset: GMDCSA24 (Alam et al., 2024, MIT), https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
  - 4 subjects, 3 home setups, day and night recordings; `Subject N/{Fall,ADL}/NN.mp4`.
  - `Subject N/{Fall,ADL}.csv` give second-level segments, e.g. `Falling (SW)[3.4 to 6]`.
  - Kept as the held-out set: rule changes are not tuned against it.

**Protocol — read before quoting a number.**

- Sampling: frames are taken at `--fps` on the video clock (default 5, the
  `CAMERA_TARGET_FPS` default), with video-relative timestamps — what a live worker sees.
- Freeze padding: every UR fall clip ends 0.1-2.1 s after the person reaches the floor, and
  `RuleEngine` requires 3 s of persistence (PRD 3.1). Unpadded, recall is zero *by
  construction of the dataset*, not of the system. `--pad-sec` (default 4) repeats each clip's
  last frame, i.e. "the person stayed where the clip left them". It is applied identically to
  ADL clips, so an ADL that ends lying on a bed gets the same chance to raise a false alarm.
  Run `--pad-sec 0` to see the unpadded result.
- Matching: on a fall clip, the first `fall_detection` alert at or after the labelled fall
  onset (minus `ONSET_TOLERANCE_SEC`) is a true positive; earlier alerts are false alarms;
  later ones are duplicates. Every alert on an ADL clip is a false alarm.
- A fresh detector, tracker and rule engine per clip, as for a newly added camera.

Usage:
    python -m src.benchmark.run_fall_benchmark --download        # fetch UR Fall (~94 MB) once
    python -m src.benchmark.run_fall_benchmark                    # padded run, writes results
    python -m src.benchmark.run_fall_benchmark --pad-sec 0 --label unpadded
    python -m src.benchmark.run_fall_benchmark --dataset gmdcsa24 --download   # ~1.1 GB
    python -m src.benchmark.run_fall_benchmark --dataset gmdcsa24
"""

import argparse
import csv
import inspect
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DATA_DIR = REPO_ROOT / "data" / "benchmark"
DEFAULT_DATA_DIR = BENCHMARK_DATA_DIR / "urfall"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
URFALL_BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
URFALL_FALLS = 30
URFALL_ADLS = 40
GMDCSA24_REPO = "ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
GMDCSA24_SUBJECTS = 4
# "Falling (SW)[3.4 to 6]" — tolerant of the stray spaces and colons in the published CSVs.
_GMDCSA24_SEGMENT = re.compile(r"([A-Za-z][^\[\];:,]*?)\s*\[\s*([\d.]+)\s*(?:to)?\s*([\d.]+)\s*\]")
_LYING_CLASSES = ("lying", "sleeping", "laying")

LABEL_FALL = "fall_detection"
# The CSV marks the first "falling" frame; a detector that latches on the first frames of the
# descent can legitimately fire a fraction of a second before the annotator's onset.
ONSET_TOLERANCE_SEC = 0.5


@dataclass(frozen=True)
class ClipTruth:
    """Ground truth for one clip. Times are seconds from the first frame."""

    clip_id: str
    path: str
    kind: str  # "fall" or "adl"
    fall_start_sec: float | None = None
    lying_start_sec: float | None = None
    # ADLs that include a lying-down segment (bed, floor exercise) — the hard negatives.
    has_lying: bool = False
    # UR Fall mp4s carry depth on the left half; only the RGB half is a camera view.
    crop_right_half: bool = False


@dataclass
class ClipRun:
    """What the pipeline did on one clip."""

    clip_id: str
    kind: str
    footage_sec: float
    analysed_sec: float
    sampled_frames: int
    person_frame_fraction: float
    alert_times: list[float] = field(default_factory=list)


@dataclass
class ClipOutcome:
    clip_id: str
    kind: str
    detected: bool
    false_alarms: int
    duplicates: int
    latency_from_lying_sec: float | None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval; stays inside [0, 1] and behaves at 0/n and n/n."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def match_clip(truth: ClipTruth, run: ClipRun) -> ClipOutcome:
    """Classify each alert on a clip as true positive, false alarm, or duplicate."""
    times = sorted(run.alert_times)
    if truth.kind != "fall" or truth.fall_start_sec is None:
        return ClipOutcome(truth.clip_id, truth.kind, False, len(times), 0, None)

    onset = truth.fall_start_sec - ONSET_TOLERANCE_SEC
    early = [t for t in times if t < onset]
    after = [t for t in times if t >= onset]
    latency = None
    if after and truth.lying_start_sec is not None:
        latency = round(after[0] - truth.lying_start_sec, 3)
    return ClipOutcome(
        clip_id=truth.clip_id,
        kind=truth.kind,
        detected=bool(after),
        false_alarms=len(early),
        duplicates=max(0, len(after) - 1),
        latency_from_lying_sec=latency,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(truths: list[ClipTruth], runs: list[ClipRun]) -> dict:
    """Aggregate clip outcomes into recall, specificity, precision, F1 and false alarms/hour."""
    by_id = {t.clip_id: t for t in truths}
    outcomes = [match_clip(by_id[r.clip_id], r) for r in runs]

    falls = [o for o in outcomes if o.kind == "fall"]
    adls = [o for o in outcomes if o.kind == "adl"]
    tp = sum(o.detected for o in falls)
    false_alarms = sum(o.false_alarms for o in outcomes)
    adl_clean = sum(o.false_alarms == 0 for o in adls)

    recall = _ratio(tp, len(falls))
    precision = _ratio(tp, tp + false_alarms)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    # Negative footage: whole ADL clips plus the pre-onset part of fall clips. Padding counts
    # because the system was running and could alarm during it.
    negative_sec = 0.0
    for run in runs:
        truth = by_id[run.clip_id]
        if truth.kind == "adl":
            negative_sec += run.analysed_sec
        elif truth.fall_start_sec is not None:
            negative_sec += min(truth.fall_start_sec, run.analysed_sec)
    negative_hours = negative_sec / 3600.0

    latencies = sorted(o.latency_from_lying_sec for o in falls if o.latency_from_lying_sec is not None)
    hard_negatives = [o for o in adls if by_id[o.clip_id].has_lying]

    return {
        "fall_clips": len(falls),
        "adl_clips": len(adls),
        "true_positives": tp,
        "missed_falls": len(falls) - tp,
        "false_alarms": false_alarms,
        "duplicate_alerts": sum(o.duplicates for o in outcomes),
        "recall": recall,
        "recall_ci95": wilson_interval(tp, len(falls)),
        "specificity": _ratio(adl_clean, len(adls)),
        "specificity_ci95": wilson_interval(adl_clean, len(adls)),
        "precision": precision,
        "precision_ci95": wilson_interval(tp, tp + false_alarms),
        "f1": f1,
        "hard_negative_adls": len(hard_negatives),
        "hard_negative_false_alarm_clips": sum(o.false_alarms > 0 for o in hard_negatives),
        "negative_footage_min": round(negative_sec / 60.0, 2),
        "false_alarms_per_hour": (false_alarms / negative_hours) if negative_hours > 0 else None,
        "latency_from_lying_median_sec": latencies[len(latencies) // 2] if latencies else None,
        "latency_from_lying_max_sec": latencies[-1] if latencies else None,
        "missed_clip_ids": [o.clip_id for o in falls if not o.detected],
        "false_alarm_clip_ids": [o.clip_id for o in outcomes if o.false_alarms],
        "outcomes": [asdict(o) for o in outcomes],
    }


def _read_labels(csv_path: Path) -> dict[str, dict[int, int]]:
    """{sequence: {frame_number (1-based): label}} from a UR Fall feature CSV."""
    labels: dict[str, dict[int, int]] = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3 and row[0]:
                labels[row[0]][int(row[1])] = int(row[2])
    return labels


def load_urfall(data_dir: Path, source_fps: float = 30.0) -> list[ClipTruth]:
    """Build ground truth for every clip present in `data_dir`."""
    fall_labels = _read_labels(data_dir / "urfall-cam0-falls.csv")
    adl_labels = _read_labels(data_dir / "urfall-cam0-adls.csv")
    truths: list[ClipTruth] = []

    for index in range(1, URFALL_FALLS + 1):
        clip_id = f"fall-{index:02d}"
        path = data_dir / f"{clip_id}-cam0.mp4"
        frames = fall_labels.get(clip_id, {})
        if not path.exists() or not frames:
            continue
        falling = [f for f, label in frames.items() if label == 0]
        lying = [f for f, label in frames.items() if label == 1]
        truths.append(
            ClipTruth(
                clip_id=clip_id,
                path=str(path),
                kind="fall",
                # CSV frame numbers are 1-based; timestamps are (frame - 1) / fps.
                fall_start_sec=(min(falling) - 1) / source_fps if falling else None,
                lying_start_sec=(min(lying) - 1) / source_fps if lying else None,
                has_lying=bool(lying),
                crop_right_half=True,
            )
        )

    for index in range(1, URFALL_ADLS + 1):
        clip_id = f"adl-{index:02d}"
        path = data_dir / f"{clip_id}-cam0.mp4"
        if not path.exists():
            continue
        frames = adl_labels.get(clip_id, {})
        truths.append(
            ClipTruth(
                clip_id=clip_id,
                path=str(path),
                kind="adl",
                has_lying=any(label == 1 for label in frames.values()),
                crop_right_half=True,
            )
        )
    return truths


def parse_gmdcsa24_segments(classes: str) -> list[tuple[str, float, float]]:
    """`"Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]"` -> [("falling (sw)", 3.4, 6.0), ...]."""
    return [
        (name.strip().lower(), float(start), float(end))
        for name, start, end in _GMDCSA24_SEGMENT.findall(classes)
    ]


def load_gmdcsa24(data_dir: Path) -> list[ClipTruth]:
    """Ground truth from each subject's Fall.csv / ADL.csv (last column holds the segments)."""
    truths: list[ClipTruth] = []
    for subject in range(1, GMDCSA24_SUBJECTS + 1):
        for kind, folder in (("fall", "Fall"), ("adl", "ADL")):
            csv_path = data_dir / f"Subject {subject}" / f"{folder}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
            for row in rows[1:]:
                if not row or not row[0].strip():
                    continue
                path = data_dir / f"Subject {subject}" / folder / row[0].strip()
                if not path.exists():
                    continue
                # Whole row, not column 5: a comma inside the description shifts the columns,
                # and only the class column contains "[a to b]" segments.
                segments = parse_gmdcsa24_segments(",".join(row[1:]))
                falls = [start for name, start, _ in segments if name.startswith("fall")]
                if kind == "fall" and not falls:
                    # A fall row with no time range (Subject 4 / 15.mp4 publishes "Falling (FW)"
                    # bare) cannot be scored; guessing its onset would be inventing ground truth.
                    print(f"  skipping {path.relative_to(data_dir)}: fall has no annotated time range")
                    continue
                lying = any(name.startswith(_LYING_CLASSES) for name, _, _ in segments)
                truths.append(
                    ClipTruth(
                        clip_id=f"s{subject}-{kind}-{path.stem}",
                        path=str(path),
                        kind=kind,
                        fall_start_sec=min(falls) if kind == "fall" and falls else None,
                        # The CSVs mark one "Falling" segment through to the end of the clip,
                        # with no separate landing time; latency is measured from the onset.
                        lying_start_sec=min(falls) if kind == "fall" and falls else None,
                        has_lying=lying or kind == "fall",
                    )
                )
    return truths


def download_gmdcsa24(data_dir: Path) -> None:
    """Fetch every mp4 and csv in the GMDCSA24 GitHub repository, skipping files present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/{GMDCSA24_REPO}/git/trees/master?recursive=1"
    with urllib.request.urlopen(tree_url) as response:
        tree = json.load(response)["tree"]
    for entry in tree:
        name = entry["path"]
        if entry["type"] != "blob" or not name.endswith((".mp4", ".csv")):
            continue
        target = data_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name}")
        quoted = urllib.parse.quote(name)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{GMDCSA24_REPO}/master/{quoted}", target)


def download_urfall(data_dir: Path) -> None:
    """Fetch the cam0 mp4s and the two label CSVs, skipping files already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    names = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
    names += [f"fall-{i:02d}-cam0.mp4" for i in range(1, URFALL_FALLS + 1)]
    names += [f"adl-{i:02d}-cam0.mp4" for i in range(1, URFALL_ADLS + 1)]
    for name in names:
        target = data_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"  downloading {name}")
        urllib.request.urlretrieve(f"{URFALL_BASE_URL}/{name}", target)


DATASETS = {
    # name: (default data dir, loader, downloader, description)
    "urfall": (DEFAULT_DATA_DIR, load_urfall, download_urfall, "UR Fall Detection Dataset, cam0 RGB half (320x240)"),
    "gmdcsa24": (BENCHMARK_DATA_DIR / "gmdcsa24", load_gmdcsa24, download_gmdcsa24, "GMDCSA24 (4 subjects, day/night)"),
}


def _build_pipeline():
    """The detector + rule engine `MonitoringService` builds, minus overlays and zones."""
    from src.config import get_settings
    from src.services.monitoring import _detector_labels
    from src.vision.detector import IncidentDetector
    from src.vision.pipeline import VisionPipeline
    from src.vision.rule_engine import RuleEngine
    from src.vision.zone_config import ZoneManager

    settings = get_settings()
    return VisionPipeline(
        detector=IncidentDetector(
            model_path=settings.model_weights_path,
            conf_threshold=settings.detector_confidence,
            target_labels=_detector_labels(settings.detector_target_labels),
        ),
        rule_engine=RuleEngine(
            zone_manager=ZoneManager(),
            alert_cooldown_sec=settings.alert_cooldown_sec,
            abandon_sec=settings.abandon_sec,
            owner_radius_ratio=settings.owner_radius_ratio,
            stationary_tolerance_ratio=settings.stationary_tolerance_ratio,
        ),
        zones_path=None,
        draw_overlays=False,
    )


def run_clip(truth: ClipTruth, target_fps: float, pad_sec: float) -> ClipRun:
    """Feed one clip through a fresh pipeline at the live sampling rate."""
    import cv2

    pipeline = _build_pipeline()
    capture = cv2.VideoCapture(truth.path)
    if not capture.isOpened():
        raise ValueError(f"Could not decode {truth.path}")
    reported = capture.get(cv2.CAP_PROP_FPS)
    source_fps = reported if reported and math.isfinite(reported) and reported > 0 else 30.0
    step = 1.0 / target_fps if target_fps > 0 else 0.0

    camera_id = f"bench_{truth.clip_id}"
    alert_times: list[float] = []
    sampled = real_sampled = person_frames = 0
    next_sample = 0.0
    last_frame = None
    frame_index = 0

    def analyse(frame, timestamp: float, padding: bool = False) -> None:
        nonlocal sampled, real_sampled, person_frames
        _, result = pipeline.process_frame(frame, frame_id=sampled, camera_id=camera_id, timestamp=timestamp)
        sampled += 1
        # Real footage only: the padded tail repeats one frame and would swamp the fraction.
        if not padding:
            real_sampled += 1
            if any(d.label == "person" for d in result.detections):
                person_frames += 1
        alert_times.extend(round(timestamp, 3) for e in result.events if e.label == LABEL_FALL)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            if truth.crop_right_half:
                frame = frame[:, frame.shape[1] // 2 :]
            last_frame = frame
            if timestamp + 1e-9 >= next_sample:
                analyse(frame, timestamp)
                next_sample += step
    finally:
        capture.release()

    footage_sec = frame_index / source_fps
    end = footage_sec + pad_sec
    if last_frame is not None and pad_sec > 0:
        timestamp = max(next_sample, footage_sec)
        while timestamp < end:
            analyse(last_frame, timestamp, padding=True)
            timestamp += step or 1.0 / source_fps

    return ClipRun(
        clip_id=truth.clip_id,
        kind=truth.kind,
        footage_sec=round(footage_sec, 3),
        analysed_sec=round(max(footage_sec, end), 3),
        sampled_frames=sampled,
        person_frame_fraction=round(person_frames / real_sampled, 3) if real_sampled else 0.0,
        alert_times=alert_times,
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ci(interval) -> str:
    return "—" if not interval else f"[{interval[0] * 100:.1f}–{interval[1] * 100:.1f}%]"


def render_markdown(report: dict) -> str:
    s = report["scores"]
    cfg = report["config"]
    far = s["false_alarms_per_hour"]
    lines = [
        f"# Fall benchmark — {cfg['dataset']} ({report['label']})",
        "",
        f"Run {report['run_at'][:19]}Z · weights `{cfg['weights']}` · conf {cfg['detector_confidence']} · "
        f"{cfg['target_fps']} fps · pad {cfg['pad_sec']} s · fall dwell {cfg['fall_dwell_sec']} s",
        "",
        "| Metric | Value | 95% CI |",
        "| --- | --- | --- |",
        f"| Recall (falls detected) | {s['true_positives']}/{s['fall_clips']} = {_pct(s['recall'])} | {_ci(s['recall_ci95'])} |",
        f"| Specificity (ADL clips with no alert) | {_pct(s['specificity'])} | {_ci(s['specificity_ci95'])} |",
        f"| Precision | {_pct(s['precision'])} | {_ci(s['precision_ci95'])} |",
        f"| F1 | {_pct(s['f1'])} | |",
        f"| False alarms | {s['false_alarms']} over {s['negative_footage_min']} min of negative footage | |",
        f"| False alarms / hour | {'—' if far is None else f'{far:.1f}'} | |",
        f"| Hard-negative ADLs (lying on bed/floor) alarmed | {s['hard_negative_false_alarm_clips']}/{s['hard_negative_adls']} | |",
        f"| Alert latency after reaching the floor (median / max) | {s['latency_from_lying_median_sec']} s / {s['latency_from_lying_max_sec']} s | |",
        "",
        f"Missed falls: {', '.join(s['missed_clip_ids']) or 'none'}  ",
        f"Clips with false alarms: {', '.join(s['false_alarm_clip_ids']) or 'none'}",
        "",
        "## Per-clip",
        "",
        "| Clip | Footage s | Person in frame | Alerts at (s) | Outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    outcomes = {o["clip_id"]: o for o in s["outcomes"]}
    for run in report["runs"]:
        o = outcomes[run["clip_id"]]
        if o["kind"] == "fall":
            verdict = "TP" if o["detected"] else "MISS"
        else:
            verdict = "ok" if not o["false_alarms"] else "FALSE ALARM"
        if o["false_alarms"] and o["kind"] == "fall":
            verdict += f" + {o['false_alarms']} early FA"
        times = ", ".join(f"{t:.1f}" for t in run["alert_times"]) or "—"
        lines.append(
            f"| {run['clip_id']} | {run['footage_sec']:.1f} | {run['person_frame_fraction'] * 100:.0f}% | {times} | {verdict} |"
        )
    lines += [
        "",
        "Protocol and limitations: see the module docstring of `src/benchmark/run_fall_benchmark.py`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="urfall")
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to data/benchmark/<dataset>.")
    parser.add_argument("--download", action="store_true", help="Fetch the dataset's clips and labels, then exit.")
    parser.add_argument("--fps", type=float, default=5.0, help="Analysis rate (CAMERA_TARGET_FPS default: 5).")
    parser.add_argument("--pad-sec", type=float, default=4.0, help="Seconds of last-frame hold appended to every clip.")
    parser.add_argument("--label", default="padded", help="Suffix for the result files.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N fall and N ADL clips (smoke test).")
    args = parser.parse_args()

    default_dir, loader, downloader, description = DATASETS[args.dataset]
    data_dir = args.data_dir or default_dir

    if args.download:
        downloader(data_dir)
        return 0

    truths = loader(data_dir)
    if not truths:
        print(f"No {args.dataset} clips under {data_dir}. Run with --download first.")
        return 1
    if args.limit:
        truths = [t for t in truths if t.kind == "fall"][: args.limit] + [t for t in truths if t.kind == "adl"][: args.limit]

    from src.config import get_settings
    from src.vision.rule_engine import RuleEngine

    settings = get_settings()
    runs: list[ClipRun] = []
    started = time.perf_counter()
    for index, truth in enumerate(truths, 1):
        run = run_clip(truth, args.fps, args.pad_sec)
        runs.append(run)
        print(
            f"  [{index:>3}/{len(truths)}] {truth.clip_id:<12} person {run.person_frame_fraction * 100:5.1f}%  "
            f"alerts {run.alert_times or '-'}"
        )

    report = {
        "label": args.label,
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "config": {
            "dataset": description,
            "weights": settings.model_weights_path,
            "detector_confidence": settings.detector_confidence,
            "target_fps": args.fps,
            "pad_sec": args.pad_sec,
            "fall_dwell_sec": inspect.signature(RuleEngine).parameters["fall_dwell_sec"].default,
            "alert_cooldown_sec": settings.alert_cooldown_sec,
            "onset_tolerance_sec": ONSET_TOLERANCE_SEC,
        },
        "scores": score(truths, runs),
        "runs": [asdict(r) for r in runs],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"fall-benchmark-{args.dataset}-{args.label}.json"
    md_path = RESULTS_DIR / f"fall-benchmark-{args.dataset}-{args.label}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    s = report["scores"]
    print()
    print(f"  Recall      {_pct(s['recall'])} {_ci(s['recall_ci95'])}")
    print(f"  Specificity {_pct(s['specificity'])} {_ci(s['specificity_ci95'])}")
    print(f"  Precision   {_pct(s['precision'])} {_ci(s['precision_ci95'])}")
    print(f"  F1          {_pct(s['f1'])}")
    print(f"  FA / hour   {s['false_alarms_per_hour']}")
    print(f"  Wrote {json_path} and {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
