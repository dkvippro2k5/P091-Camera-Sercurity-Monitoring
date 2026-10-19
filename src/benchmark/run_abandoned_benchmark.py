"""Event-level abandoned-object benchmark on ABODA, AVSS 2007 and CAVIAR.

Datasets (`--dataset`):

`aboda` - ABODA (Abandoned Object Dataset, Lin et al.), https://github.com/kevinlin311tw/ABODA
  - 11 clips, 640/720x480, ~30 fps, indoor and outdoor, day and night, sudden lighting changes,
    one crowded station hall. No licence is stated and the repository ships no annotations, so
    the videos are not redistributed here: put `video1.mp4` ... `video11.mp4` (re-encoded from
    the published .avi with `src.benchmark.video_convert`) in `data/benchmark/aboda/`.

`avss2007` - i-LIDS AVSS 2007 abandoned-baggage clips (EASY / MEDIUM / HARD), a London
  Underground platform, 720x576, ~3 min each, with crowds walking past. Videos and ViPER ground
  truth come from the material of Luna et al., "Abandoned Object Detection in Video-Surveillance:
  Survey and Comparison" (Sensors 2018), http://www-vpu.eps.uam.es/publications/AODsurvey/ -
  not redistributed here; extract `datasets/AVSS2007/*` into `data/benchmark/avss2007/` and
  re-encode the .mpg files. The box, the drop and the removal come from that ground truth
  (its frame numbers are on a 30 fps clock). Its own "abandoned" frame applies a 30 s
  stationary rule; the moment the owner walked away (`AVSS_OWNER_LEFT_SEC`) is ours, read off
  the video like the ABODA labels.

`caviar-inria` - CAVIAR INRIA lobby (CC BY-SA), https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/,
  384x288, 25 fps. Four clips label a "leaving object" (LeftBag, LeftBag_AtChair, LeftBox,
  LeftBag_PickedUp); its first and last labelled frames are the drop and the removal, and the
  owner - the person nearest the object when it appears - is taken to have left once they stay
  farther than `CAVIAR_OWNER_GONE_RATIO` of the frame width for a second. The other lobby clips
  (walking, browsing, meeting, fighting) contain no abandoned object and measure false alarms.

**ABODA labels are ours.** `ABODA_EVENTS` was written on 2026-10-19 by stepping through each clip
(contact sheets at 0.5-3 s) and locating each object by differencing the scene before the drop
against after the owner left. For each abandonment:

  - `box`: the object, (x, y, w, h) in pixels.
  - `placed`: when it was put down.
  - `unattended`: when the owner walked away from it — the moment the rule's "no owner within
    radius" should start counting. Accurate to about a second.
  - `removed`: when someone picked it up again, or None if it stays to the end.

`video11` (crowded hall) holds no abandonment: a bag is put down at ~36 s and the man who put it
there stands beside it to the end. It is kept as a hard negative — "attended luggage in a crowd".

**Protocol — read before quoting a number.**

- An `abandoned_object` alert whose box centre is within `MATCH_RADIUS_PX` (or the object's own
  diagonal, if larger) of a labelled object is *on* that object.
- On an abandoned object, the first alert between `unattended - EARLY_TOLERANCE_SEC` and
  `removed` (or the clip end) is a hit; later ones are duplicates. An alert on it before that
  window fired while the owner was still beside it: a false alarm ("early").
- Any alert on nothing labelled, or on the attended bag in `video11`, is a false alarm.
- Sampling at `--fps` (default 5) on the video clock, fresh pipeline per clip, the
  `ABANDON_SEC` / `OWNER_RADIUS_RATIO` / `STATIONARY_TOLERANCE_RATIO` settings in effect.

Usage:
    python -m src.benchmark.run_abandoned_benchmark --dataset all --label baseline
"""

import argparse
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.benchmark.run_fall_benchmark import BENCHMARK_DATA_DIR, RESULTS_DIR, _ci, _pct, _ratio, wilson_interval

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_DATA_DIR = BENCHMARK_DATA_DIR / "aboda"
CAVIAR_OWNER_GONE_RATIO = 0.25
LABEL_ABANDONED = "abandoned_object"
MATCH_RADIUS_PX = 60.0
EARLY_TOLERANCE_SEC = 3.0


@dataclass(frozen=True)
class ObjectLabel:
    box: tuple[int, int, int, int]
    placed: float
    unattended: float | None  # None: attended throughout (a hard negative)
    removed: float | None = None
    note: str = ""


ABODA_EVENTS: dict[str, list[ObjectLabel]] = {
    "video1": [ObjectLabel((129, 323, 59, 57), 43.0, 48.5, None, "corridor, backpack on the floor, two people walk off")],
    "video2": [ObjectLabel((455, 76, 23, 36), 69.5, 72.5, None, "outdoor pavement, bag by a pole, far from camera")],
    "video3": [ObjectLabel((214, 216, 44, 31), 56.5, 68.0, None, "outdoor pavement, owner lingers ~10 s first")],
    "video4": [ObjectLabel((449, 369, 55, 43), 14.0, 20.0, None, "outdoor pavement, bag against a utility box")],
    "video5": [
        ObjectLabel((135, 199, 35, 31), 31.0, 67.0, None, "night lobby (IR), owner stands by the box until 67 s")
    ],
    "video6": [
        ObjectLabel((186, 307, 46, 29), 115.5, 118.0, 132.0, "night lobby (IR), picked up by someone else"),
        ObjectLabel((149, 314, 41, 37), 180.5, 184.0, 207.0, "after a lighting change, picked up by someone else"),
    ],
    "video7": [ObjectLabel((125, 87, 57, 70), 119.0, 128.5, None, "classroom, shopping bag on a table, light switch earlier")],
    "video8": [ObjectLabel((100, 200, 185, 50), 118.0, 123.5, None, "dark classroom (IR), long case on a table")],
    "video9": [ObjectLabel((415, 136, 67, 83), 32.0, 37.0, None, "classroom, shopping bag on a table")],
    "video10": [ObjectLabel((155, 186, 98, 67), 41.0, 46.5, None, "classroom, holdall on a table")],
    "video11": [ObjectLabel((400, 232, 25, 22), 36.0, None, None, "crowded hall; owner stands beside the bag throughout")],
}


@dataclass
class ClipTruth:
    clip_id: str
    path: str
    objects: list[ObjectLabel]


@dataclass
class Alert:
    time_sec: float
    center: tuple[float, float]
    label: str | None


@dataclass
class ClipRun:
    clip_id: str
    footage_sec: float
    sampled_frames: int
    alerts: list[Alert] = field(default_factory=list)


@dataclass
class ClipOutcome:
    clip_id: str
    abandonments: int
    detected: int
    duplicates: int
    false_alarms: int
    latencies_sec: list[float]
    # "<time>s <early|attended|unlabelled>" per false alarm.
    false_alarm_details: list[str]


def load_aboda(data_dir: Path) -> list[ClipTruth]:
    return [
        ClipTruth(clip, str(data_dir / f"{clip}.mp4"), objects)
        for clip, objects in ABODA_EVENTS.items()
        if (data_dir / f"{clip}.mp4").exists()
    ]


# Owner walking away, in seconds, read off each clip. EASY and HARD agree with the ground
# truth's "abandoned" frame; in MEDIUM the woman who set the bag down stands a couple of metres
# behind it until 84 s, ~7 s after that frame (which is just the drop plus 30 s), and comes back
# for it at ~150 s. The man beside her with a suitcase is not part of the event.
AVSS_OWNER_LEFT_SEC = {"EASY": 96.0, "MEDIUM": 84.0, "HARD": 98.6}
_AVSS_GT_FPS = 30.0
_VIPER_OBJECT = re.compile(
    r'<object framespan="(?P<span>[^"]+)" id="[^"]*" name="(?P<name>\w+)">.*?<data:bbox (?P<bbox>[^/]*)/>', re.S
)


def parse_viper_objects(xml_path: Path) -> dict[str, tuple[tuple[int, int, int, int], int, int]]:
    """{descriptor name: (box, first frame, last frame)} from a ViPER file with one object each.

    Read with regular expressions: the published AVSS files are not well-formed XML (a
    mismatched closing tag in the config block), so an XML parser rejects them.
    """
    events = {}
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    for match in _VIPER_OBJECT.finditer(text):
        first, last = (int(v) for v in match["span"].strip().split(":"))
        attrs = dict(re.findall(r'(\w+)="(\d+)"', match["bbox"]))
        events[match["name"]] = (tuple(int(attrs[k]) for k in ("x", "y", "width", "height")), first, last)
    return events


def load_avss2007(data_dir: Path) -> list[ClipTruth]:
    import cv2

    truths = []
    for level, owner_left in AVSS_OWNER_LEFT_SEC.items():
        video, labels = data_dir / f"AVSSS07_{level}.mp4", data_dir / f"AVSSS07_{level}.txt"
        if not (video.exists() and labels.exists()):
            continue
        box, put, removed = parse_viper_objects(labels)["PutObject"]
        capture = cv2.VideoCapture(str(video))
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / (capture.get(cv2.CAP_PROP_FPS) or 25.0)
        capture.release()
        removed_sec = removed / _AVSS_GT_FPS
        label = ObjectLabel(
            box,
            put / _AVSS_GT_FPS,
            owner_left,
            removed_sec if removed_sec < duration - 1.0 else None,
            f"AVSS 2007 {level}",
        )
        truths.append(ClipTruth(f"avss-{level.lower()}", str(video), [label]))
    return truths


CAVIAR_INRIA_CLIPS = (
    "LeftBag", "LeftBag_AtChair", "LeftBox", "LeftBag_PickedUp",
    "Walk1", "Walk2", "Walk3", "Browse1", "Browse2", "Browse3", "Browse4",
    "Browse_WhileWaiting1", "Browse_WhileWaiting2", "Rest_InChair", "Rest_SlumpOnFloor",
    "Rest_WiggleOnFloor", "Rest_FallOnFloor", "Meet_WalkTogether1", "Meet_WalkTogether2",
    "Meet_WalkSplit", "Meet_Split_3rdGuy", "Meet_Crowd", "Fight_RunAway1", "Fight_RunAway2",
    "Fight_Chase",
)  # fmt: skip


def caviar_left_objects(xml_path: Path, fps: float = 25.0, frame_width: int = 384) -> list[ObjectLabel]:
    """Abandonments in a CAVIAR file: each "leaving object" and when its owner walked off."""
    tracks: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    roles: dict[str, str] = {}
    last_frame = 0
    for frame in ET.parse(xml_path).getroot().iter("frame"):
        number = int(frame.get("number"))
        last_frame = max(last_frame, number)
        for obj in frame.iter("object"):
            box = obj.find("box")
            tracks.setdefault(obj.get("id"), {})[number] = tuple(float(box.get(k)) for k in ("xc", "yc", "w", "h"))
            roles.setdefault(obj.get("id"), (obj.findtext(".//role") or "").strip())

    labels = []
    gone_px = CAVIAR_OWNER_GONE_RATIO * frame_width
    for object_id, track in tracks.items():
        if roles[object_id] != "leaving object":
            continue
        first, last = min(track), max(track)
        # The box once it has settled: the drop frame often catches it mid-air.
        xc, yc, w, h = track[sorted(track)[len(track) // 2]]

        def distance(person: str, frame: int, centre=(xc, yc)) -> float:
            box = tracks[person].get(frame)
            return math.inf if box is None else math.dist(centre, box[:2])

        window = range(first - int(fps), first + int(fps) + 1)
        people = [p for p in tracks if roles[p] != "leaving object" and any(f in tracks[p] for f in window)]
        owner = min(people, key=lambda p: min(distance(p, f) for f in window), default=None)
        left = first
        if owner is not None:
            far_since = None
            for frame in range(first, last + 1):
                if distance(owner, frame) <= gone_px:
                    far_since = None
                    continue
                far_since = frame if far_since is None else far_since
                if frame - far_since >= fps:
                    break
            left = far_since if far_since is not None else last
        labels.append(
            ObjectLabel(
                (int(xc - w / 2), int(yc - h / 2), int(w), int(h)),
                first / fps,
                left / fps,
                last / fps if last < last_frame - fps else None,
                f"CAVIAR {xml_path.stem}",
            )
        )
    return labels


def load_caviar_inria(data_dir: Path) -> list[ClipTruth]:
    return [
        ClipTruth(f"caviar-{clip}", str(data_dir / f"{clip}.mp4"), caviar_left_objects(data_dir / f"{clip}.xml"))
        for clip in CAVIAR_INRIA_CLIPS
        if (data_dir / f"{clip}.mp4").exists() and (data_dir / f"{clip}.xml").exists()
    ]


DATASETS = {
    "aboda": (DEFAULT_DATA_DIR, load_aboda, "ABODA, 11 clips, own labels"),
    "avss2007": (BENCHMARK_DATA_DIR / "avss2007", load_avss2007, "AVSS 2007 (i-LIDS) EASY/MEDIUM/HARD"),
    "caviar-inria": (
        BENCHMARK_DATA_DIR / "caviar" / "inria",
        load_caviar_inria,
        "CAVIAR INRIA lobby: 4 left-object clips + 21 without one",
    ),
}


def _on_object(alert: Alert, obj: ObjectLabel) -> bool:
    x, y, w, h = obj.box
    radius = max(MATCH_RADIUS_PX, math.hypot(w, h))
    return math.dist(alert.center, (x + w / 2.0, y + h / 2.0)) <= radius


def match_clip(truth: ClipTruth, run: ClipRun) -> ClipOutcome:
    hit_at: dict[int, float] = {}
    duplicates = false_alarms = 0
    details: list[str] = []
    end = run.footage_sec
    for alert in sorted(run.alerts, key=lambda a: a.time_sec):
        targets = [i for i, obj in enumerate(truth.objects) if _on_object(alert, obj)]
        # Nearest object first when two labelled objects are close together.
        targets.sort(key=lambda i: math.dist(alert.center, truth.objects[i].box[:2]))
        verdict = "unlabelled"
        for i in targets:
            obj = truth.objects[i]
            if obj.unattended is None:
                verdict = "attended"
                break
            if alert.time_sec < obj.unattended - EARLY_TOLERANCE_SEC:
                verdict = "early"
                break
            if alert.time_sec <= (obj.removed if obj.removed is not None else end) + 1.0:
                verdict = "duplicate" if i in hit_at else "hit"
                if verdict == "hit":
                    hit_at[i] = alert.time_sec
                break
            verdict = "after removal"
        if verdict == "duplicate":
            duplicates += 1
        elif verdict != "hit":
            false_alarms += 1
            details.append(f"{alert.time_sec:.1f}s {verdict} ({alert.label})")
    return ClipOutcome(
        clip_id=truth.clip_id,
        abandonments=sum(obj.unattended is not None for obj in truth.objects),
        detected=len(hit_at),
        duplicates=duplicates,
        false_alarms=false_alarms,
        latencies_sec=[round(t - truth.objects[i].unattended, 2) for i, t in sorted(hit_at.items())],
        false_alarm_details=details,
    )


def score(truths: list[ClipTruth], runs: list[ClipRun]) -> dict:
    by_id = {t.clip_id: t for t in truths}
    outcomes = [match_clip(by_id[r.clip_id], r) for r in runs]
    events = sum(o.abandonments for o in outcomes)
    detected = sum(o.detected for o in outcomes)
    false_alarms = sum(o.false_alarms for o in outcomes)
    hours = sum(r.footage_sec for r in runs) / 3600.0
    recall = _ratio(detected, events)
    precision = _ratio(detected, detected + false_alarms)
    latencies = sorted(x for o in outcomes for x in o.latencies_sec)
    return {
        "clips": len(outcomes),
        "footage_min": round(hours * 60, 2),
        "abandonments": events,
        "detected": detected,
        "false_alarms": false_alarms,
        "duplicate_alerts": sum(o.duplicates for o in outcomes),
        "recall": recall,
        "recall_ci95": wilson_interval(detected, events),
        "precision": precision,
        "precision_ci95": wilson_interval(detected, detected + false_alarms),
        "f1": (2 * precision * recall / (precision + recall)) if recall and precision else None,
        "false_alarms_per_hour": false_alarms / hours if hours else None,
        "latency_median_sec": latencies[len(latencies) // 2] if latencies else None,
        "latency_max_sec": latencies[-1] if latencies else None,
        "outcomes": [asdict(o) for o in outcomes],
    }


def run_clip(truth: ClipTruth, target_fps: float) -> ClipRun:
    import cv2

    from src.benchmark.run_fall_benchmark import _build_pipeline

    pipeline = _build_pipeline()
    capture = cv2.VideoCapture(truth.path)
    if not capture.isOpened():
        raise ValueError(f"Could not decode {truth.path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    run = ClipRun(clip_id=truth.clip_id, footage_sec=0.0, sampled_frames=0)
    step = 1.0 / target_fps
    next_sample = 0.0
    frame_index = 0
    camera_id = f"bench_{truth.clip_id}"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            if timestamp + 1e-9 < next_sample:
                continue
            next_sample += step
            _, result = pipeline.process_frame(frame, frame_id=run.sampled_frames, camera_id=camera_id, timestamp=timestamp)
            run.sampled_frames += 1
            for event in result.events:
                if event.label == LABEL_ABANDONED:
                    center = ((event.bbox.xmin + event.bbox.xmax) / 2.0, (event.bbox.ymin + event.bbox.ymax) / 2.0)
                    # The reason names the detector class: "<label> (<track>) unattended ...".
                    label = (event.reason or "").split(" ", 1)[0] or None
                    run.alerts.append(Alert(round(timestamp, 3), center, label))
    finally:
        capture.release()
    run.footage_sec = round(frame_index / source_fps, 3)
    return run


def render_markdown(report: dict) -> str:
    s = report["scores"]
    cfg = report["config"]
    far = s["false_alarms_per_hour"]
    lines = [
        f"# Abandoned-object benchmark — {cfg['dataset']} ({report['label']})",
        "",
        f"Run {report['run_at'][:19]}Z · weights `{cfg['weights']}` · conf {cfg['detector_confidence']} · "
        f"{cfg['target_fps']} fps · abandon {cfg['abandon_sec']} s · owner radius {cfg['owner_radius_ratio']} × width",
        "",
        "| Metric | Value | 95% CI |",
        "| --- | --- | --- |",
        f"| Recall (abandonments alerted) | {s['detected']}/{s['abandonments']} = {_pct(s['recall'])} | {_ci(s['recall_ci95'])} |",
        f"| Precision | {_pct(s['precision'])} | {_ci(s['precision_ci95'])} |",
        f"| F1 | {_pct(s['f1'])} | |",
        f"| False alarms | {s['false_alarms']} over {s['footage_min']} min | |",
        f"| False alarms / hour | {'—' if far is None else f'{far:.1f}'} | |",
        f"| Duplicate alerts | {s['duplicate_alerts']} | |",
        f"| Alert latency after the owner left (median / max) | {s['latency_median_sec']} s / {s['latency_max_sec']} s | |",
        "",
        "## Per-clip",
        "",
        "| Clip | Abandonments | Detected | Latency s | False alarms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for o in s["outcomes"]:
        lines.append(
            f"| {o['clip_id']} | {o['abandonments']} | {o['detected']} | {o['latencies_sec'] or '—'} | "
            f"{'; '.join(o['false_alarm_details']) or '—'} |"
        )
    lines += ["", "Labels and protocol: see the module docstring of `src/benchmark/run_abandoned_benchmark.py`.", ""]
    return "\n".join(lines)


def run_dataset(name: str, data_dir: Path, fps: float, label: str) -> int:
    _, loader, description = DATASETS[name]
    truths = loader(data_dir)
    if not truths:
        print(f"No {name} clips under {data_dir}; see the module docstring for where they come from.")
        return 1

    from src.config import get_settings

    settings = get_settings()
    runs = []
    started = time.perf_counter()
    for index, truth in enumerate(truths, 1):
        run = run_clip(truth, fps)
        runs.append(run)
        print(f"  [{index:>2}/{len(truths)}] {truth.clip_id:<26} alerts {[(a.time_sec, a.label) for a in run.alerts] or '-'}")

    report = {
        "label": label,
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "config": {
            "dataset": description,
            "weights": settings.model_weights_path,
            "detector_confidence": settings.detector_confidence,
            "target_fps": fps,
            "abandon_sec": settings.abandon_sec,
            "owner_radius_ratio": settings.owner_radius_ratio,
            "stationary_tolerance_ratio": settings.stationary_tolerance_ratio,
            "alert_cooldown_sec": settings.alert_cooldown_sec,
            "match_radius_px": MATCH_RADIUS_PX,
            "early_tolerance_sec": EARLY_TOLERANCE_SEC,
        },
        "labels": {t.clip_id: [asdict(o) for o in t.objects] for t in truths},
        "scores": score(truths, runs),
        "runs": [asdict(r) for r in runs],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"abandoned-benchmark-{name}-{label}.json"
    md_path = RESULTS_DIR / f"abandoned-benchmark-{name}-{label}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    s = report["scores"]
    print()
    print(
        f"  {name}: recall {_pct(s['recall'])} {_ci(s['recall_ci95'])} ({s['detected']}/{s['abandonments']}), "
        f"precision {_pct(s['precision'])}, false alarms {s['false_alarms']} ({s['false_alarms_per_hour']:.1f}/h)"
    )
    print(f"  Wrote {json_path} and {md_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="aboda")
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to the dataset's own folder.")
    parser.add_argument("--fps", type=float, default=5.0, help="Analysis rate (CAMERA_TARGET_FPS default: 5).")
    parser.add_argument("--label", default="baseline", help="Suffix for the result files.")
    args = parser.parse_args()

    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    status = 0
    for name in names:
        status |= run_dataset(name, args.data_dir or DATASETS[name][0], args.fps, args.label)
    return status


if __name__ == "__main__":
    sys.exit(main())
