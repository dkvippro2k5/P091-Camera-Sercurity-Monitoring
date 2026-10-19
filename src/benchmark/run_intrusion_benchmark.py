"""Event-level intrusion benchmark on public clips with every person's box labelled.

None of these datasets has intrusion labels, so they are derived. A restricted zone is drawn
once per camera (`SCENES`), and a person is *inside* when the bottom-centre of their labelled
box — the same foot point the rule uses — is inside the polygon. The labels therefore test
detection, tracking and the rule's geometry against hand-drawn boxes; they say nothing about
whether a zone is well chosen. Each zone was placed where labelled people both enter and walk
past along its edge, after plotting every labelled foot track for that camera.

Datasets:
  - `caviar-front` / `caviar-corridor`: CAVIAR Lisbon shopping centre (EC Funded CAVIAR
    project/IST 2001 37540, CC BY-SA), https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/ —
    the same 26 clips seen by two cameras, 384x288, 25 fps. Front: the zone is the shop floor
    behind the entrance. Corridor: the zone is the strip along the shop fronts on the left wall.
  - `mot15`: three static-camera sequences of MOT15 (https://motchallenge.net, CC BY-NC-SA):
    PETS09-S2L1 (outdoor campus, 7 fps), Venice-2 (crowded square, 1080p) and ADL-Rundle-6
    (busy pavement in front of shops, 1080p). Short (~2.5 min in all) but crowded and outdoor.

Clips are kept as H.264 .mp4 — `--download` re-encodes CAVIAR's MPEG-2 and MOT's JPEG frames,
checking every frame survives, because the labels index frames.

**Protocol — read before quoting a number.**

- Distances are in pixels of a 288-line frame and scaled by frame height for other sizes.
- Ground-truth entry: a labelled person's foot point inside the zone for at least
  `MIN_INSIDE_SEC`. Leaving for at least `REENTRY_GAP_SEC` and coming back is a new entry.
  Someone already inside when the clip starts counts as entering on their first frame.
- Boundary walker: a labelled person who never entered but whose foot point came within
  `NEAR_BOUNDARY_PX` of the zone — the "walking along the boundary" case.
- Matching: an `intrusion` alert matches an entry when it falls between `EARLY_TOLERANCE_SEC`
  before the entry and `LATE_TOLERANCE_SEC` after the person left, and the alert box's foot
  point is within `MATCH_RADIUS_PX` of that person's labelled foot point. Each entry matches
  once; a further alert on an already matched entry is a duplicate (usually a track ID change
  inside the zone). Any other alert is a false alarm.
- MOT15 labels only part of each crowd (people far away are often unlabelled), so its zones
  sit in the near field; an alert on an unlabelled person there still counts as false.
- Sampling at `--fps` (default 5) on the video clock; fresh detector, tracker and rule engine
  per clip. The fall and abandoned-object rules run too but are ignored.

Usage:
    python -m src.benchmark.run_intrusion_benchmark --dataset all --download   # ~2 GB, once
    python -m src.benchmark.run_intrusion_benchmark --dataset all --label baseline
"""

import argparse
import json
import math
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
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

CAVIAR_BASE_URL = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2"
MOT15_URL = "https://motchallenge.net/data/MOT15.zip"
# Clip name -> ground-truth file stem without its view letter ("f" front, "c" corridor).
CAVIAR_CLIPS = {
    "WalkByShop1": "wbs1gt",
    "EnterExitCrossingPaths1": "eecp1gt",
    "EnterExitCrossingPaths2": "eecp2gt",
    "OneLeaveShop1": "ols1gt",
    "OneLeaveShop2": "ols2gt",
    "OneLeaveShopReenter1": "olsr1gt",
    "OneLeaveShopReenter2": "olsr2gt",
    "OneShopOneWait1": "osow1gt",
    "OneShopOneWait2": "osow2gt",
    "OneStopEnter1": "ose1gt",
    "OneStopEnter2": "ose2gt",
    "OneStopMoveEnter1": "osme1gt",
    "OneStopMoveEnter2": "osme2gt",
    "OneStopMoveNoEnter1": "osmne1gt",
    "OneStopMoveNoEnter2": "osmne2gt",
    "OneStopNoEnter1": "osne1gt",
    "OneStopNoEnter2": "osne2gt",
    "ShopAssistant1": "sa1gt",
    "ShopAssistant2": "sa2gt",
    "ThreePastShop1": "3ps1gt",
    "ThreePastShop2": "3ps2gt",
    "TwoEnterShop1": "2es1gt",
    "TwoEnterShop2": "2es2gt",
    "TwoEnterShop3": "2es3gt",
    "TwoLeaveShop1": "2ls1gt",
    "TwoLeaveShop2": "2ls2gt",
}
LABEL_INTRUSION = "intrusion"
ZONE_NAME = "restricted"
REFERENCE_HEIGHT = 288.0

# Front view: the shop floor behind the right-hand glass entrance. The threshold is y~148:
# labelled "shop enter" foot points cluster at 140-160 and corridor walkers at 152-185, so a
# walker along the shop front passes 5-35 px below the zone edge.
SHOP_ZONE_PX = [(212, 60), (335, 60), (335, 146), (212, 146)]
# Corridor view: the strip along the shop fronts on the left wall, covering both entrances
# (labelled "shop enter" points at x 50-80, y 115-135 and 165-205); corridor walkers pass just
# to its right. 101 entries and 23 boundary walkers over the 26 clips.
CORRIDOR_ZONE_PX = [(40, 112), (75, 112), (75, 212), (40, 212)]
# MOT15, per sequence, in that sequence's pixels: where labelled tracks enter and pass by.
MOT15_ZONES_PX = {
    "PETS09-S2L1": [(440, 240), (700, 240), (700, 330), (440, 330)],  # road right of the lamp
    "Venice-2": [(900, 650), (1500, 650), (1500, 900), (900, 900)],  # square, near field
    "ADL-Rundle-6": [(800, 700), (1400, 700), (1400, 860), (800, 860)],  # pavement by the shops
}

MIN_INSIDE_SEC = 0.4
REENTRY_GAP_SEC = 1.0
NEAR_BOUNDARY_PX = 15.0
EARLY_TOLERANCE_SEC = 1.0
LATE_TOLERANCE_SEC = 1.0
MATCH_RADIUS_PX = 40.0


@dataclass
class Entry:
    """One ground-truth stay in the zone by one labelled person."""

    object_id: str
    start_sec: float
    end_sec: float


@dataclass
class ClipTruth:
    clip_id: str
    path: str
    duration_sec: float
    entries: list[Entry]
    # object_id -> {frame: (foot_x, foot_y)}
    feet: dict[str, dict[int, tuple[float, float]]]
    boundary_walkers: list[str]
    fps: float = 25.0
    frame_size: tuple[int, int] = (384, 288)
    zone_px: list[tuple[float, float]] = field(default_factory=lambda: list(SHOP_ZONE_PX))


@dataclass
class Alert:
    time_sec: float
    foot: tuple[float, float]
    track_id: str | None
    # Which test fired: the foot point itself, or the lower-body overlap fallback.
    via: str = "foot"


@dataclass
class ClipRun:
    clip_id: str
    footage_sec: float
    sampled_frames: int
    alerts: list[Alert] = field(default_factory=list)


@dataclass
class ClipOutcome:
    clip_id: str
    entries: int
    detected: int
    duplicates: int
    false_alarms: int
    boundary_walkers: int
    boundary_walkers_alarmed: int
    latencies_sec: list[float]
    missed_entries: list[str]
    # "<time>s <via> <walker|other>" for each false alarm.
    false_alarm_details: list[str] = field(default_factory=list)


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _distance_to_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> float:
    if _point_in_polygon(x, y, polygon):
        return 0.0
    best = math.inf
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


def derive_entries(
    feet: dict[str, dict[int, tuple[float, float]]],
    polygon: list[tuple[float, float]],
    fps: float = 25.0,
    near_px: float = NEAR_BOUNDARY_PX,
) -> tuple[list[Entry], list[str]]:
    """Ground-truth entries and boundary walkers from per-frame labelled foot points."""
    entries: list[Entry] = []
    walkers: list[str] = []
    for object_id, track in feet.items():
        frames = sorted(track)
        runs: list[list[int]] = []  # [first inside frame, last inside frame]
        for frame in frames:
            if not _point_in_polygon(*track[frame], polygon):
                continue
            if runs and (frame - runs[-1][1]) / fps < REENTRY_GAP_SEC:
                runs[-1][1] = frame
            else:
                runs.append([frame, frame])
        stays = [r for r in runs if (r[1] - r[0] + 1) / fps >= MIN_INSIDE_SEC]
        entries += [Entry(object_id, first / fps, last / fps) for first, last in stays]
        if not stays and min(_distance_to_polygon(*track[f], polygon) for f in frames) <= near_px:
            walkers.append(object_id)
    entries.sort(key=lambda e: e.start_sec)
    return entries, walkers


def _truth(clip_id, video: Path, feet, frames: int, fps: float, size, zone_px) -> ClipTruth:
    scale = size[1] / REFERENCE_HEIGHT
    entries, walkers = derive_entries(feet, zone_px, fps, NEAR_BOUNDARY_PX * scale)
    return ClipTruth(clip_id, str(video), frames / fps, entries, feet, walkers, fps, size, list(zone_px))


# --- CAVIAR -------------------------------------------------------------------------------


def parse_caviar_feet(xml_path: Path) -> tuple[dict[str, dict[int, tuple[float, float]]], int]:
    """{object id: {frame: bottom-centre of the labelled box}}, and the frame count."""
    feet: dict[str, dict[int, tuple[float, float]]] = {}
    frames = 0
    for frame in ET.parse(xml_path).getroot().iter("frame"):
        number = int(frame.get("number"))
        frames = max(frames, number + 1)
        for obj in frame.iter("object"):
            box = obj.find("box")
            xc, yc, h = (float(box.get(k)) for k in ("xc", "yc", "h"))
            feet.setdefault(obj.get("id"), {})[number] = (xc, yc + h / 2.0)
    return feet, frames


def _caviar_loader(view: str, zone_px) -> Callable[[Path], list[ClipTruth]]:
    def load(data_dir: Path) -> list[ClipTruth]:
        truths = []
        for clip in CAVIAR_CLIPS:
            video, labels = data_dir / f"{clip}{view}.mp4", data_dir / f"{clip}{view}.xml"
            if video.exists() and labels.exists():
                feet, frames = parse_caviar_feet(labels)
                truths.append(_truth(clip, video, feet, frames, 25.0, (384, 288), zone_px))
        return truths

    return load


def _caviar_downloader(view: str) -> Callable[[Path], None]:
    def download(data_dir: Path) -> None:
        from src.benchmark.video_convert import to_mp4

        data_dir.mkdir(parents=True, exist_ok=True)
        for clip, stem in CAVIAR_CLIPS.items():
            remote_stem = ("f" if view == "front" else "c") + stem
            for remote, local in ((f"{remote_stem}.xml", f"{clip}{view}.xml"), (f"{clip}{view}.mpg", f"{clip}{view}.mpg")):
                target = data_dir / local
                if target.exists() or target.with_suffix(".mp4").exists():
                    continue
                url = f"{CAVIAR_BASE_URL}/{clip}{view}/{remote}"
                print(f"  {url}")
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    target.write_bytes(response.read())
            source = data_dir / f"{clip}{view}.mpg"
            if source.exists():
                to_mp4(source)
                source.unlink()

    return download


# --- MOT15 --------------------------------------------------------------------------------


def parse_mot_feet(gt_path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """MOT gt.txt (frame 1-based, id, left, top, width, height, ...) as 0-based foot points."""
    feet: dict[str, dict[int, tuple[float, float]]] = {}
    for line in gt_path.read_text().splitlines():
        if not line.strip():
            continue
        frame, object_id, left, top, width, height = (float(v) for v in line.split(",")[:6])
        feet.setdefault(str(int(object_id)), {})[int(frame) - 1] = (left + width / 2.0, top + height)
    return feet


def load_mot15(data_dir: Path) -> list[ClipTruth]:
    import cv2

    truths = []
    for seq, zone_px in MOT15_ZONES_PX.items():
        video, labels = data_dir / f"{seq}.mp4", data_dir / f"{seq}.gt.txt"
        if not (video.exists() and labels.exists()):
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        truths.append(_truth(seq, video, parse_mot_feet(labels), frames, fps, size, zone_px))
    return truths


def download_mot15(data_dir: Path) -> None:
    """Fetch MOT15 (1.3 GB) and keep only the three sequences used, as .mp4 + gt."""
    import configparser

    import cv2
    import numpy as np

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "MOT15.zip"
    if not archive.exists():
        print(f"  {MOT15_URL}")
        urllib.request.urlretrieve(MOT15_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for seq in MOT15_ZONES_PX:
            prefix = f"MOT15/train/{seq}/"
            config = configparser.ConfigParser()
            config.read_string(bundle.read(prefix + "seqinfo.ini").decode())
            fps = float(config["Sequence"]["frameRate"])
            (data_dir / f"{seq}.gt.txt").write_bytes(bundle.read(prefix + "gt/gt.txt"))
            images = sorted(n for n in names if n.startswith(prefix + "img1/") and n.endswith(".jpg"))
            first = cv2.imdecode(np.frombuffer(bundle.read(images[0]), np.uint8), cv2.IMREAD_COLOR)
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(str(data_dir / f"{seq}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
            for name in images:
                writer.write(cv2.imdecode(np.frombuffer(bundle.read(name), np.uint8), cv2.IMREAD_COLOR))
            writer.release()
            print(f"  {seq}: {len(images)} frames at {fps} fps")


DATASETS: dict[str, tuple[Path, Callable, Callable, str]] = {
    "caviar-front": (
        BENCHMARK_DATA_DIR / "caviar" / "lisbon",
        _caviar_loader("front", SHOP_ZONE_PX),
        _caviar_downloader("front"),
        "CAVIAR Lisbon, front view (26 clips), zone = shop floor",
    ),
    "caviar-corridor": (
        BENCHMARK_DATA_DIR / "caviar" / "corridor",
        _caviar_loader("cor", CORRIDOR_ZONE_PX),
        _caviar_downloader("cor"),
        "CAVIAR Lisbon, corridor view (26 clips), zone = strip along the shop fronts",
    ),
    "mot15": (
        BENCHMARK_DATA_DIR / "mot15",
        load_mot15,
        download_mot15,
        "MOT15 static cameras: PETS09-S2L1, Venice-2, ADL-Rundle-6",
    ),
}


# --- Running and scoring ------------------------------------------------------------------


def _build_pipeline(camera_id: str, zone: list[tuple[float, float]]):
    """Detector + rule engine as `MonitoringService` builds them, with one zone set."""
    from src.config import get_settings
    from src.services.monitoring import _detector_labels
    from src.vision.detector import IncidentDetector
    from src.vision.pipeline import VisionPipeline
    from src.vision.rule_engine import RuleEngine
    from src.vision.schemas import Zone
    from src.vision.zone_config import ZoneManager

    settings = get_settings()
    zones = ZoneManager()
    zones.set_zones(camera_id, [Zone(name=ZONE_NAME, polygon=zone)])
    return VisionPipeline(
        detector=IncidentDetector(
            model_path=settings.model_weights_path,
            conf_threshold=settings.detector_confidence,
            target_labels=_detector_labels(settings.detector_target_labels),
        ),
        rule_engine=RuleEngine(
            zone_manager=zones,
            alert_cooldown_sec=settings.alert_cooldown_sec,
            abandon_sec=settings.abandon_sec,
            owner_radius_ratio=settings.owner_radius_ratio,
            stationary_tolerance_ratio=settings.stationary_tolerance_ratio,
        ),
        zones_path=None,
        draw_overlays=False,
    )


def run_clip(truth: ClipTruth, target_fps: float) -> ClipRun:
    import cv2

    camera_id = f"bench_{truth.clip_id}"
    width, height = truth.frame_size
    pipeline = _build_pipeline(camera_id, [(x / width, y / height) for x, y in truth.zone_px])
    capture = cv2.VideoCapture(truth.path)
    if not capture.isOpened():
        raise ValueError(f"Could not decode {truth.path}")
    run = ClipRun(clip_id=truth.clip_id, footage_sec=0.0, sampled_frames=0)
    step = 1.0 / target_fps
    next_sample = 0.0
    frame_index = 0
    # Zone active-hour checks need a wall clock; any fixed time works for a 24/7 zone.
    now = datetime(2026, 1, 1, 12, 0)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / truth.fps
            frame_index += 1
            if timestamp + 1e-9 < next_sample:
                continue
            next_sample += step
            _, result = pipeline.process_frame(
                frame, frame_id=run.sampled_frames, camera_id=camera_id, timestamp=timestamp, now=now
            )
            run.sampled_frames += 1
            for event in result.events:
                if event.label == LABEL_INTRUSION:
                    foot = ((event.bbox.xmin + event.bbox.xmax) / 2.0, float(event.bbox.ymax))
                    via = "lower_body" if "lower-body" in (event.reason or "") else "foot"
                    run.alerts.append(Alert(round(timestamp, 3), foot, event.track_id, via))
    finally:
        capture.release()
    run.footage_sec = round(frame_index / truth.fps, 3)
    return run


def _foot_at(track: dict[int, tuple[float, float]], time_sec: float, fps: float) -> tuple[float, float] | None:
    frame = int(round(time_sec * fps))
    if frame in track:
        return track[frame]
    nearest = min(track, key=lambda f: abs(f - frame))
    return track[nearest] if abs(nearest - frame) <= fps * 0.5 else None


def match_clip(truth: ClipTruth, run: ClipRun) -> ClipOutcome:
    radius = MATCH_RADIUS_PX * truth.frame_size[1] / REFERENCE_HEIGHT
    matched: dict[int, float] = {}  # entry index -> alert time
    duplicates = false_alarms = 0
    alarmed_walkers: set[str] = set()
    details: list[str] = []
    for alert in sorted(run.alerts, key=lambda a: a.time_sec):
        candidates = []
        for index, entry in enumerate(truth.entries):
            if not entry.start_sec - EARLY_TOLERANCE_SEC <= alert.time_sec <= entry.end_sec + LATE_TOLERANCE_SEC:
                continue
            foot = _foot_at(truth.feet[entry.object_id], alert.time_sec, truth.fps)
            if foot is not None and math.dist(foot, alert.foot) <= radius:
                candidates.append((index in matched, math.dist(foot, alert.foot), index))
        if candidates:
            already, _, index = min(candidates)
            if already:
                duplicates += 1
            else:
                matched[index] = alert.time_sec
            continue
        false_alarms += 1
        kind = "other"
        for walker in truth.boundary_walkers:
            foot = _foot_at(truth.feet[walker], alert.time_sec, truth.fps)
            if foot is not None and math.dist(foot, alert.foot) <= radius:
                alarmed_walkers.add(walker)
                kind = "walker"
        details.append(f"{alert.time_sec:.1f}s {alert.via} {kind}")
    return ClipOutcome(
        clip_id=truth.clip_id,
        entries=len(truth.entries),
        detected=len(matched),
        duplicates=duplicates,
        false_alarms=false_alarms,
        boundary_walkers=len(truth.boundary_walkers),
        boundary_walkers_alarmed=len(alarmed_walkers),
        latencies_sec=[round(t - truth.entries[i].start_sec, 2) for i, t in sorted(matched.items())],
        missed_entries=[
            f"obj{e.object_id}@{e.start_sec:.1f}s" for i, e in enumerate(truth.entries) if i not in matched
        ],
        false_alarm_details=details,
    )


def score(truths: list[ClipTruth], runs: list[ClipRun]) -> dict:
    by_id = {t.clip_id: t for t in truths}
    outcomes = [match_clip(by_id[r.clip_id], r) for r in runs]
    entries = sum(o.entries for o in outcomes)
    detected = sum(o.detected for o in outcomes)
    false_alarms = sum(o.false_alarms for o in outcomes)
    walkers = sum(o.boundary_walkers for o in outcomes)
    walkers_alarmed = sum(o.boundary_walkers_alarmed for o in outcomes)
    hours = sum(r.footage_sec for r in runs) / 3600.0
    recall = _ratio(detected, entries)
    precision = _ratio(detected, detected + false_alarms)
    latencies = sorted(x for o in outcomes for x in o.latencies_sec)
    return {
        "clips": len(outcomes),
        "footage_min": round(hours * 60, 2),
        "entries": entries,
        "detected": detected,
        "false_alarms": false_alarms,
        "duplicate_alerts": sum(o.duplicates for o in outcomes),
        "false_alarms_via_lower_body": sum(" lower_body " in d for o in outcomes for d in o.false_alarm_details),
        "recall": recall,
        "recall_ci95": wilson_interval(detected, entries),
        "precision": precision,
        "precision_ci95": wilson_interval(detected, detected + false_alarms),
        "f1": (2 * precision * recall / (precision + recall)) if recall and precision else None,
        "false_alarms_per_hour": false_alarms / hours if hours else None,
        "boundary_walkers": walkers,
        "boundary_walkers_alarmed": walkers_alarmed,
        "boundary_false_alarm_rate": _ratio(walkers_alarmed, walkers),
        "boundary_false_alarm_rate_ci95": wilson_interval(walkers_alarmed, walkers),
        "latency_median_sec": latencies[len(latencies) // 2] if latencies else None,
        "latency_max_sec": latencies[-1] if latencies else None,
        "outcomes": [asdict(o) for o in outcomes],
    }


def render_markdown(report: dict) -> str:
    s = report["scores"]
    cfg = report["config"]
    far = s["false_alarms_per_hour"]
    lines = [
        f"# Intrusion benchmark — {cfg['dataset']} ({report['label']})",
        "",
        f"Run {report['run_at'][:19]}Z · weights `{cfg['weights']}` · conf {cfg['detector_confidence']} · "
        f"{cfg['target_fps']} fps",
        "",
        "| Metric | Value | 95% CI |",
        "| --- | --- | --- |",
        f"| Recall (zone entries alerted) | {s['detected']}/{s['entries']} = {_pct(s['recall'])} | {_ci(s['recall_ci95'])} |",
        f"| Precision | {_pct(s['precision'])} | {_ci(s['precision_ci95'])} |",
        f"| F1 | {_pct(s['f1'])} | |",
        f"| False alarms | {s['false_alarms']} over {s['footage_min']} min | |",
        f"| False alarms / hour | {'—' if far is None else f'{far:.1f}'} | |",
        f"| Boundary walkers alarmed | {s['boundary_walkers_alarmed']}/{s['boundary_walkers']} = "
        f"{_pct(s['boundary_false_alarm_rate'])} | {_ci(s['boundary_false_alarm_rate_ci95'])} |",
        f"| Duplicate alerts (same entry) | {s['duplicate_alerts']} | |",
        f"| False alarms raised by the lower-body fallback | {s['false_alarms_via_lower_body']}/{s['false_alarms']} | |",
        f"| Alert latency after entry (median / max) | {s['latency_median_sec']} s / {s['latency_max_sec']} s | |",
        "",
        "## Per-clip",
        "",
        "| Clip | Entries | Detected | Missed | False alarms | Duplicates | Boundary walkers (alarmed) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for o in s["outcomes"]:
        lines.append(
            f"| {o['clip_id']} | {o['entries']} | {o['detected']} | {', '.join(o['missed_entries']) or '—'} | "
            f"{o['false_alarms']} | {o['duplicates']} | {o['boundary_walkers']} ({o['boundary_walkers_alarmed']}) |"
        )
    lines += ["", "Protocol and limitations: see the module docstring of `src/benchmark/run_intrusion_benchmark.py`.", ""]
    return "\n".join(lines)


def run_dataset(name: str, data_dir: Path, fps: float, label: str, limit: int) -> int:
    _, loader, _, description = DATASETS[name]
    truths = loader(data_dir)
    if not truths:
        print(f"No {name} clips under {data_dir}. Run with --download first.")
        return 1
    if limit:
        truths = truths[:limit]

    from src.config import get_settings

    settings = get_settings()
    runs = []
    started = time.perf_counter()
    for index, truth in enumerate(truths, 1):
        run = run_clip(truth, fps)
        runs.append(run)
        print(
            f"  [{index:>2}/{len(truths)}] {truth.clip_id:<24} entries {len(truth.entries)}  "
            f"alerts {[a.time_sec for a in run.alerts] or '-'}"
        )

    report = {
        "label": label,
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "config": {
            "dataset": description,
            "weights": settings.model_weights_path,
            "detector_confidence": settings.detector_confidence,
            "target_fps": fps,
            "alert_cooldown_sec": settings.alert_cooldown_sec,
            "zones_px": {t.clip_id: t.zone_px for t in truths},
            "min_inside_sec": MIN_INSIDE_SEC,
            "near_boundary_px_at_288": NEAR_BOUNDARY_PX,
            "match_radius_px_at_288": MATCH_RADIUS_PX,
        },
        "scores": score(truths, runs),
        "runs": [asdict(r) for r in runs],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"intrusion-benchmark-{name}-{label}.json"
    md_path = RESULTS_DIR / f"intrusion-benchmark-{name}-{label}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    s = report["scores"]
    print()
    print(f"  {name}: recall {_pct(s['recall'])} {_ci(s['recall_ci95'])} ({s['detected']}/{s['entries']}), "
          f"precision {_pct(s['precision'])}, boundary walkers alarmed {s['boundary_walkers_alarmed']}/{s['boundary_walkers']}")
    print(f"  Wrote {json_path} and {md_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="caviar-front")
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to the dataset's own folder.")
    parser.add_argument("--download", action="store_true", help="Fetch the clips and labels, then exit.")
    parser.add_argument("--fps", type=float, default=5.0, help="Analysis rate (CAMERA_TARGET_FPS default: 5).")
    parser.add_argument("--label", default="baseline", help="Suffix for the result files.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N clips (smoke test).")
    args = parser.parse_args()

    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    status = 0
    for name in names:
        default_dir, _, downloader, _ = DATASETS[name]
        data_dir = args.data_dir or default_dir
        if args.download:
            downloader(data_dir)
        else:
            status |= run_dataset(name, data_dir, args.fps, args.label, args.limit)
    return status


if __name__ == "__main__":
    sys.exit(main())
