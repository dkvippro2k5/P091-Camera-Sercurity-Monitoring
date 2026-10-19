"""Event-level intrusion benchmark on the CAVIAR Lisbon shopping-centre clips.

Dataset: CAVIAR (EC Funded CAVIAR project/IST 2001 37540), CC BY-SA,
https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
  - 26 clips from one fixed "front" camera facing a shop entrance across a corridor, 384x288,
    25 fps. People walk the corridor, stop at the window, enter the shop and come back out.
  - Every person is hand-labelled with a box on every frame (`<object><box xc yc w h>`).

The dataset has no intrusion labels, so they are derived: the restricted zone is the shop
floor behind the entrance (`SHOP_ZONE`, drawn once for this camera), and a person is *inside*
when the bottom-centre of their labelled box — the same foot point the rule uses — is inside
the polygon. The labels therefore test detection, tracking and the rule's geometry against
hand-drawn boxes; they say nothing about whether the zone itself is well chosen.

**Protocol — read before quoting a number.**

- Ground-truth entry: a labelled person's foot point inside the zone for at least
  `MIN_INSIDE_SEC`. Leaving for at least `REENTRY_GAP_SEC` and coming back is a new entry.
  Someone already inside when the clip starts counts as entering on their first frame.
- Boundary walker: a labelled person who never entered but whose foot point came within
  `NEAR_BOUNDARY_PX` of the zone — the "walking along the boundary" case. An alert on one
  of them is the false alarm that matters most for a corridor camera.
- Matching: an `intrusion` alert matches an entry when it falls between `EARLY_TOLERANCE_SEC`
  before the entry and `LATE_TOLERANCE_SEC` after the person left, and the alert box's foot
  point is within `MATCH_RADIUS_PX` of that person's labelled foot point. Each entry matches
  once; a further alert on an already matched entry is a duplicate (usually a track ID change
  inside the zone). Any other alert is a false alarm.
- Sampling at `--fps` (default 5) on the video clock; fresh detector, tracker and rule engine
  per clip. The fall and abandoned-object rules run too but are ignored.

Usage:
    python -m src.benchmark.run_intrusion_benchmark --download   # ~400 MB, once
    python -m src.benchmark.run_intrusion_benchmark --label baseline
"""

import argparse
import json
import math
import sys
import time
import urllib.request
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

DEFAULT_DATA_DIR = BENCHMARK_DATA_DIR / "caviar" / "lisbon"
CAVIAR_BASE_URL = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2"
# Clip name -> ground-truth file stem, front view. From the CAVIARDATA1 index page.
CAVIAR_FRONT_CLIPS = {
    "WalkByShop1": "fwbs1gt",
    "EnterExitCrossingPaths1": "feecp1gt",
    "EnterExitCrossingPaths2": "feecp2gt",
    "OneLeaveShop1": "fols1gt",
    "OneLeaveShop2": "fols2gt",
    "OneLeaveShopReenter1": "folsr1gt",
    "OneLeaveShopReenter2": "folsr2gt",
    "OneShopOneWait1": "fosow1gt",
    "OneShopOneWait2": "fosow2gt",
    "OneStopEnter1": "fose1gt",
    "OneStopEnter2": "fose2gt",
    "OneStopMoveEnter1": "fosme1gt",
    "OneStopMoveEnter2": "fosme2gt",
    "OneStopMoveNoEnter1": "fosmne1gt",
    "OneStopMoveNoEnter2": "fosmne2gt",
    "OneStopNoEnter1": "fosne1gt",
    "OneStopNoEnter2": "fosne2gt",
    "ShopAssistant1": "fsa1gt",
    "ShopAssistant2": "fsa2gt",
    "ThreePastShop1": "f3ps1gt",
    "ThreePastShop2": "f3ps2gt",
    "TwoEnterShop1": "f2es1gt",
    "TwoEnterShop2": "f2es2gt",
    "TwoEnterShop3": "f2es3gt",
    "TwoLeaveShop1": "f2ls1gt",
    "TwoLeaveShop2": "f2ls2gt",
}
FRAME_WIDTH, FRAME_HEIGHT = 384, 288
SOURCE_FPS = 25.0
LABEL_INTRUSION = "intrusion"
ZONE_NAME = "shop_floor"
# The shop floor behind the right-hand glass entrance, in pixels of the 384x288 front view.
# The threshold is y~148: labelled "shop enter" foot points cluster at 140-160 and corridor
# walkers at 152-185, so a walker along the shop front passes 5-35 px below the zone edge.
SHOP_ZONE_PX = [(212, 60), (335, 60), (335, 146), (212, 146)]
SHOP_ZONE = [(x / FRAME_WIDTH, y / FRAME_HEIGHT) for x, y in SHOP_ZONE_PX]
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
    feet: dict[str, dict[int, tuple[float, float]]], polygon: list[tuple[float, float]], fps: float = SOURCE_FPS
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
        if not stays and min(_distance_to_polygon(*track[f], polygon) for f in frames) <= NEAR_BOUNDARY_PX:
            walkers.append(object_id)
    entries.sort(key=lambda e: e.start_sec)
    return entries, walkers


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


def load_caviar_lisbon(data_dir: Path) -> list[ClipTruth]:
    truths = []
    for clip in CAVIAR_FRONT_CLIPS:
        video, labels = data_dir / f"{clip}front.mpg", data_dir / f"{clip}front.xml"
        if not (video.exists() and labels.exists()):
            continue
        feet, frames = parse_caviar_feet(labels)
        entries, walkers = derive_entries(feet, SHOP_ZONE_PX)
        truths.append(ClipTruth(clip, str(video), frames / SOURCE_FPS, entries, feet, walkers))
    return truths


def download_caviar_lisbon(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for clip, stem in CAVIAR_FRONT_CLIPS.items():
        for remote, local in ((f"{stem}.xml", f"{clip}front.xml"), (f"{clip}front.mpg", f"{clip}front.mpg")):
            target = data_dir / local
            if target.exists() and target.stat().st_size > 0:
                continue
            url = f"{CAVIAR_BASE_URL}/{clip}front/{remote}"
            print(f"  {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                target.write_bytes(response.read())


def _build_pipeline(camera_id: str):
    """Detector + rule engine as `MonitoringService` builds them, with the shop zone set."""
    from src.config import get_settings
    from src.services.monitoring import _detector_labels
    from src.vision.detector import IncidentDetector
    from src.vision.pipeline import VisionPipeline
    from src.vision.rule_engine import RuleEngine
    from src.vision.schemas import Zone
    from src.vision.zone_config import ZoneManager

    settings = get_settings()
    zones = ZoneManager()
    zones.set_zones(camera_id, [Zone(name=ZONE_NAME, polygon=SHOP_ZONE)])
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
    pipeline = _build_pipeline(camera_id)
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
            timestamp = frame_index / SOURCE_FPS
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
    run.footage_sec = round(frame_index / SOURCE_FPS, 3)
    return run


def _foot_at(track: dict[int, tuple[float, float]], time_sec: float) -> tuple[float, float] | None:
    frame = int(round(time_sec * SOURCE_FPS))
    if frame in track:
        return track[frame]
    nearest = min(track, key=lambda f: abs(f - frame))
    return track[nearest] if abs(nearest - frame) <= SOURCE_FPS * 0.5 else None


def match_clip(truth: ClipTruth, run: ClipRun) -> ClipOutcome:
    matched: dict[int, float] = {}  # entry index -> alert time
    duplicates = false_alarms = 0
    alarmed_walkers: set[str] = set()
    details: list[str] = []
    for alert in sorted(run.alerts, key=lambda a: a.time_sec):
        candidates = []
        for index, entry in enumerate(truth.entries):
            if not entry.start_sec - EARLY_TOLERANCE_SEC <= alert.time_sec <= entry.end_sec + LATE_TOLERANCE_SEC:
                continue
            foot = _foot_at(truth.feet[entry.object_id], alert.time_sec)
            if foot is not None and math.dist(foot, alert.foot) <= MATCH_RADIUS_PX:
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
            foot = _foot_at(truth.feet[walker], alert.time_sec)
            if foot is not None and math.dist(foot, alert.foot) <= MATCH_RADIUS_PX:
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
        f"# Intrusion benchmark — CAVIAR Lisbon front view ({report['label']})",
        "",
        f"Run {report['run_at'][:19]}Z · weights `{cfg['weights']}` · conf {cfg['detector_confidence']} · "
        f"{cfg['target_fps']} fps · zone `{ZONE_NAME}` {cfg['zone_px']}",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="Fetch the clips and labels, then exit.")
    parser.add_argument("--fps", type=float, default=5.0, help="Analysis rate (CAMERA_TARGET_FPS default: 5).")
    parser.add_argument("--label", default="baseline", help="Suffix for the result files.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N clips (smoke test).")
    args = parser.parse_args()

    if args.download:
        download_caviar_lisbon(args.data_dir)
        return 0
    truths = load_caviar_lisbon(args.data_dir)
    if not truths:
        print(f"No CAVIAR clips under {args.data_dir}. Run with --download first.")
        return 1
    if args.limit:
        truths = truths[: args.limit]

    from src.config import get_settings

    settings = get_settings()
    runs = []
    started = time.perf_counter()
    for index, truth in enumerate(truths, 1):
        run = run_clip(truth, args.fps)
        runs.append(run)
        print(
            f"  [{index:>2}/{len(truths)}] {truth.clip_id:<24} entries {len(truth.entries)}  "
            f"alerts {[a.time_sec for a in run.alerts] or '-'}"
        )

    report = {
        "label": args.label,
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "config": {
            "dataset": "CAVIAR Lisbon shopping centre, front view (26 clips)",
            "weights": settings.model_weights_path,
            "detector_confidence": settings.detector_confidence,
            "target_fps": args.fps,
            "alert_cooldown_sec": settings.alert_cooldown_sec,
            "zone_px": SHOP_ZONE_PX,
            "min_inside_sec": MIN_INSIDE_SEC,
            "near_boundary_px": NEAR_BOUNDARY_PX,
            "match_radius_px": MATCH_RADIUS_PX,
        },
        "scores": score(truths, runs),
        "runs": [asdict(r) for r in runs],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"intrusion-benchmark-caviar-{args.label}.json"
    md_path = RESULTS_DIR / f"intrusion-benchmark-caviar-{args.label}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    s = report["scores"]
    print()
    print(f"  Recall     {_pct(s['recall'])} {_ci(s['recall_ci95'])}  ({s['detected']}/{s['entries']})")
    print(f"  Precision  {_pct(s['precision'])} {_ci(s['precision_ci95'])}")
    print(f"  Boundary walkers alarmed {s['boundary_walkers_alarmed']}/{s['boundary_walkers']}")
    print(f"  Wrote {json_path} and {md_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
