"""Re-encode benchmark videos (MPEG-2 .mpg and the like) to H.264 .mp4 with OpenCV.

Frame-labelled datasets (CAVIAR labels every frame by index) only stay valid if the conversion
keeps every frame in order, so `to_mp4` checks the decoded frame count of the output against
the input and refuses a mismatch. No ffmpeg binary is needed: OpenCV's bundled FFmpeg reads the
input, and the H.264 encoder comes from whatever backend OpenCV finds (Media Foundation on
Windows). Where no H.264 encoder exists it falls back to MPEG-4 Part 2 (`mp4v`), still in an
.mp4 container.

Usage:
    python -m src.benchmark.video_convert data/benchmark/caviar/lisbon --delete-source
"""

import argparse
import sys
from pathlib import Path

_CODECS = ("avc1", "mp4v")


def _frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.grab():
        count += 1
    capture.release()
    return count


def to_mp4(source: Path, target: Path | None = None) -> Path:
    """Write `source` as `target` (default: same name, .mp4); return the target path."""
    import cv2

    target = target or source.with_suffix(".mp4")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not decode {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    writer = None
    for codec in _CODECS:
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        capture.release()
        raise RuntimeError("No MP4 encoder available in this OpenCV build")

    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()

    decoded = _frame_count(target)
    if decoded != written:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{target.name}: wrote {written} frames but {decoded} decode back")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pattern", default="*.mpg")
    parser.add_argument("--delete-source", action="store_true", help="Remove each source once its .mp4 checks out.")
    args = parser.parse_args()

    sources = sorted(args.directory.glob(args.pattern))
    for index, source in enumerate(sources, 1):
        target = to_mp4(source)
        print(f"  [{index}/{len(sources)}] {source.name} -> {target.name}")
        if args.delete_source:
            source.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
