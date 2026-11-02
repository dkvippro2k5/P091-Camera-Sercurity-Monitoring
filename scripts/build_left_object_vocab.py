"""Bake the left-object vocabulary into YOLOE weights for src/vision/left_object_check.py.

YOLOE reads text prompts through a CLIP text encoder (and pip-installs the `clip` package the
first time). Doing that once here and saving the result means the running system only needs
ultralytics and one weights file.

Usage:
    python scripts/build_left_object_vocab.py [--base yoloe-26s-seg.pt] [--out src/vision/weights/yoloe-26s-left-objects.pt]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision.left_object_check import DEFAULT_LEFT_OBJECT_WEIGHTS_PATH, VOCABULARY  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", default="yoloe-26s-seg.pt", help="YOLOE weights; downloaded by ultralytics if absent.")
    parser.add_argument("--out", default=DEFAULT_LEFT_OBJECT_WEIGHTS_PATH)
    args = parser.parse_args()

    from ultralytics import YOLOE

    model = YOLOE(args.base)
    words = list(VOCABULARY)
    model.set_classes(words, model.get_text_pe(words))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Wrote {args.out} with classes {words}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
