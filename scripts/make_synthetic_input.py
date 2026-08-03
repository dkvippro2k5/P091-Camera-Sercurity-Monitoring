"""Generate a blank timed video, long enough for the mock timeline to exercise every rule."""

import sys

import cv2
import numpy as np

out_path = sys.argv[1] if len(sys.argv) > 1 else "data/vision/test/input_long.mp4"
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 20

width, height, fps = 640, 480, 30
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

for i in range(fps * seconds):
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.putText(
        frame, f"t={i / fps:5.1f}s", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA,
    )
    writer.write(frame)

writer.release()
print(f"wrote {seconds}s / {fps * seconds} frames to {out_path}")
