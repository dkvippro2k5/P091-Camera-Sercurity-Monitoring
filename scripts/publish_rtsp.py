"""Helper script to stream a local video file to a local RTSP server (e.g. MediaMTX) using bundled FFmpeg."""

import subprocess
import sys
from pathlib import Path

# The Windows console defaults to cp1252, which cannot encode the emoji below — printing
# one raises UnicodeEncodeError and kills the publisher before FFmpeg ever starts.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"


def main():
    video_input = sys.argv[1] if len(sys.argv) > 1 else "data/vision/test/input_test.mp4"
    rtsp_url = sys.argv[2] if len(sys.argv) > 2 else "rtsp://localhost:8554/cam_01"

    if not Path(video_input).exists():
        print(f"Error: Input video '{video_input}' not found!")
        sys.exit(1)

    print("=" * 70)
    print("📡 PUBLISHING RTSP STREAM TO MEDIAMTX")
    print(f"Using FFmpeg binary: {FFMPEG_EXE}")
    print(f"Input video: {video_input}")
    print(f"RTSP Target URL: {rtsp_url}")
    print("=" * 70)
    print("Press Ctrl+C to stop publishing...\n")

    cmd = [
        FFMPEG_EXE,
        "-re",
        "-stream_loop", "-1",
        "-i", video_input,
        "-c", "copy",
        "-f", "rtsp",
        rtsp_url,
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopped RTSP stream publishing.")


if __name__ == "__main__":
    main()
