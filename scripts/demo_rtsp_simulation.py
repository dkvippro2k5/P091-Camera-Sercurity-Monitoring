"""Demo script for simulating multi-camera RTSP streams in parallel using MultiStreamManager.

Includes live GUI preview window (--show) rendered safely on the Main Thread and video recorder (--output).
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2

# Fix Windows console encoding for Unicode emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision import MultiStreamManager, RTSPSimulator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("demo_rtsp_simulation")


def main():
    parser = argparse.ArgumentParser(description="Multi-stream RTSP Vision Analysis with live preview & incident detection")
    parser.add_argument("source", nargs="?", default="data/vision/test/input_test.mp4", help="Video file or rtsp:// URL")
    parser.add_argument("count", nargs="?", type=int, default=1, help="Number of simulated camera streams")
    parser.add_argument("duration", nargs="?", type=int, default=0, help="Run duration in seconds (0 = run continuously until 'q' or Ctrl+C)")
    parser.add_argument("--show", action="store_true", help="Display live visual window with bounding boxes & face blur")
    parser.add_argument("--output", type=str, default=None, help="Save annotated output video to path (e.g. output_rtsp.mp4)")

    args = parser.parse_args()

    source_arg = args.source
    num_cameras = args.count
    run_duration_sec = args.duration
    show_gui = args.show
    output_path = args.output

    latest_frames = {}
    frame_lock = threading.Lock()
    video_writer = None
    # Only one camera is recorded; set once the cameras are registered below.
    recording_camera_id = None

    def on_incident(camera_id: str, result, frame):
        """Alerting path: only fires for frames carrying rule-engine events."""
        for event in result.events:
            logger.warning(
                f"🚨 [INCIDENT DETECTED] Cam '{camera_id}' | Frame {result.frame_id} | "
                f"Event: {event.label.upper()} | Zone: {event.zone_name or 'N/A'} | Reason: {event.reason}"
            )

    def on_frame(camera_id: str, result, frame):
        """Preview/recording path: fires for every analyzed frame.

        Runs on worker threads, so it only stores frames — cv2.imshow has to happen on
        the main thread. Recording is limited to the first camera: several workers
        interleaving into one writer produces a video that cuts between views.
        """
        nonlocal video_writer
        if frame is None:
            return

        with frame_lock:
            latest_frames[camera_id] = frame.copy()

            if output_path and camera_id == recording_camera_id:
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))
                video_writer.write(frame)

    print("=" * 70)
    print("🎬 VISION RTSP STREAM MONITOR & INCIDENT DETECTOR")
    print(f"Source URL/File: {source_arg}")
    print(f"Cameras count: {num_cameras}")
    print(f"Duration: {'UNLIMITED (press q or Ctrl+C to stop)' if run_duration_sec <= 0 else f'{run_duration_sec}s'}")
    print(f"Live GUI Preview: {'ENABLED' if show_gui else 'DISABLED (run with --show to view window)'}")
    print(f"Output Recording: {output_path or 'DISABLED'}")
    print("=" * 70)

    manager = MultiStreamManager(
        global_on_incident_callback=on_incident,
        global_on_frame_callback=on_frame,
    )

    if source_arg.startswith("rtsp://") or source_arg.startswith("http://"):
        for i in range(1, num_cameras + 1):
            cam_id = f"cam_{i:02d}"
            # Zero-padded to match cam_id and publish_rtsp.py's default path (cam_01), so
            # a base of rtsp://host/cam fans out to cam_01, cam_02, ... rather than cam_1.
            url = source_arg if num_cameras == 1 else f"{source_arg.rstrip('/')}_{i:02d}"
            manager.add_camera(camera_id=cam_id, rtsp_url=url, target_fps=15.0, autostart=False, loop_finite_sources=True)
            print(f"  [+] Added camera stream: {cam_id} ({url})")
    else:
        mock_configs = RTSPSimulator.generate_mock_camera_configs(count=num_cameras)
        for cfg in mock_configs:
            cam_id = cfg["camera_id"]
            manager.add_camera(camera_id=cam_id, rtsp_url=source_arg, target_fps=15.0, autostart=False, loop_finite_sources=True)
            print(f"  [+] Added simulated camera stream: {cam_id} ({cfg['location']})")

    print("\n🚀 Starting vision processing...")
    # Record the first camera only; see on_frame.
    recording_camera_id = "cam_01"
    manager.start_all()

    start_time = time.time()
    last_stat_time = time.time()

    try:
        while run_duration_sec <= 0 or (time.time() - start_time < run_duration_sec):
            # 1. Main thread renders GUI windows (Windows GUI loop requirement)
            if show_gui:
                with frame_lock:
                    for cam_id, frame in list(latest_frames.items()):
                        cv2.imshow(f"Vision Monitor - {cam_id}", frame)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    print("\nUser pressed 'q' to stop.")
                    break
            else:
                time.sleep(0.05)

            # 2. Print periodic stats every 1 second
            if time.time() - last_stat_time >= 1.0:
                last_stat_time = time.time()
                status = manager.get_status()
                print(
                    f"📊 Active: {status['active_cameras']}/{status['total_cameras']} | "
                    f"Processed: {status['total_processed_frames']} frames | "
                    f"Capacity/cam: {status['average_inference_capacity_fps']} fps"
                )
                # A worker that hit EOF or tripped its breaker stops silently otherwise.
                for cam in status["cameras"]:
                    if not cam["running"] and cam["stopped_reason"]:
                        print(f"   ⚠ {cam['camera_id']} stopped: {cam['stopped_reason']}")

    except KeyboardInterrupt:
        print("\nStopping streams...")

    # Snapshot before stopping: stop_all() drops the workers from the registry, so a
    # status read afterwards has nothing left to count and reports zero.
    final_status = manager.get_status()

    print("\n🛑 Stopping camera stream workers...")
    manager.stop_all()

    with frame_lock:
        if video_writer:
            video_writer.release()

    if show_gui:
        cv2.destroyAllWindows()

    print("=" * 70)
    print("🏁 ANALYSIS COMPLETED")
    print(f"Total Frames Analyzed: {final_status['total_processed_frames']}")
    for cam in final_status["cameras"]:
        print(
            f"  {cam['camera_id']}: {cam['processed_frames']} frames | "
            f"{cam['total_read_frames']} read | last latency {cam['last_latency_ms']} ms"
            + (f" | stopped: {cam['stopped_reason']}" if cam["stopped_reason"] else "")
        )
    if output_path:
        print(f"Annotated output video saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
