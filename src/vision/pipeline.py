import logging
import math
import os
import time
from datetime import UTC, datetime

import cv2
import numpy as np

from src.vision.anonymizer import FaceAnonymizer
from src.vision.detector import IncidentDetector
from src.vision.overlay import draw_events, draw_tracks, draw_zones
from src.vision.rule_engine import RuleEngine
from src.vision.schemas import BoundingBox, FrameAnalysisResult, VideoAnalysisSummary
from src.vision.zone_config import ZoneManager

logger = logging.getLogger(__name__)

# Repo-relative default resolved against the project root, so a process launched from
# another directory still finds the zone config instead of silently running with no zones.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ZONES_PATH = os.path.join(_PROJECT_ROOT, "data", "vision", "zones.json")


class VisionPipeline:
    """End-to-End Vision Pipeline for Incident Detection and Face Anonymization.
    Outputs anonymized frames/videos and generates structured JSON handover metadata.
    """

    def __init__(
        self,
        detector: IncidentDetector | None = None,
        anonymizer: FaceAnonymizer | None = None,
        rule_engine: RuleEngine | None = None,
        zones_path: str | None = DEFAULT_ZONES_PATH,
        draw_overlays: bool = True,
        inference_size: tuple[int, int] | None = None,
    ):
        self.detector = detector or IncidentDetector()
        self.anonymizer = anonymizer or FaceAnonymizer()
        self.rule_engine = rule_engine or RuleEngine(zone_manager=ZoneManager.from_file(zones_path))
        self.draw_overlays = draw_overlays
        # When set (e.g. (640, 480)), frames are resized before detection for faster
        # inference, and bounding boxes are scaled back to the original resolution.
        self.inference_size = inference_size

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        camera_id: str | None = None,
        timestamp: float | None = None,
        now: datetime | None = None,
    ) -> tuple[np.ndarray, FrameAnalysisResult]:
        """Detect objects, evaluate incident rules, then anonymize and annotate the frame.

        Detection runs on the original frame; blurring is applied only to the frame
        that gets written out, so anonymization never degrades detection accuracy.

        `frame` is never modified. The returned array may still *alias* `frame` when
        there was nothing to blur and overlays are disabled, so treat it as read-only
        unless you copy it first.

        Args:
            frame: Input BGR image numpy array.
            frame_id: Frame index.
            camera_id: Source camera identifier, stamped onto the result and detections.
            timestamp: Seconds into the stream, driving the rule dwell timers. Defaults to
                a monotonic clock, which is what a live camera wants; process_video passes
                video-relative time so a 3s dwell means 3s of footage regardless of how
                fast the file is decoded.
            now: Wall-clock time for zone active-hour checks. Defaults to local now.

        Returns:
            Tuple of (anonymized_frame_numpy, FrameAnalysisResult_schema).
        """
        start_time = time.perf_counter()
        stream_time = timestamp if timestamp is not None else time.monotonic()
        wall_time = now or datetime.now()
        height, width = frame.shape[:2]

        # 1. Optionally resize for faster detection, then scale boxes back.
        if self.inference_size and (width, height) != self.inference_size:
            inf_w, inf_h = self.inference_size
            det_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)
            scale_x = width / inf_w
            scale_y = height / inf_h
        else:
            det_frame = frame
            scale_x = scale_y = 1.0

        # 2. Detect tracked objects on the (possibly resized) frame
        detections = self.detector.detect(
            det_frame, frame_id=frame_id, camera_id=camera_id, timestamp=timestamp
        )

        # Scale bounding boxes back to original resolution if we resized
        if scale_x != 1.0 or scale_y != 1.0:
            for det in detections:
                det.bbox = BoundingBox(
                    xmin=int(det.bbox.xmin * scale_x),
                    ymin=int(det.bbox.ymin * scale_y),
                    xmax=int(det.bbox.xmax * scale_x),
                    ymax=int(det.bbox.ymax * scale_y),
                )

        # 3. Apply temporal rules to decide which of them constitute incidents
        events = self.rule_engine.evaluate(
            detections=detections,
            frame_width=width,
            frame_height=height,
            timestamp=stream_time,
            now=wall_time,
            camera_id=camera_id,
        )

        # 4. Blur faces for privacy on the outgoing frame
        anonymized_frame = self.anonymizer.blur_faces(frame)

        # 5. Annotate zones and incidents for the operator view
        if self.draw_overlays:
            # blur_faces hands back the input array untouched when it found no faces, and
            # the draw_* helpers composite in place. Without this copy the overlays land on
            # the caller's frame — in the live path that is the reader thread's shared
            # single-slot buffer, so the "original" frame would no longer exist anywhere.
            if anonymized_frame is frame:
                anonymized_frame = frame.copy()

            zones = self.rule_engine.zone_manager.zones_for(camera_id)
            active_incidents = self.rule_engine.get_active_incidents(detections, camera_id=camera_id)
            anonymized_frame = draw_zones(anonymized_frame, zones, wall_time)
            anonymized_frame = draw_tracks(anonymized_frame, detections, active_incidents)
            anonymized_frame = draw_events(anonymized_frame, active_incidents)

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result = FrameAnalysisResult(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=datetime.now(UTC).isoformat(),
            detections=detections,
            events=events,
            processing_time_ms=processing_time_ms,
        )

        return anonymized_frame, result

    def process_video(
        self,
        video_path: str,
        output_json_path: str | None = None,
        output_video_path: str | None = None,
        max_frames: int | None = None,
        camera_id: str | None = None,
    ) -> VideoAnalysisSummary:
        """Process a video file, generate anonymized video output and save handover JSON metadata.

        Args:
            video_path: Path to input mp4 video.
            output_json_path: Path to output handover JSON file.
            output_video_path: Path to output processed video file.
            max_frames: Optional frame processing limit for fast testing.
            camera_id: Source camera identifier propagated into every frame result.

        Returns:
            VideoAnalysisSummary metadata object.

        Raises:
            FileNotFoundError: The path does not exist.
            ValueError: The file exists but cannot be decoded.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found at '{video_path}'")

        camera_id = camera_id or "cam_01"

        cap = cv2.VideoCapture(video_path)
        # os.path.exists above only proves the path is there. A truncated or non-video file
        # opens as a closed capture, and without this the loop below never runs and we
        # return a summary — plus a handover JSON — claiming a successful zero-frame job.
        if not cap.isOpened():
            cap.release()
            raise ValueError(
                f"Could not decode video '{video_path}'. The file may be corrupt, truncated, "
                "or in a codec this OpenCV build does not support."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Guard NaN as well as 0.0/None: NaN propagates into every dwell timer through
        # `frame_idx / fps` and makes each rule comparison silently False.
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = reported_fps if reported_fps and math.isfinite(reported_fps) and reported_fps > 0 else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_results = []
        frame_idx = 0
        total_events = 0
        start_processing_time = time.perf_counter()

        # Each video is an independent scene; carrying dwell timers over from the previous
        # one would fire bogus events on the first frames.
        self.rule_engine.reset()

        # Released in finally: an exception mid-decode would otherwise hold the file
        # handle (and the writer's partial output) until GC.
        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Video-relative time, so a 3s dwell means 3s of footage no matter how fast
                # the file decodes.
                anonymized_frame, frame_res = self.process_frame(
                    frame, frame_id=frame_idx, camera_id=camera_id, timestamp=frame_idx / fps
                )
                frame_results.append(frame_res)
                total_events += len(frame_res.events)

                if writer:
                    writer.write(anonymized_frame)

                frame_idx += 1
        finally:
            cap.release()
            if writer:
                writer.release()

        total_elapsed = time.perf_counter() - start_processing_time
        avg_fps = round(frame_idx / total_elapsed, 2) if total_elapsed > 0 else 0.0

        summary = VideoAnalysisSummary(
            video_path=video_path,
            camera_id=camera_id,
            total_frames=total_frames,
            processed_frames=frame_idx,
            average_fps=avg_fps,
            total_events=total_events,
            frame_results=frame_results,
        )

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                f.write(summary.model_dump_json(indent=2))
            logger.info(f"Exported handover JSON summary to '{output_json_path}'")

        logger.info(f"Processed {frame_idx} frame(s), triggering {total_events} incident event(s)")
        return summary
