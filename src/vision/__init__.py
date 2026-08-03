"""Vision module for incident event detection and face anonymization."""

from src.vision.anonymizer import FaceAnonymizer
from src.vision.detector import IncidentDetector
from src.vision.overlay import draw_events, draw_tracks, draw_zones
from src.vision.pipeline import VisionPipeline
from src.vision.rule_engine import (
    LABEL_ABANDONED,
    LABEL_FALL,
    LABEL_INTRUSION,
    RuleEngine,
)
from src.vision.schemas import (
    BoundingBox,
    CameraZoneConfig,
    FrameAnalysisResult,
    IncidentDetection,
    VideoAnalysisSummary,
    Zone,
)
from src.vision.stream_manager import MultiStreamManager
from src.vision.stream_simulator import RTSPSimulator
from src.vision.stream_worker import CameraStreamWorker
from src.vision.zone_config import ZoneManager, is_zone_active, point_in_polygon

__all__ = [
    "IncidentDetector",
    "FaceAnonymizer",
    "VisionPipeline",
    "RuleEngine",
    "LABEL_INTRUSION",
    "LABEL_FALL",
    "LABEL_ABANDONED",
    "ZoneManager",
    "point_in_polygon",
    "is_zone_active",
    "draw_zones",
    "draw_events",
    "draw_tracks",
    "BoundingBox",
    "IncidentDetection",
    "FrameAnalysisResult",
    "VideoAnalysisSummary",
    "Zone",
    "CameraZoneConfig",
    "MultiStreamManager",
    "CameraStreamWorker",
    "RTSPSimulator",
]
