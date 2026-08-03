
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Coordinates of a bounding box normalized or pixel-based [xmin, ymin, xmax, ymax]."""
    xmin: int = Field(..., description="Top-left X coordinate")
    ymin: int = Field(..., description="Top-left Y coordinate")
    xmax: int = Field(..., description="Bottom-right X coordinate")
    ymax: int = Field(..., description="Bottom-right Y coordinate")


class IncidentDetection(BaseModel):
    """Detection result for a single incident event label."""
    label: str = Field(..., description="Event label (e.g. intrusion, fall_detection, abandoned_object)")
    track_id: str | None = Field(None, description="Unique tracking identifier across frames (e.g. person_105)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: float | None = Field(None, description="Unix timestamp of detection (seconds)")
    bbox: BoundingBox = Field(..., description="Bounding box location")
    camera_id: str | None = Field(None, description="Identifier of camera stream (e.g. cam_hallway_01)")
    zone_name: str | None = Field(None, description="Zone that triggered the event, for intrusion")
    reason: str | None = Field(None, description="Human-readable explanation of which rule fired")


class Zone(BaseModel):
    """A virtual polygon boundary on one camera view, in normalized [0, 1] coordinates.

    Normalized coordinates keep a zone valid across resolution changes, so the same
    config works for a 720p live stream and a 1080p recording of the same camera.
    """
    name: str = Field(..., description="Zone identifier, e.g. restricted_stairwell")
    polygon: list[tuple[float, float]] = Field(
        ..., min_length=3, description="Normalized (x, y) vertices in [0, 1]"
    )
    active_hours: str | None = Field(
        None, description="Forbidden window 'HH:MM-HH:MM' (may wrap midnight); None means always active"
    )
    description: str | None = Field(None, description="Human-readable note surfaced in alerts")

    @field_validator("polygon")
    @classmethod
    def _validate_normalized(cls, polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"Zone polygon vertices must be normalized to [0, 1], got ({x}, {y}). "
                    "Divide pixel coordinates by frame width/height."
                )
        return polygon


class CameraZoneConfig(BaseModel):
    """Zones configured for a single camera."""
    camera_id: str = Field(..., description="Camera identifier these zones belong to")
    zones: list[Zone] = Field(default_factory=list, description="Virtual boundaries on this camera")


class FrameAnalysisResult(BaseModel):
    """Output metadata for a single processed video frame."""
    frame_id: int = Field(..., ge=0, description="Frame index")
    timestamp: str | None = Field(None, description="ISO timestamp of frame")
    camera_id: str | None = Field(None, description="Camera stream identifier")
    detections: list[IncidentDetection] = Field(
        default_factory=list, description="Raw tracked objects present in the frame (person, backpack, ...)"
    )
    events: list[IncidentDetection] = Field(
        default_factory=list, description="Incidents triggered on this frame by the rule engine"
    )
    anonymized_image_path: str | None = Field(None, description="Path to anonymized frame output if saved")
    processing_time_ms: float = Field(default=0.0, description="Time taken to process frame in milliseconds")


class VideoAnalysisSummary(BaseModel):
    """Overall summary of video processing handover."""
    video_path: str = Field(..., description="Source video path or stream ID")
    camera_id: str | None = Field("cam_01", description="Camera identifier")
    total_frames: int = Field(default=0, description="Total number of frames in video")
    processed_frames: int = Field(default=0, description="Number of successfully analyzed frames")
    average_fps: float = Field(default=0.0, description="Average frames per second processed")
    total_events: int = Field(default=0, description="Number of incident events triggered across the video")
    frame_results: list[FrameAnalysisResult] = Field(default_factory=list, description="Frame-by-frame analysis data")
