"""API-facing models for the multi-camera monitoring UI.

Separate from `src/vision/schemas.py` on purpose: those model what the *pipeline* produces
(pixels, tracks, dwell), these model what the *operator* configures and sees. The one
place they must agree is the zone polygon — `vision.schemas.Zone` rejects anything outside
[0, 1], so `CameraCreate` validates the same range at the edge, where the error message can
still reach the person drawing the polygon.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SourceType = Literal["demo", "upload", "rtsp"]
IncidentLabel = Literal["fall_detection", "abandoned_object", "intrusion"]

# Fall and abandoned-object run on every camera unconditionally (PRD §3.1). Intrusion is
# opt-in because it is meaningless without a polygon to be inside of.
ALWAYS_ON_LABELS: tuple[str, ...] = ("fall_detection", "abandoned_object")


class DemoVideo(BaseModel):
    """A sample clip shipped in the repo, offered as a one-click camera source."""

    id: str
    name: str
    suggested_label: str | None = None


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    source_type: SourceType
    source: str = Field(..., min_length=1, description="Demo id, uploaded filename, or RTSP URL")
    intrusion_enabled: bool = False
    zone_polygon: list[tuple[float, float]] | None = Field(
        default=None,
        description="Normalized [0,1] vertices drawn on the setup canvas. Required when intrusion is on.",
    )
    target_fps: float | None = Field(default=None, gt=0.0, le=60.0)
    loop_source: bool = Field(default=True, description="Restart file sources at EOF so the grid keeps running.")

    @model_validator(mode="after")
    def _check_zone(self) -> "CameraCreate":
        if not self.intrusion_enabled:
            return self

        polygon = self.zone_polygon or []
        if len(polygon) < 3:
            raise ValueError("Intrusion detection needs a zone polygon with at least 3 points.")
        for x, y in polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"Zone vertices must be normalized to [0, 1], got ({x}, {y}). "
                    "Divide canvas pixel coordinates by the displayed width/height."
                )
        return self


class CameraInfo(BaseModel):
    """A configured camera plus live worker stats, as the grid renders it."""

    id: str
    name: str
    source_type: SourceType
    source: str
    resolved_source: str
    intrusion_enabled: bool
    zone_polygon: list[tuple[float, float]] | None = None
    active_labels: list[str] = Field(default_factory=list)
    target_fps: float
    running: bool = False
    processed_frames: int = 0
    inference_capacity_fps: float = 0.0
    stream_url: str = ""


class Alert(BaseModel):
    """One incident, from rule-engine event to acknowledgement.

    `image_data_url` carries the *anonymized* frame — the same array the pipeline returns
    with faces already blurred (PRD §3.3). The raw frame is never stored or transmitted.

    `reason` is the rule engine's own evidence string and the only prose here. The VLM
    caption fields this once carried (`description`, `description_status`, provider,
    latency, error) belonged to another workstream and were removed with it — what
    remains is what the vision module can state on its own.
    """

    id: str
    camera_id: str
    camera_name: str
    label: str
    confidence: float
    created_at: str
    reason: str | None = None
    zone_name: str | None = None
    track_id: str | None = None
    image_data_url: str = ""
    acknowledged: bool = False


class SystemStatus(BaseModel):
    total_cameras: int = 0
    active_cameras: int = 0
    average_inference_capacity_fps: float = 0.0
    total_processed_frames: int = 0
    unacknowledged_alerts: int = 0
