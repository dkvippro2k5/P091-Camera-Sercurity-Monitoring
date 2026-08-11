"""API-facing models for the multi-camera monitoring UI.

Separate from `src/vision/schemas.py` on purpose: those model what the *pipeline* produces
(pixels, tracks, dwell), these model what the *operator* configures and sees. The one
place they must agree is the zone polygon — `vision.schemas.Zone` rejects anything outside
[0, 1], so `CameraCreate` validates the same range at the edge, where the error message can
still reach the person drawing the polygon.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.vision.zone_config import validate_active_hours

SourceType = Literal["demo", "upload", "rtsp"]
IncidentLabel = Literal["fall_detection", "abandoned_object", "intrusion"]

# Second-stage triage (src/services/vlm.py). "skipped" is the default because the layer is
# off by default — an alert that was never assessed must not look like one assessed and
# found inconclusive.
TriageStatus = Literal["pending", "complete", "skipped"]
TriageVerdictLabel = Literal["real", "false_positive", "uncertain"]

# What the guard concluded (PRD §3.2). Distinct from the model's vocabulary on purpose: the
# operator answers "is this real", not "was the detector right".
OperatorVerdict = Literal["real", "false"]

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
    zone_active_hours: str | None = Field(
        default=None,
        description=(
            "Forbidden window 'HH:MM-HH:MM' for the drawn zone, e.g. '22:00-05:00' (PRD §3.1). "
            "May wrap midnight. None means the zone is always armed."
        ),
    )
    target_fps: float | None = Field(default=None, gt=0.0, le=60.0)
    loop_source: bool = Field(default=True, description="Restart file sources at EOF so the grid keeps running.")

    @model_validator(mode="after")
    def _check_zone(self) -> "CameraCreate":
        # Normalized here, not in the UI, so an API client gets the same guarantee: by the
        # time this reaches ZoneManager the window is either valid 'HH:MM-HH:MM' or None.
        # An empty string from a cleared form field means "no window", not a parse error.
        if self.zone_active_hours is not None:
            if self.zone_active_hours.strip():
                self.zone_active_hours = validate_active_hours(self.zone_active_hours)
            else:
                self.zone_active_hours = None

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
    zone_active_hours: str | None = None
    active_labels: list[str] = Field(default_factory=list)
    target_fps: float
    running: bool = False
    processed_frames: int = 0
    inference_capacity_fps: float = 0.0
    stream_url: str = ""


class Alert(BaseModel):
    """One incident, from rule-engine event to operator judgement.

    `image_data_url` carries the captured frame, annotated with zones and tracks. Face
    blurring was removed on 2026-08-10, so faces in it are identifiable — PRD §3.3 asks
    for anonymization and this no longer satisfies it.

    `reason` is the rule engine's own evidence string. The `triage_*` fields are the second
    stage's reading of it (PRD §1.2) and are strictly **advisory**: a `false_positive`
    verdict annotates this alert, it never hides it. See `src/services/vlm.py`.

    `operator_verdict` is deliberately separate from `acknowledged`. Acknowledging means
    "the operator dismissed the modal"; it does not say whether the incident was real. HITL
    needs three states — real, false, and not-yet-reviewed — and overloading the boolean
    would silently record every dismissal as a confirmation.
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

    # Second-stage triage. "pending" is what the UI shows a spinner for; "skipped" means the
    # layer was off or the queue was saturated, which is not the same as an inconclusive
    # judgement and must not read like one.
    triage_status: TriageStatus = "skipped"
    triage_verdict: TriageVerdictLabel | None = None
    triage_confidence: float | None = None
    triage_rationale: str | None = None
    triage_error: str | None = None

    operator_verdict: OperatorVerdict | None = None
    # Who judged it. A labelled dataset whose labels have no author cannot be audited when
    # two guards disagree about what a false alarm looks like. None on rows judged before
    # accounts existed, and on any path that records a verdict without auth in front of it.
    judged_by: str | None = None


class AlertVerdict(BaseModel):
    """The guard's HITL judgement on one alert (PRD §3.2)."""

    verdict: OperatorVerdict


class SystemStatus(BaseModel):
    total_cameras: int = 0
    active_cameras: int = 0
    average_inference_capacity_fps: float = 0.0
    total_processed_frames: int = 0
    unacknowledged_alerts: int = 0
