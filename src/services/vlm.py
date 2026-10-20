"""Second-stage triage: a vision model judging whether a rule-engine event is real.

PRD §1.2 describes two stages — computer vision detects events, then an LLM reasons about
context to filter false alarms. `rule_engine.py` is the first stage and was deliberately
built as a cheap bbox filter whose job is to *trigger* this one, not to adjudicate: its
`reason` string is already formatted as measured evidence
(`2/3 indicators [+V_y=1.62H/s(>=1.2) -AR=0.83(>=1.2) +dH=65%(>=50%)]`) precisely so a model
could read it. This module is what finally reads it.

**The verdict is advisory. It never suppresses an alert.** Every rule-engine event still
reaches the operator; this layer only annotates. Two reasons: a small model silently
swallowing a real fall is unrecoverable for the person on the floor, and keeping every alert
visible is what makes a rules-alone vs rules-plus-VLM precision/recall comparison possible
at all (PRD §4, "Tỷ lệ lọc nhờ LLM"). Suppression becomes a config flip once the measured
numbers justify it — not before.

Three implementations behind one protocol:

    VLMTriage        - the protocol MonitoringService depends on
    OpenAIVLMTriage  - the real client
    NullVLMTriage    - returns `unavailable` without a network call

The null client is what runs when the feature is off or no API key is set, and it is what the
test suite uses — the suite stays offline and fast, and a fresh clone behaves exactly as it
did before this layer existed.
"""

import base64
import io
import json
import logging
import os
import threading
import time
from typing import Literal, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Verdict = Literal["real", "false_positive", "uncertain"]

_CONTROLLED_ACTIVITY_TERMS = (
    "push-up",
    "push up",
    "pushup",
    "plank",
    "burpee",
    "workout",
    "exercise",
    "stretching",
    "stretch",
    "floor exercise",
)


def _mentions_controlled_activity(texts: list[str]) -> bool:
    """Recognize explicit exercise cues without guessing from posture alone."""
    combined = " ".join(texts).lower()
    return any(term in combined for term in _CONTROLLED_ACTIVITY_TERMS)

# Ordered oldest-first and labelled in the prompt, because "did they fall or stand up" is
# answerable only from the sequence. A single frame cannot separate a fall from lying down.
_SYSTEM_PROMPT = """Bạn là hệ thống phân tích cảnh báo camera an ninh cho nhân viên bảo vệ.

Một bộ luật chuyển động/hình học đã phát hiện sự kiện trong đoạn video và cung cấp bằng chứng đo được.
Hãy xác định đây là sự cố THẬT đúng loại hay báo động giả mà bộ luật chưa loại bỏ được.


QUY TẮC:
- fall_detection: phải thấy chuyển động ngã bất ngờ/không kiểm soát và người tiếp tục nằm hoặc không hồi phục. Ngồi, cúi, quỳ, nằm có chủ đích, chống đẩy, plank, burpee, giãn cơ là báo động giả.
- intrusion: chỉ real khi người thực sự vượt vào vùng hạn chế; đi ngang bên ngoài không phải xâm nhập. Nếu ranh giới không rõ, dùng uncertain.
- abandoned_object: chỉ real khi túi/đồ vật bị bỏ lại và không còn chủ nhân ở gần; đồ cạnh người hoặc đang được sử dụng không phải đồ bỏ lại.

Các frame được sắp theo thời gian, nhãn BEFORE/AFTER. Đánh giá diễn biến của cả chuỗi, không dựa vào một frame.

CHỈ trả JSON theo schema. `rationale` phải đúng một câu tiếng Việt, tối đa 35 từ, nêu đối tượng, diễn biến và bằng chứng chính; không lặp lại máy móc tên cảnh báo. `frame_observations` phải có đúng một ghi chú tiếng Việt cho mỗi frame, chỉ mô tả điều nhìn thấy. Không suy đoán danh tính, nguyên nhân hoặc ý định.

Dùng uncertain nếu thiếu bằng chứng. confidence từ 0 đến 1: chỉ dùng 0.9-1.0 khi bản thân các frame đã đủ chắc chắn không còn nghi ngờ; 0.6-0.8 khi có khả năng cao nhưng chưa chắc; 0.5 trở xuống (hoặc uncertain) khi chuỗi hình mơ hồ hoặc mâu thuẫn với các phán quyết trước của camera này. Một phán quyết 'false_positive' mà bạn không chắc là câu trả lời tệ nhất: 'false_positive' confidence cao có thể bị tự động ẩn, nên bảo vệ sẽ không bao giờ thấy cảnh báo đó. Khi phân vân, hãy trả uncertain."""

_LABEL_GUIDANCE = {
    "fall_detection": "Tập trung vào chuyển đổi tư thế và khả năng hồi phục sau khi xuống sàn.",
    "intrusion": "Tập trung vào việc người có vượt qua ranh giới vùng hạn chế hay không.",
    "abandoned_object": "Tập trung vào khoảng cách giữa đồ vật và chủ nhân theo thời gian.",
}


_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "triage_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["real", "false_positive", "uncertain"]},
                "confidence": {
                    "type": "number",
                    "description": (
                        "0-1. Chỉ 0.9-1.0 khi các frame tự nó đã rõ ràng; 0.6-0.8 khi có khả "
                        "năng cao nhưng chưa chắc; <=0.5 hoặc verdict 'uncertain' khi chuỗi hình "
                        "mơ hồ hoặc mâu thuẫn với các phán quyết trước của camera này."
                    ),
                },
                "rationale": {"type": "string"},
                "frame_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["verdict", "confidence", "rationale", "frame_observations"],
            "additionalProperties": False,
        },
    },
}


class TriageVerdict(BaseModel):
    """One model judgement on one alert.

    `error` is set on every failure path, and the verdict is `uncertain` whenever it is —
    see `OpenAIVLMTriage.assess` for why this never raises.
    """

    verdict: Verdict = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    frame_observations: list[str] = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    error: str | None = None


class TriageRequest(BaseModel):
    """Everything the model is shown about one alert.

    `reason` is the rule engine's own evidence string, passed verbatim — it carries the
    measured indicator values, which is the whole point of routing it here.
    """

    label: str
    reason: str | None = None
    zone_name: str | None = None
    camera_name: str = ""
    # JPEG bytes, chronological. `pre_count` marks the split: the first `pre_count` frames
    # precede the moment the rule fired, the rest follow it.
    frames: list[bytes] = Field(default_factory=list)
    pre_count: int = 0
    # Rendered lines from past operator-labelled incidents on this camera (alert_history).
    examples: list[str] = Field(default_factory=list)
    # True when `frames` were cropped to the one person this alert's track_id names
    # (MonitoringService._collect_clip), rather than showing the full camera scene. The
    # prompt needs to say so explicitly: without it, a model seeing a tight, single-person
    # frame could read "nobody else visible" as evidence, when the truth is only that
    # everyone else was cropped out of shot. See docs/plans/vlm-throughput-and-concurrent-
    # events.md stage 3 — this is what makes several simultaneous alerts on one camera
    # separable, each judged on the person it is actually about.
    cropped: bool = False
    # True when the last frame is not from the clip but a before/after of the object's spot:
    # the background model's picture of it (left) beside the frame the alert fired on (right).
    comparison_last: bool = False


class VLMTriage(Protocol):
    """What MonitoringService depends on. Keeps the real client out of the tests."""

    def assess(self, request: TriageRequest) -> TriageVerdict: ...


class NullVLMTriage:
    """No-op triage: used when the feature is off, no key is configured, or in tests."""

    def __init__(self, reason: str = "VLM triage disabled") -> None:
        self._reason = reason

    def assess(self, request: TriageRequest) -> TriageVerdict:
        return TriageVerdict(verdict="uncertain", error=self._reason)


class OpenAIVLMTriage:
    """Vision-model triage over the OpenAI chat completions API.

    Uses the OpenAI SDK directly rather than LangChain: this is one structured call with
    images, and the template's LangGraph scaffold belongs to a different product surface.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_sec: float = 20.0,
        max_retries: int = 1,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # imported lazily so the package is optional at runtime

        self.model = model
        # The SDK's own retry handles the transient 429/5xx case; a hand-rolled loop around
        # a 20s timeout would multiply worst-case latency instead of bounding it.
        # `base_url=None` keeps the SDK's own default (api.openai.com); anything else points
        # at an OpenAI-compatible proxy (e.g. DashScope for Qwen) — without this, a configured
        # OPENAI_BASE_URL was silently ignored and every call went to the real OpenAI API with
        # a key that API had never issued, failing 401 on every single alert.
        self._client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout_sec, max_retries=max_retries)
        # Qwen's "thinking" models (qwen3*, e.g. qwen3.7-flash) default to an extended
        # reasoning mode that burns thousands of hidden tokens per call — billed as output,
        # not bounded by max_tokens — which llm_reasoner.py measured at 8-25s latency for a
        # task that should take ~1s. Triage is a temperature=0.0 classification, so we opt
        # out with the DashScope-only `enable_thinking: False` extension. The vision models
        # (qwen-vl-plus/-max, qwen3-vl-*) do not run thinking mode and some reject the
        # parameter, and the real OpenAI API rejects any unknown extra field — so send it
        # only for a qwen3* model reached through a configured base_url.
        wants_no_thinking = bool(base_url) and str(model).lower().startswith("qwen3")
        self._extra_body = {"enable_thinking": False} if wants_no_thinking else {}

    def assess(self, request: TriageRequest) -> TriageVerdict:
        """Judge one alert. Never raises.

        A triage layer that propagated an exception would take out the alert it was meant to
        annotate — the operator would lose a real incident because a third-party API had a bad
        minute. Every failure becomes `uncertain` plus an `error` string instead, which the UI
        renders as "assessment unavailable" while leaving the alert itself untouched.
        """
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_content(request)},
                ],
                response_format=_RESPONSE_SCHEMA,
                # Deterministic-ish: this is a classification, not a writing task.
                temperature=0.0,
                max_tokens=600,
                # See __init__: disables Qwen's default thinking mode; empty for real OpenAI.
                extra_body=getattr(self, "_extra_body", {}),
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            frame_observations = [
                str(observation).strip()
                for observation in payload.get("frame_observations", [])
                if str(observation).strip()
            ]
            # A count mismatch used to raise, which sent an otherwise-complete judgement --
            # verdict, confidence, rationale and all -- down the `except` below and turned it
            # into `uncertain` with a zeroed confidence. That trade is backwards: the
            # observations are an annotation the guard reads alongside the verdict, while the
            # verdict is the thing an alert exists to produce. Losing a "real, 0.95" on a
            # fall because the model wrote five notes for six frames is a safety regression,
            # not strictness.
            #
            # The count does drift in practice: `_RESPONSE_SCHEMA` cannot express "exactly N
            # items", so the count is only ever *asked* for in the prompt, and `strict`
            # json_schema is an OpenAI feature that the OpenAI-compatible providers this
            # points at (Qwen and friends, via OPENAI_BASE_URL) do not enforce.
            #
            # When the count is wrong, the whole observation list is dropped rather than
            # trimmed or padded. The response is a bare array with no frame indices, so a
            # short list could be missing its last entry (padding at the end would be right)
            # or an entry in the middle (every later note then describes the wrong frame).
            # `frame_observations[i]` is read positionally against the evidence strip by both
            # the modal and `embedding_text`, and a fall's BEFORE/AFTER captions pinned to
            # the wrong frames actively mislead. A verdict with no per-frame notes is the
            # honest degrade; a verdict with misaligned ones is not.
            expected = len(request.frames)
            if len(frame_observations) != expected:
                logger.warning(
                    f"VLM returned {len(frame_observations)} frame observations for {expected} frames; "
                    "dropping the observations and keeping the verdict."
                )
                frame_observations = []
            verdict = payload.get("verdict", "uncertain")
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
            rationale = str(payload.get("rationale", "")).strip()
            if verdict == "real" and _mentions_controlled_activity([rationale, *frame_observations]):
                verdict = "false_positive"
                confidence = min(confidence, 0.9)
                rationale = "Visible exercise cues indicate controlled floor movement rather than an uncontrolled fall."
            return TriageVerdict(
                verdict=verdict,
                # Clamped rather than trusted: a model returning 0-100 instead of 0-1 would
                # otherwise fail Pydantic validation and lose an otherwise-good verdict.
                confidence=confidence,
                rationale=rationale,
                frame_observations=frame_observations,
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        except Exception as exc:  # noqa: BLE001 - see docstring; no failure may reach the alert
            logger.warning(f"VLM triage failed: {type(exc).__name__}: {exc}")
            return TriageVerdict(
                verdict="uncertain",
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_content(self, request: TriageRequest) -> list[dict]:
        """Text block first, then the frames labelled BEFORE / AFTER in order."""
        content: list[dict] = [{"type": "text", "text": self._build_prompt(request)}]
        for index, jpeg in enumerate(request.frames):
            phase = "BEFORE" if index < request.pre_count else "AFTER"
            content.append({"type": "text", "text": f"[{phase} frame {index + 1}]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("utf-8"),
                        # Small objects and zone boundaries can determine the verdict, so
                        # preserve spatial detail even though it costs more image tokens.
                        "detail": "high",
                    },
                }
            )
        return content

    @staticmethod
    def _build_prompt(request: TriageRequest) -> str:
        lines = [
            f"Alert type: {request.label}",
            f"Camera: {request.camera_name or 'unknown'}",
        ]
        if request.zone_name:
            lines.append(f"Zone: {request.zone_name}")
        if request.reason:
            # Verbatim. The indicator values and their thresholds are the evidence.
            lines.append(f"Rule engine evidence: {request.reason}")
        lines.append(
            f"Frames: {request.pre_count} before the trigger, "
            f"{max(0, len(request.frames) - request.pre_count)} after, in chronological order."
        )
        if request.comparison_last:
            lines.append(
                "The last frame is not part of the clip: it shows the spot the alert is about "
                "twice, outlined in red -- on the left as it looked before the object appeared "
                "(the background model's picture, blurrier), on the right as it is now. Use it to "
                "tell an object that was left there from one that was always there."
            )
        if request.cropped:
            lines.append(
                "These frames are cropped to the one person this alert is about, padded to "
                "keep nearby context. Other people or objects may exist just outside the "
                "crop -- do not conclude anyone is alone or unattended from the crop alone."
            )
        lines.append(f"Incident-specific focus: {_LABEL_GUIDANCE.get(request.label, 'Describe the visible event progression.')}" )
        if request.label == "fall_detection":
            lines.append(
                "Fall check: require an uncontrolled collapse and sustained inability to recover. "
                "push-ups, planks, burpees, stretching, and other controlled floor exercises "
                "are false positives, even when the person becomes prone."
            )
        lines.append(f"Return exactly {len(request.frames)} frame_observations, one for each labelled frame.")

        if request.examples:
            lines.append("")
            lines.append("Các phán quyết trước đây của bảo vệ (mới nhất ở cuối):")
            lines.extend(f"- {example}" for example in request.examples)
            lines.append(
                "Dùng để hiệu chỉnh: nếu đánh giá của bạn mâu thuẫn với một xu hướng nhất quán "
                "trong các phán quyết này, hãy hạ confidence hoặc trả 'uncertain'. Chỉ giữ "
                "confidence cao khi bằng chứng trong các frame tự nó đã rõ ràng."
            )

        return "\n".join(lines)


class LocalSmolVLMTriage:
    """Local SmolVLM2 posture captioner with deterministic fall-transition reasoning."""

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        device: str = "auto",
        max_image_size: int = 512,
        max_new_tokens: int = 64,
    ) -> None:
        self.model = model
        self.device = device
        self.max_image_size = max_image_size
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._torch = None
        # Monitoring can run two triage workers, but this workstation has one 4 GB GPU.
        self._inference_lock = threading.Lock()

    def assess(self, request: TriageRequest) -> TriageVerdict:
        started = time.perf_counter()
        try:
            with self._inference_lock:
                self._ensure_loaded()
                images = self._decode_images(request.frames)
                observations = [self._caption_image(image) for image in images]

            verdict, confidence, rationale = self._infer_temporal_verdict(request, observations)
            return TriageVerdict(
                verdict=verdict,
                confidence=confidence,
                rationale=rationale,
                frame_observations=observations,
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        except Exception as exc:  # noqa: BLE001 - local inference must never suppress an alert
            logger.warning(f"Local VLM triage failed: {type(exc).__name__}: {exc}")
            return TriageVerdict(
                verdict="uncertain",
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # Some processor components perform optional metadata probes even with
        # local_files_only=True. The local provider must remain fully offline.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"VLM device '{resolved_device}' requested but CUDA is unavailable")

        dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
        logger.info(f"Loading local VLM {self.model} on {resolved_device} ({dtype})")
        processor = AutoProcessor.from_pretrained(self.model, local_files_only=True)
        processor.image_processor.size = {"longest_edge": self.max_image_size}
        processor.image_processor.max_image_size = {"longest_edge": self.max_image_size}
        processor.image_processor.do_image_splitting = False
        model = AutoModelForImageTextToText.from_pretrained(
            self.model,
            dtype=dtype,
            _attn_implementation="eager",
            local_files_only=True,
        )
        model.to(resolved_device)
        model.eval()

        self.device = resolved_device
        self._torch = torch
        self._processor = processor
        self._model = model

    def _decode_images(self, frames: list[bytes]) -> list:
        from PIL import Image

        images = []
        for jpeg in frames:
            image = Image.open(io.BytesIO(jpeg)).convert("RGB")
            image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            images.append(image)
        return images

    def _caption_image(self, image) -> str:  # noqa: ANN001 - PIL is an optional local dependency
        """Ask the small model one visual question it can answer reliably."""
        content = [
            {"type": "image"},
            {"type": "text", "text": "Describe the person's posture in one short sentence."},
        ]
        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt")
        for key, value in inputs.items():
            if self._torch.is_tensor(value):
                inputs[key] = value.to(
                    device=self.device,
                    dtype=(
                        self._torch.float16
                        if value.is_floating_point() and self.device.startswith("cuda")
                        else None
                    ),
                )

        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=min(self.max_new_tokens, 64),
            )
        prompt_tokens = inputs["input_ids"].shape[1]
        caption = self._processor.batch_decode(
            generated_ids[:, prompt_tokens:], skip_special_tokens=True
        )[0].strip()
        if not caption:
            raise ValueError("model returned an empty posture caption")
        return caption

    @staticmethod
    def _posture_state(observation: str) -> Literal["upright", "lying", "other", "unknown"]:
        """Map conservative, visible posture vocabulary to a small state vocabulary."""
        text = observation.lower()
        if any(term in text for term in ("lying", "laying", "prone", "on the floor", "on floor")):
            return "lying"
        if any(term in text for term in ("standing", "upright", "walking", "running")):
            return "upright"
        if any(term in text for term in ("sitting", "seated", "kneeling", "crouching")):
            return "other"
        return "unknown"

    @classmethod
    def _infer_temporal_verdict(
        cls, request: TriageRequest, observations: list[str]
    ) -> tuple[Verdict, float, str]:
        """Use the model as a captioner; keep incident reasoning deterministic and cautious."""
        if request.label != "fall_detection":
            return "uncertain", 0.0, "Local posture captions do not determine this alert type."

        split = min(max(request.pre_count, 0), len(observations))
        before = [cls._posture_state(text) for text in observations[:split]]
        after = [cls._posture_state(text) for text in observations[split:]]
        if _mentions_controlled_activity(observations):
            return "false_positive", 0.85, "The visual descriptions contain controlled exercise cues rather than an uncontrolled fall."
        if "upright" in before and "lying" in after:
            return "real", 0.7, "Posture captions show an upright person before the alert and a person lying on the floor after it."
        return "uncertain", 0.0, "Posture captions do not establish an upright-to-floor transition."

def build_triage_client(settings) -> VLMTriage:  # noqa: ANN001 - Settings, avoided to dodge a config import cycle
    """Pick a triage client from settings, degrading to the null client rather than failing.

    Both switches matter: `vlm_enabled` is the deliberate off switch, and a missing API key
    is the accidental one. Treating a missing key as an error would make the whole monitoring
    service refuse to start over an optional annotation layer.
    """
    if not getattr(settings, "vlm_enabled", False):
        return NullVLMTriage("VLM triage disabled (VLM_ENABLED=false)")
    if getattr(settings, "vlm_provider", "openai") == "local":
        return LocalSmolVLMTriage(
            model=getattr(settings, "vlm_local_model", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
            device=getattr(settings, "vlm_local_device", "auto"),
            max_image_size=getattr(settings, "vlm_local_max_image_size", 512),
            max_new_tokens=getattr(settings, "vlm_local_max_new_tokens", 64),
        )
    if not settings.openai_api_key:
        logger.warning("VLM_ENABLED is true but OPENAI_API_KEY is empty; alerts will carry no assessment.")
        return NullVLMTriage("No OPENAI_API_KEY configured")

    try:
        return OpenAIVLMTriage(
            api_key=settings.openai_api_key,
            model=settings.vlm_model,
            timeout_sec=settings.vlm_timeout_sec,
            base_url=getattr(settings, "openai_base_url", "") or None,
        )
    except Exception as exc:  # noqa: BLE001 - a missing openai package must not block startup
        logger.error(f"Could not build the VLM client, continuing without triage: {exc}")
        return NullVLMTriage(f"VLM client unavailable: {exc}")
