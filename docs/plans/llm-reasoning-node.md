# Thêm node LLM (Vision → VLM → LLM) + Benchmark có/không có LLM

## Context

Hiện tại pipeline chỉ có **1 bước suy luận**: `RuleEngine` phát hiện sự kiện → `CompatibleVLMTriage`
(Qwen 3.7 Flash, nhận ảnh) trả về verdict advisory duy nhất. Team đã từng thiết kế (báo cáo
12/08, tài liệu `docs/benchmark/pipelines-spec.md` trên nhánh `feature/database-schema-infra`,
chưa merge) một kiến trúc 2 chặng — VLM tả ảnh, LLM text-only suy luận sâu hơn dựa trên mô tả +
bằng chứng đo lường + lịch sử — nhưng chưa từng được code hoá vào hệ thống thật.

Yêu cầu lần này: thêm một bước LLM sau VLM (Vision → VLM → LLM), rồi benchmark so sánh
**có LLM vs không có LLM** trên cùng bộ case. Mục tiêu là đo xem thêm LLM có cải thiện
accuracy đủ để bù lại latency/cost tăng thêm không — đúng 3 nhóm metric team đã thống nhất
từ đầu (suy luận, tốc độ, chi phí).

**Quyết định kiến trúc đã chốt với user:** dùng lời gọi trực tiếp (class đơn giản đằng sau
1 protocol, giống hệt pattern `CompatibleVLMTriage`/`NullVLMTriage` trong `vlm.py`), **không**
dùng LangGraph. Lý do: `src/agents/` đã có sẵn scaffold LangGraph nhưng hoàn toàn chưa nối vào
pipeline vision, các node chỉ format string (`analyze_node` không hề gọi LLM thật) — dùng nó
nghĩa là xây từ đầu, phức tạp hơn hẳn cho một yêu cầu chỉ là "thêm 1 bước suy luận". Pattern
protocol-based hiện tại đã proven, test dễ, và quan trọng nhất: **`MonitoringService` không
cần sửa gì cả** vì nó chỉ phụ thuộc vào protocol `VLMTriage.assess(request) -> TriageVerdict`.

## Thiết kế

### 1. `src/services/llm_reasoner.py` (file mới, mirror cấu trúc `vlm.py`)

- `LLMVerdict` (BaseModel): `verdict`, `confidence`, `rationale`, `model`, `latency_ms`,
  `total_tokens: int = 0` (mới — cần để tính cost, xem mục Benchmark), `error`.
- `LLMReasoner` (Protocol): `assess(vlm_verdict: TriageVerdict, request: TriageRequest) -> LLMVerdict`
- `NullLLMReasoner`: trả `uncertain` không gọi mạng — dùng khi tắt hoặc test.
- `CompatibleLLMReasoner`: dùng **raw OpenAI SDK** (`openai.OpenAI`, giống hệt
  `CompatibleVLMTriage`, **không** dùng `get_llm()`/LangChain) — nhất quán với triết lý đã ghi
  trong docstring `vlm.py`: "Uses the OpenAI SDK directly rather than LangChain... the
  template's LangGraph scaffold belongs to a different product surface." Structured output
  qua `response_format` strict json_schema, cùng kiểu `_RESPONSE_SCHEMA` đã có.
  - **Text-only, không gửi ảnh** — đúng nguyên tắc "Node CoT không nhận ảnh" mà team đã
    quyết định trong `pipelines-spec.md` cho bước suy luận sau vision.
  - **Never raises** — cùng triết lý `assess()` của `CompatibleVLMTriage`: mọi lỗi trả về
    `uncertain` + `error`, không bao giờ ném exception ra ngoài.

### 2. Input cho LLM — JSON metadata có cấu trúc, không phải text rời rạc

**Quyết định đã chốt lại với user:** user message gửi cho LLM là **1 khối JSON**, khớp đúng
pattern P2 gốc ("[THAY ĐỔI] Mô tả hình ảnh: {caption} / Dữ liệu cảm biến: {metadata JSON rút
gọn}") — không phải text rời rạc kiểu `vlm.py._build_prompt()`. Lý do: LLM đọc đúng field thay
vì phải suy đoán lại từ câu văn, và tránh nhầm lẫn khi user hỏi lại "đầu vào có phải JSON
không" — xác nhận: có.

`CompatibleLLMReasoner` build message dạng:

```python
{
    "role": "user",
    "content": json.dumps({
        "label": request.label,
        "zone_name": request.zone_name,
        "camera_name": request.camera_name,
        "rule_evidence": request.reason,                # verbatim, như VLM đang nhận
        "frame_summary": {
            "before_count": request.pre_count,
            "after_count": len(request.frames) - request.pre_count,
        },
        "vlm_assessment": {                              # đầu ra CompatibleVLMTriage.assess()
            "verdict": vlm_verdict.verdict,
            "confidence": vlm_verdict.confidence,
            "rationale": vlm_verdict.rationale,
        },
        "camera_history": request.examples,               # đã có sẵn qua alert_history.recent_examples()
    }, ensure_ascii=False)
}
```

System prompt (phần **CỐ ĐỊNH**, tách khỏi user message JSON): tái dùng định nghĩa 3 nhãn từ
`_SYSTEM_PROMPT` trong `vlm.py`, và **nhắc rõ nguyên tắc never-suppress** — alert đã hiển thị
cho bảo vệ trước khi bước này chạy (đúng latency contract hiện có), LLM chỉ annotate thêm,
không có quyền ẩn.

**Chống echo chamber (bổ sung sau review, 2026-08-15):** không gọi LLM là "người review độc
lập" — sai về ngữ nghĩa vì LLM không có ảnh để tự đối chiếu, chỉ có lời VLM đã nói. Prompt phải
nói rõ:
- `rule_evidence` (bằng chứng đo lường của rule engine) là **bằng chứng chính**, có trọng số
  cao nhất — đây là con số đo được, không phải suy diễn.
- `vlm_assessment` là **một giả thuyết cần kiểm chứng** đối chiếu với `rule_evidence` và
  `camera_history`, **không phải kết luận có sẵn để xác nhận lại**.
- Nếu `rule_evidence` và `vlm_assessment` mâu thuẫn nhau (vd rule đo 3/3 chỉ báo té ngã nhưng
  VLM lại nói "uncertain"), **phải nêu rõ mâu thuẫn đó trong `rationale`**, không được tự dung
  hòa/im lặng bỏ qua.
- Rủi ro còn lại (không giải quyết hết bằng prompt): LLM vẫn không có ảnh, nên đây là giảm
  thiểu, không loại bỏ được hoàn toàn nguy cơ hùa theo VLM khi cả rule_evidence lẫn VLM đều
  cùng sai theo một hướng.

**Vì sao không gọi VLM caption riêng (khác bản `pipelines-spec.md` P2 gốc):** mục tiêu benchmark
là "có LLM vs không có LLM" — phải giữ nguyên bước VLM y hệt giữa 2 nhánh so sánh, chỉ thêm
đúng 1 biến (bước LLM). Gọi thêm 1 lời gọi ảnh riêng cho caption sẽ trộn 2 thay đổi cùng lúc
(đổi cách hỏi VLM + thêm LLM), làm benchmark không còn đo đúng một biến. Tái dùng verdict VLM
sẵn có cũng rẻ hơn (không tốn thêm 1 lời gọi ảnh).

**Rủi ro đã biết, chưa xử lý:** `vlm_assessment.rationale` bị giới hạn 1 câu theo thiết kế
hiện tại của `vlm.py` (viết cho người đọc lướt nhanh, không phải để làm input suy luận tiếp).
JSON hoá không tự động làm giàu nội dung — nếu benchmark cho thấy LLM không thêm giá trị, nghi
phạm đầu tiên là input quá mỏng, không phải kiến trúc sai. Quyết định của user: cứ chạy benchmark
thật trước để đo, chưa sửa `vlm.py` để làm giàu rationale ngay.

**Rủi ro thứ hai, cũng chưa xử lý — `rule_evidence` không đồng đều giữa 3 loại sự cố** (phát
hiện khi đọc lại `rule_engine.py`): `reason` string thật sự sinh ra khác nhau rất nhiều về độ
định lượng:
- `fall_detection`: đầy đủ — `"... 3/3 indicators [+V_y=1.8H/s(>=1.2) +AR=1.6(>=1.2) +dH=72%(>=50%)]"`
- `intrusion`: gần như trống — chỉ `"{track_id} entered restricted zone '{zone}' during {window}"`,
  không có số đo nào (không có thời gian trong zone, không có độ sâu xâm nhập)
- `abandoned_object`: có ngưỡng cấu hình (`"no person within 250px"`) nhưng **không có khoảng
  cách thật đo được** tới người gần nhất, chỉ có ngưỡng dùng để so sánh

**Không phải do thiếu tọa độ bbox** — bbox thô (`IncidentDetection.bbox`) không giúp được LLM
text-only vì không có ảnh để đối chiếu không gian; rule engine chuyển hoá bbox thành chỉ số diễn
giải được là đúng hướng. Vấn đề thật là `_check_intrusion`/`_check_abandoned` trong
`rule_engine.py` sinh `reason` string mỏng hơn hẳn `_fall_indicators`, không đồng đều.

**Quyết định của user:** không sửa `rule_engine.py` ngay (file lõi, 163 test pin, sửa thêm rủi
ro ngoài phạm vi yêu cầu ban đầu). Chỉ ghi nhận: benchmark rất có thể cho thấy LLM giúp ích rõ
với case `fall_detection` nhưng gần như không giúp gì với `intrusion`/`abandoned_object` — nếu
xảy ra đúng vậy, đó là do input mỏng ở tầng rule engine, không phải do kiến trúc CascadingTriage
sai. Việc làm giàu `reason` string cho 2 loại còn lại là follow-up riêng, không nằm trong scope
lần này.

### 3. `CascadingTriage` — implement lại `VLMTriage` protocol, không cần sửa `MonitoringService`

Class mới trong `vlm.py`, nhận `vlm: OpenAIVLMTriage`-tương-đương + `reasoner: LLMReasoner`:

```
assess(request) -> TriageVerdict:
    vlm_verdict = self.vlm.assess(request)
    if vlm_verdict.error:                      # VLM đã lỗi -> LLM không chạy, trả nguyên verdict VLM
        return vlm_verdict
    llm_verdict = self.reasoner.assess(vlm_verdict, request)
    if llm_verdict.error:                       # LLM lỗi -> fallback về verdict VLM, không mất kết quả đã có
        return vlm_verdict
    return TriageVerdict(verdict=llm_verdict.verdict, confidence=llm_verdict.confidence,
                          rationale=llm_verdict.rationale, model=f"{vlm_verdict.model}+llm")
```

`TriageVerdict` schema **giữ nguyên, không đổi** — `Alert`, frontend (`AlertModal.jsx`...) không
cần sửa gì. Đây là thiết kế tối thiểu blast-radius.

### 4. Config (`src/config.py`)

- `llm_reasoning_enabled: bool = False` (mirror `vlm_enabled`)
- `llm_reasoning_timeout_sec: float = Field(default=15.0, gt=0.0)` (mirror `vlm_timeout_sec`,
  text-only nên nhanh hơn VLM)
- Tái dùng `llm_api_key`/`llm_base_url`/`model_name` đã có (đã trỏ Qwen sẵn) — không cần key mới

`build_triage_client()` trong `vlm.py` mở rộng:
- `vlm_enabled=false` → `NullVLMTriage` (không đổi)
- `vlm_enabled=true`, `llm_reasoning_enabled=false` → `CompatibleVLMTriage` (không đổi — đây
  chính là baseline **"không có LLM"** cho benchmark)
- cả hai `true` → `CascadingTriage(vlm=..., reasoner=CompatibleLLMReasoner(...))` — đây là
  nhánh **"có LLM"**

### 5. Test — file mới `tests/test_llm_reasoner.py`, mirror `test_vlm_triage.py`

- `FakeLLMReasoner` / `ExplodingLLMReasoner` (mirror `FakeTriage`/`ExplodingTriage`)
- `CascadingTriage.assess()`: happy path (VLM+LLM đều ok) · LLM lỗi → fallback verdict VLM ·
  VLM lỗi → LLM không chạy
- Prompt construction: verify rule evidence + VLM rationale + alert history đều xuất hiện
  trong prompt gửi LLM (mirror `test_prompt_carries_the_rule_evidence_verbatim`)
- `build_triage_client`: verify đúng class được chọn theo tổ hợp 2 cờ enable
- Toàn bộ offline, dùng fake double — không gọi mạng, giữ đúng yêu cầu cứng của suite hiện tại

### 6. Benchmark — script mới `src/benchmark/run_llm_reasoning_compare.py`

Sibling của `run_vlm_compare.py` (không gộp chung — trục so sánh khác: kiến trúc có/không LLM,
không phải chọn model). Với mỗi case trong `cases.jsonl`:

- **Nhánh A "VLM only"** — `CompatibleVLMTriage` (baseline hiện tại)
- **Nhánh B "VLM+LLM"** — `CascadingTriage`

Metric in ra, theo đúng 3 nhóm team đã chốt trong `docs/benchmark/pipeline-benchmark.md`:
- **Suy luận:** accuracy khớp `expected_verdict` mỗi nhánh
- **Tốc độ:** latency VLM riêng, latency LLM riêng (delta), tổng — để thấy rõ đúng cái giá
  phải trả thêm khi bật LLM, không chỉ tổng gộp
- **Chi phí:** cần thêm field `total_tokens` vào cả `TriageVerdict` và `LLMVerdict` (đọc từ
  `response.usage` của OpenAI SDK — hiện **chưa được capture ở đâu cả** trong `vlm.py`, đây là
  bổ sung nhỏ bắt buộc để benchmark cost đúng yêu cầu, không phải scope creep)

**Lưu ý quan trọng:** `cases.jsonl` hiện chỉ có 3 case (2 abandoned_object, 1 fall_detection) —
đủ để dựng harness và chạy thử, nhưng quá mỏng để kết luận chắc chắn "LLM có giúp ích không".
Việc mở rộng case (đã là pending task từ trước) nên làm song song, không chặn việc build.

### Không động tới

- `MonitoringService`, `Alert` schema, frontend — không cần sửa, nhờ thiết kế behind-protocol
- `src/agents/` (LangGraph scaffold) — để nguyên, không dùng cho tính năng này
- `src/services/llm.py`/`get_llm()` — giữ nguyên riêng cho endpoint `/chat` template, không tái dùng

## Verification

1. `pytest tests/test_llm_reasoner.py tests/test_vlm_triage.py -q` — toàn bộ pass, offline
2. `pytest -q` (full suite, hiện 163 test) — không có gì bị vỡ
3. `ruff check src/ tests/` sạch
4. Chạy thật: `VLM_API_KEY=... python -m src.benchmark.run_llm_reasoning_compare` — xem bảng
   so sánh accuracy/latency/cost giữa 2 nhánh trên 3 case hiện có, xác nhận cột "VLM+LLM" chạy
   được và không lỗi
5. Set `LLM_REASONING_ENABLED=true` trong `.env`, khởi động lại backend, xác nhận alert vẫn
   hiển thị ngay (latency contract không đổi) và verdict cuối cùng phản ánh đúng bước LLM
