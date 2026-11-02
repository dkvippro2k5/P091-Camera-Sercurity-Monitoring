# Tiêu chí benchmark đánh giá hệ thống

## Context

Báo cáo hiện tại (`docs/gate2/evaluation.md`, trỏ tới từ `eval/results/report.md`) chỉ có **5 test
case thủ công**, mỗi case ghi lại confidence của Vision + verdict của VLM cho 1 video cụ thể. Đủ
để demo Gate 2 (chứng minh pipeline chạy đúng end-to-end), nhưng không phải benchmark theo nghĩa
academic/industry: n=5 quá nhỏ để có ý nghĩa thống kê, không có Precision/Recall tổng hợp, và
không đo chất lượng tracking.

Tài liệu này định nghĩa bộ tiêu chí benchmark phù hợp, dựa trên 3 mảng literature liên quan trực
tiếp đến hệ thống: fall/anomaly detection trong video giám sát (UR Fall Dataset, Le2i, Multiple
Cameras Fall Dataset, UCF-Crime, ShanghaiTech Campus, CUHK Avenue), multi-object tracking chuẩn
MOTChallenge (MOT16/17/20), và video-LLM/VLM reasoning (Holmes-VAD và các benchmark VideoLLM
tương tự). Với mỗi nhóm, ghi rõ **có áp dụng được ngay với dữ liệu hiện có không**.

**Xác định loại bài toán trước khi chọn benchmark:** hệ thống là bài toán **MOT (Multi-Object
Tracking)**, không phải SOT. Bằng chứng: `IncidentDetector._detect_real()` gọi
`self.model.track(..., persist=True)` (ByteTrack), gán `track_id` ổn định cho **nhiều đối tượng
đồng thời** (nhiều người, nhiều vali cùng lúc — xem case 4 "nhà kho nhiều người"), và
`RuleEngine` giữ state độc lập theo từng `track_id` (dwell timer fall, chủ sở hữu abandoned-object,
foot point intrusion). Điều này kéo theo một nhóm tiêu chí benchmark hiện đang **thiếu hoàn toàn**:
chất lượng tracking, không chỉ chất lượng phát hiện sự kiện cuối cùng.

**Phạm vi:** chỉ có **VLM triage** (`src/services/vlm.py`) trong pipeline hiện tại, không có node
LLM reasoning riêng — tiêu chí ở đây chỉ đánh giá VLM, không đặt ra metric cho một bước LLM chưa
tồn tại trong hệ thống thật.

## Nhóm 1 — Chất lượng phát hiện & kiểm soát báo động giả (Detection Quality + False Alarm Rate)

False Alarm Rate về bản chất là một lát cắt khác của cùng bộ ground truth dùng cho Precision/Recall
— cả hai đọc từ đúng 1 nguồn (`operator_verdict` trong `alert_history`), nên gộp chung một nhóm
thay vì tách riêng.

Chuẩn từ các dataset fall-detection kinh điển (UR Fall Detection Dataset, Le2i, Multiple Cameras
Fall Dataset) và anomaly-detection video (UCF-Crime, ShanghaiTech Campus, CUHK Avenue):

| Metric | Ý nghĩa |
|---|---|
| Sensitivity/Recall | Tỉ lệ sự kiện thật bắt được — quan trọng nhất vì bỏ sót fall/intrusion là nguy hiểm thật |
| Specificity | Tỉ lệ non-event không bị báo nhầm |
| Precision, F1-score | Cân bằng 2 chỉ số trên |
| AUC-ROC | Dùng được vì hệ thống đã có confidence liên tục (rule engine + VLM) |
| False Alarm Rate per hour/per camera | Số alert cần review / giờ / camera — quan trọng vì `%accuracy` đơn thuần che giấu thực tế vận hành trong domain sự kiện hiếm (té ngã, xâm nhập) |
| Tỉ lệ VLM sửa đúng false positive từ rule engine | `triage_verdict='false_positive'` / tổng số alert có `operator_verdict='false'` — case 2 Gate 2 (push-up bị nhận nhầm fall) là ví dụ đơn lẻ, nên tổng hợp thành số liệu qua toàn bộ lịch sử |

**Áp dụng ngay được:** mỗi alert trong `alert_history` đã có `operator_verdict` (real/false) làm
ground truth từ HITL — đủ để tính toàn bộ bảng trên bằng 1 script query DB, không cần thu thập dữ
liệu mới.

## Nhóm 2 — Chất lượng tracking (MOT-specific)

Chuẩn từ MOTChallenge (MOT16/17/20), benchmark MOT được trích dẫn nhiều nhất:

| Metric | Ý nghĩa | Vì sao liên quan trực tiếp |
|---|---|---|
| IDF1 | Độ chính xác gán ID nhất quán theo thời gian | Track_id đổi giữa chừng khi đang ngã → `fall_dwell_sec` reset sai → bỏ sót cảnh báo thật |
| ID Switches (IDSW) | Số lần track bị đổi ID khi vẫn là cùng 1 đối tượng | Phá hỏng trực tiếp owner-tracking của rule `abandoned_object` |
| MOTA | Độ chính xác tổng hợp (miss + false positive + ID switch) | Chỉ số tổng quan để so sánh giữa các cấu hình tracker |
| MT/ML | % track theo dõi được ≥80% / <20% thời lượng | Cho biết tracker có mất dấu người khi bị che khuất không |

**Chưa áp dụng được ngay:** cần video có nhãn track_id ground-truth, tốn công gán nhãn hơn hẳn so
với Nhóm 1. Với quy mô hiện tại, khuyến nghị đo định tính thay thế: đếm số lần `track_id` đổi bất
thường trong cùng 1 clip liên tục (lấy được từ log inference sẵn có, không cần gán nhãn mới) —
không phải MOTA/IDF1 đầy đủ, nhưng phát hiện được vấn đề tracking rõ nhất (ID switch giữa một
chuỗi hành động liên tục) mà không cần đầu tư thêm dữ liệu.

## Nhóm 3 — Chất lượng suy luận VLM

Từ paper video-LLM anomaly reasoning (Holmes-VAD và các benchmark VideoLLM tương tự), áp dụng cho
đúng bước VLM triage đang có trong hệ thống (`CompatibleVLMTriage`):

- **Accuracy phân loại** (real/false_positive/uncertain) so với ground truth guard
- **Latency** — `VLM_TIMEOUT_SEC=20s` đã set làm ngưỡng, nên đo p50/p95 thật thay vì chỉ biết
  ngưỡng timeout
- **Chi phí** (token/request)

**Áp dụng ngay được:** ground truth verdict đã có; latency/token đã được ghi trong `TriageVerdict`
— chỉ cần tổng hợp, không cần đo mới.

## Để sau

Hai nhóm sau chỉ đáng đầu tư khi đã qua giai đoạn demo, không phải ưu tiên hiện tại:

- **Hiệu năng hệ thống/hạ tầng** (throughput, latency p50/p95/p99, mAP detector) — số throughput/
  latency đã có sẵn trong docstring `stream_manager.py`/`get_status()`, nhưng chưa cần tổng hợp
  thành báo cáo chính thức lúc này; mAP detector cần tập ảnh có nhãn, đầu tư sau.
- **Kết quả vận hành thực tế** (time-to-acknowledge, tỉ lệ alert hết hạn không xử lý) — chỉ có ý
  nghĩa sau khi có dữ liệu vận hành thật qua nhiều ca trực.

## Ưu tiên triển khai

1. **Nhóm 1 (Detection Quality + FAR)** — ưu tiên cao nhất. Ground truth đã có sẵn trong
   `alert_history`, chỉ cần 1 script tổng hợp Precision/Recall/F1/FAR trên toàn bộ lịch sử thay
   cho báo cáo 5 case thủ công hiện tại.
2. **Nhóm 3 (VLM reasoning)** — dữ liệu latency/token đã có sẵn trong schema, chỉ cần tổng hợp.
3. **Nhóm 2 (MOT tracking)** — làm ở mức định tính (đếm ID switch bất thường qua log) trước khi
   đầu tư gán nhãn ground-truth cho MOTA/IDF1 đầy đủ.
