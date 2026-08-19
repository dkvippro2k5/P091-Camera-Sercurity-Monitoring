# Việc cần cải thiện sau đánh giá Gate 2 (tổng hợp 2 vòng review)

## Context

Sau khi Gate 2 merge (`main @ d0bf54e`), có 2 vòng đánh giá độc lập trên cùng bộ thay đổi
(fall detection ngã dọc, LangGraph incident workflow, HITL operator note, notification,
hạ tầng CSDL/event bus/evidence store):

1. **Đánh giá kỹ thuật trực tiếp** (đọc code thật, verify từng claim) — 6 hạng mục, mỗi
   hạng mục có điểm tốt + điểm chưa tốt + đề xuất cụ thể.
2. **Đánh giá phản biện qua Codex** (`danhgia.md`) — tập trung vào tính chặt chẽ của bằng
   chứng (implementation vs verified vs production-ready), đặt ra các câu hỏi mở về
   notification outbox pattern, Redis Pub/Sub vs Streams, login limiter phân tán. Các câu
   hỏi mở đó đã được trả lời bằng cách đọc code thật (xem mục "Đã xác nhận" bên dưới).

Điểm chung của cả 2 vòng: **kiến trúc đã vượt mức prototype, phần lớn đã wired thật vào
runtime** (không phải code chết) — điểm yếu còn lại là **thiếu bằng chứng kiểm thử định
lượng**, không phải thiếu tính năng.

## Đã xác nhận đúng — không cần sửa (trả lời câu hỏi mở của Codex)

- **Notification outbox pattern đã đúng.** `monitoring.py:495-521`: `record_alert()` persist
  DB trước, `notification_worker.submit()` chạy sau, async, không block detection loop.
  Codex lo ngại "gửi đồng bộ trong detection loop" — không đúng với code thật.
- **Test suite đã mock detector đúng cách.** 16 chỗ dùng `IncidentDetector(model_path=None)`
  (null detector, không cần weight); chỉ `test_vision.py` cần detector thật — đúng pattern
  Codex khuyến nghị (unit test dùng fake, chỉ smoke test cần model thật).

## Việc cần làm — theo mức ưu tiên

### 🔴 Cao — gap chức năng/bảo mật thật, chi phí sửa thấp-trung bình

1. **Notification: bổ sung kênh email, hoặc xác nhận chính thức Zalo OA thay thế.**
   Điểm yếu được cả 2 vòng đánh giá đồng thuận là quan trọng nhất. Đề xuất: tách
   `NotificationProvider` protocol (theo đúng pattern `VLMTriage`/`LLMReasoner` project đã
   dùng), cấu hình kênh qua `NOTIFICATION_CHANNELS=email,zalo`. Không cần bỏ Zalo.
2. **Login limiter chỉ in-memory** (`login_limiter.py:9-18`, `dict` + `threading.Lock`,
   không Redis). Nếu deploy nhiều backend instance (`docker-compose.production.yml` hỗ trợ
   việc này), kẻ tấn công bypass được bằng cách rải request qua nhiều instance. Sửa: chuyển
   state sang Redis nếu multi-instance là kịch bản thật; nếu chỉ chạy 1 instance cho Demo
   Day, chỉ cần **sửa docs** — không gọi đây là "distributed rate limiting".
3. **Recovery check cho fall detection** — rủi ro false positive khi cúi nhặt đồ rồi đứng
   dậy nhanh (`foot_stable_hit` + `dh_hit` có thể cùng trigger 2/3, giống hệt ngã dọc về mặt
   hình học). Đề xuất: nếu track quay lại `_is_confidently_upright()` trong vài giây sau khi
   latch, hủy alert trước khi tới VLM — tận dụng `upright_baseline` đã có sẵn.
4. **Camera registry — kiểm tra SSRF.** Nếu URL RTSP nhập tùy ý và backend chủ động kết nối,
   cần validate/chặn URL nội bộ (đúng câu hỏi Codex nêu, chưa verify trong code — cần kiểm
   tra `camera_registry.py`/`vision_routes.py` phần tạo camera).

### 🟡 Trung bình — nợ kỹ thuật, cần bằng chứng định lượng hơn là code mới

5. **Fall detection thiếu regression test/metric trước-sau.** Cả 2 vòng đánh giá đồng thuận:
   cần N video dương tính + M video âm tính, đo recall/FP rate cụ thể — không chỉ mô tả
   logic. Công cụ đã có (`scripts/extract_benchmark_frames.py` mới viết) — việc còn lại là
   thu thập case + chạy benchmark thật.
6. **LangGraph — chưa có test failure-path** (VLM timeout, lỗi mạng) dù đã có `guarded()`
   wrapper. Cần test: timeout giả lập → route đúng `persist_failure`, không treo worker.
7. **`operator_note`/HITL — chưa verify persistence sau reload, validation, audit trail.**
   Code có đủ 3 lớp (model/API/UI) nhưng chưa có test xác nhận dữ liệu còn nguyên sau khi
   restart server hoặc bị nhập input độc hại (XSS trong textarea 2000 ký tự?).
8. **Redis event bus dùng Pub/Sub, không phải Streams** (`event_bus.py:69-70`). Phạm vi rủi
   ro đã xác nhận: **chỉ ảnh hưởng đồng bộ SSE giữa nhiều replica**, không ảnh hưởng dữ liệu
   incident (đã persist DB riêng, trước khi qua event bus) — mức độ nghiêm trọng thấp hơn
   Codex ngụ ý ban đầu. Nếu cần chống mất update real-time khi multi-instance, đổi sang
   `XADD`/`XREADGROUP`; nếu chỉ 1 instance, không cấp thiết.
9. **PostgreSQL path chưa có test matrix thật** — dual SQLite/Postgres mới chứng minh đổi
   được connection string, chưa chứng minh chạy đúng trên Postgres thật (JSON/JSONB,
   timezone, concurrent writes là các điểm khác biệt hay gây lỗi giữa 2 dialect).
10. **Evidence store "S3-ready" chưa test với S3/MinIO thật** — mới có abstraction + local
    backend hoạt động, cần 1 integration test với MinIO container để xác nhận thật.
11. **CI evidence nên ghi cụ thể hơn** — thay "đã làm trong phiên này" bằng: tên workflow đã
    sửa, commit hash, số test pass/fail, cơ chế cache weight (đã có, chỉ cần ghi rõ trong
    docs Gate 2).

### 🟢 Thấp — cải thiện chất lượng tài liệu, không chặn Gate 2

12. **Áp dụng khung 4 lớp bằng chứng của Codex cho mọi báo cáo Gate tiếp theo:**
    Implementation (file:line) → Integration (nơi gọi trong runtime) → Verification (test +
    kết quả) → Acceptance (tiêu chí đạt/không đạt cụ thể, có số liệu).
13. **Phân biệt rõ 4 mức trạng thái** trong mọi tài liệu nghiệm thu, tránh dùng 1 dấu ✅
    chung chung: `Implemented and verified` / `Implemented, verification pending` /
    `Partially implemented` / `Planned`.
14. **Zalo token hết hạn chỉ log warning, chưa có refresh tự động** — chấp nhận được ở quy mô
    hiện tại (OAuth refresh flow phức tạp hơn tương xứng lợi ích), nhưng nên ghi rõ đây là
    giới hạn đã biết, không phải thiếu sót bị bỏ sót.

## Kết luận

Gate 2: **đạt có điều kiện**, đồng thuận giữa cả 2 vòng đánh giá. Điều kiện quan trọng nhất
còn lại: **quyết định dứt điểm về notification email vs Zalo OA** (mục #1) — đây là gap
chức năng thật (lệch so với quyết định họp), không phải thiếu bằng chứng. Các mục còn lại
(#5-#11) là action item cải thiện tiếp theo, không nhất thiết chặn Gate 2 nếu rubric không
yêu cầu production hardening đầy đủ.
