t

# Worklog — Team P-091

> Ghi lại tất cả công việc đã làm theo ngày.

---

## 2026-07-29 (Thứ Ba)

| Member          | Task                                             | Status | Output                                                                               | Time       |
| --------------- | ------------------------------------------------ | ------ | ------------------------------------------------------------------------------------ | ---------- |
| Lục Minh Đức | Viết báo cáo cho buổi Mentor Duty thứ nhất | Done   | File đã được xóa để dùng thống nhất 1 file tài liệu (update 01/08/2026) | 29/07/2026 |

---

## 2026-08-01 (Thứ Bảy)

| Member                 | Task                                         | Status | Output                                                  | Time       |
| ---------------------- | -------------------------------------------- | ------ | ------------------------------------------------------- | ---------- |
| Lê Nguyễn Minh Đức | Viết báo cáo cho buổi Mentor Duty thứ 2 | Done   | [docs/reports/2026-08-01.md](docs/reports/2026-08-01.md) | 01/08/2026 |

---

## 2026-08-02 (Chủ Nhật)

| Member                 | Task                                                | Status | Output                        | Time       |
| ---------------------- | --------------------------------------------------- | ------ | ----------------------------- | ---------- |
| Lê Nguyễn Minh Đức | Viết các tài liệu deliverables cho Gate 1       | Done   | [docs/gate 1](<docs/gate%201>) | 02/08/2026 |
| Lục Minh Đức        | Chỉnh sửa các tài liệu deliverables cho Gate 1 | Done   | [docs/gate 1](<docs/gate%201>) | 02/08/2026 |

---

## 2026-08-05 (Thứ Tư)

Học hỏi kiến trúc từ DeepCamera (SharpAI) và port **ý tưởng, không phải code** sang `src/vision/`.
Fall stack của DeepCamera là bản clone bên thứ ba chạy Python 3.8 / torch 1.5 (không port được sang
Python 3.11 / torch 2.13 của ta), còn motion check của nó là code chết (`blenderRegion` rỗng, lời gọi
bị comment, `has_motion` hardcode `true`). ReID / Face ID của DeepCamera bị **PRD §5 cấm**.

| Member            | Task                                                                                                                                                                                                                                                          | Status | Output                                                                                                   | Time       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------- | ---------- |
| _(điền tên)_ | Phát hiện ngã bằng pose — cascade 2 tầng: lọc thô bằng tỉ lệ bbox (`fall_pose_gate_ratio` 0.8, lỏng hơn ngưỡng 1.2), pose chỉ chạy trên crop được lọc. Phân biệt được nằm ngã vs ngồi xổm.                                   | Done   | [src/vision/pose.py](src/vision/pose.py), [src/vision/fall_classifier.py](src/vision/fall_classifier.py)   | 05/08/2026 |
| _(điền tên)_ | Motion gating giảm độ trễ. Bỏ qua frame chỉ khi**đủ 3 điều kiện**: không chuyển động **và** không còn track sống **và** chưa hết heartbeat — nếu chỉ xét chuyển động sẽ xóa mất nhãn `abandoned_object`. | Done   | [src/vision/motion.py](src/vision/motion.py)                                                              | 05/08/2026 |
| _(điền tên)_ | Cooldown cảnh báo theo`(camera_id, label)` để 1 track nhấp nháy không làm ngập bảo vệ (PRD §4 — giảm báo động giả).                                                                                                                       | Done   | [src/vision/rule_engine.py](src/vision/rule_engine.py)                                                    | 05/08/2026 |
| _(điền tên)_ | Khôi phục mock fallback của detector (`is_mock`, `allow_mock_fallback`) + sửa đường dẫn weights tuyệt đối thành tương đối repo.                                                                                                           | Done   | [src/vision/detector.py](src/vision/detector.py)                                                          | 05/08/2026 |
| _(điền tên)_ | Thêm 25 test mới (chạy không cần weights) + đo A/B độ trễ.                                                                                                                                                                                           | Done   | [tests/test_pose_fall.py](tests/test_pose_fall.py), [tests/test_motion_gate.py](tests/test_motion_gate.py) | 05/08/2026 |

### Đo A/B độ trễ của motion gate (PRD §4 chấm điểm latency)

Cùng một máy, yolo11n, CPU. Gate **mặc định TẮT** nên bảng capacity cũ vẫn còn giá trị.

| Cảnh                                                  | Gate | FPS              | Số lần inference | % frame bỏ qua | Sự kiện |
| ------------------------------------------------------ | ---- | ---------------- | ------------------ | --------------- | --------- |
| Footage đông người (`input_test.mp4`, 200 frame) | OFF  | 5.36             | 200                | 0.0%            | 2         |
| Footage đông người (`input_test.mp4`, 200 frame) | ON   | 5.72             | 200                | 0.0%            | 2         |
| Cảnh tĩnh 720p (có nhiễu cảm biến, 120 frame)    | OFF  | 5.62             | 120                | 0.0%            | —        |
| Cảnh tĩnh 720p (có nhiễu cảm biến, 120 frame)    | ON   | **136.89** | 4                  | **96.7%** | —        |

**Kết luận trung thực:** khi có người di chuyển, gate không bỏ qua frame nào → không nhanh hơn
(chênh 6% là nhiễu đo). Giá trị thật nằm ở cảnh vắng (hành lang lúc 3h sáng): **nhanh ~24 lần**.
Quan trọng nhất: cả hai lần chạy đều cho ra **đúng 2 sự kiện giống nhau** — bật gate không làm
thay đổi kết quả phát hiện. Phép đo cũng xác nhận ngưỡng 21/255 coi nhiễu cảm biến là "đứng yên".

**Toàn bộ test:** 65 passed / ~13s. `ruff check src/ tests/` sạch.

---

## 2026-08-06 (Thứ Năm)

Nối `src/vision/` vào một **web app giám sát hoàn chỉnh**: FastAPI + React (Vite), 3 màn hình
(Grid View, Camera Setup, Alert Pop-up). Trước hôm nay không có gì trong `src/api/` import
`src.vision` — pipeline chạy được nhưng chỉ qua script.

| Member            | Task                                                                                                                                                                                                                                                                           | Status | Output                                                                                                           | Time       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------- | ---------- |
| _(điền tên)_ | Hàm VLM thật, tách theo provider (`claude` / `openai` / `stub`). Nhận base64/bytes/đường dẫn ảnh, trả 1 câu mô tả sự cố. Không bao giờ raise — provider lỗi thì trả caption dự phòng kèm `error`, để VLM chết không nuốt mất cảnh báo. | Done   | [src/services/vlm.py](src/services/vlm.py)                                                                        | 06/08/2026 |
| _(điền tên)_ | `MonitoringService` — chỗ nối giữa `src/vision/` và `src/api/`: quản lý camera, buffer preview, ring buffer 50 alert, fan-out SSE, lưu `data/cameras.json`.                                                                                                    | Done   | [src/services/monitoring.py](src/services/monitoring.py)                                                          | 06/08/2026 |
| _(điền tên)_ | REST + MJPEG + SSE routes; upload video (tên file do server sinh, chặn path traversal), snapshot, acknowledge.                                                                                                                                                               | Done   | [src/api/vision_routes.py](src/api/vision_routes.py), [src/models/camera_schemas.py](src/models/camera_schemas.py) | 06/08/2026 |
| _(điền tên)_ | Frontend React: grid responsive, canvas vẽ polygon vùng cấm (toạ độ chuẩn hoá`[0,1]`), modal cảnh báo có hàng đợi.                                                                                                                                             | Done   | [src/frontend/](src/frontend/)                                                                                    | 06/08/2026 |
| _(điền tên)_ | `ZoneManager.set_zones` / `clear_zones` để polygon vẽ trên UI tới được worker đang chạy. Lưu list rỗng **có ý nghĩa khác** với không lưu gì (tránh rò zone wildcard `"*"`).                                                                  | Done   | [src/vision/zone_config.py](src/vision/zone_config.py)                                                            | 06/08/2026 |
| _(điền tên)_ | 18 test mới cho service + routes (không cần weights).                                                                                                                                                                                                                       | Done   | [tests/test_monitoring_api.py](tests/test_monitoring_api.py)                                                      | 06/08/2026 |

### Đo độ trễ VLM (PRD §4 chấm điểm latency)

Cùng một frame (`fall.mp4` frame 200), cùng prompt 1 câu, model `claude-opus-5`:

| Cấu hình                                    | Độ trễ         | Caption          |
| --------------------------------------------- | ----------------- | ---------------- |
| `output_config={"effort":"low"}`            | 54210 ms          | tương đương |
| `thinking={"type":"disabled"}` + effort low | **5487 ms** | tương đương |

Suy luận mở rộng không mua được gì cho "mô tả ảnh này", trong khi call này chặn modal của bảo vệ.
Đã chốt `thinking={"type":"disabled"}`, có fallback nếu API từ chối tham số.

> **Ghi chú (buổi chiều cùng ngày):** phép đo trên là **lịch sử**. Chiều 06/08 team chốt chỉ dùng
> OpenAI key, nhánh Claude đã bị gỡ khỏi `vlm.py` nên `thinking` / `output_config` không còn trong
> code. Giữ lại bảng này vì nó là bằng chứng cho quyết định latency, không phải mô tả code hiện tại.

### Chạy thử end-to-end (footage thật, API key thật)

`fall.mp4`, polygon vùng cấm, ~12 giây: **384 frame, 11.44 fps capacity, 6 alert** (5 thật + 1
`simulated` — cờ `simulated` và `mock_after_sec` đã bị gỡ chiều cùng ngày, xem mục dưới). Lý do từ
rule engine là thật, ví dụ `"track_63 stayed collapsed for 12.9s (threshold 3.0s) — bbox aspect ratio 1.54"`. Đáng chú ý: ở một event bị gán nhầm nhãn
`abandoned_object`, VLM trả lời *"A person has collapsed and is lying motionless on the floor —
the system has misclassified them"* — đúng giá trị giảm báo động giả mà PRD §4 yêu cầu.

**Toàn bộ test:** 83 passed / ~15s. `ruff check src/ tests/` sạch. Frontend build 1.78s.

---

## 2026-08-06 (Thứ Năm — buổi chiều)

Dọn dẹp theo 3 quyết định của team: **chỉ dùng OpenAI key**, **bỏ mock detector**, **đưa
`frontend/` vào trong `src/`**.

| Member            | Task                                                                                                                                                                                                                                                                                                                  | Status | Output                                                                                                               | Time       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------- | ---------- |
| _(điền tên)_ | VLM chỉ còn OpenAI + stub offline. Gỡ toàn bộ nhánh Claude/Anthropic (`_describe_with_claude`, `anthropic` trong requirements, `ANTHROPIC_API_KEY` / `VLM_PROVIDER` / `VLM_MODEL` trong config & `.env.example`). Provider tự chọn: có `OPENAI_API_KEY` → `openai`, không có → `stub`. | Done   | [src/services/vlm.py](src/services/vlm.py), [src/config.py](src/config.py)                                             | 06/08/2026 |
| _(điền tên)_ | Bỏ mock detector.`_detect_mock` / `is_mock` / `allow_mock_fallback` không còn; thiếu weights giờ **raise `FileNotFoundError`** ngay lúc khởi tạo thay vì bịa ra detection. `model_path=None` vẫn là "null detector" trả `[]` cho các test chỉ chạy tầng rule.                      | Done   | [src/vision/detector.py](src/vision/detector.py)                                                                      | 06/08/2026 |
| _(điền tên)_ | Bỏ luôn demo aid`mock_after_sec` / cờ `simulated` khỏi schema, `MonitoringService` và UI — mọi alert hiển thị từ nay đều là sự kiện thật.                                                                                                                                                       | Done   | [src/models/camera_schemas.py](src/models/camera_schemas.py), [src/services/monitoring.py](src/services/monitoring.py) | 06/08/2026 |
| _(điền tên)_ | Chuyển`frontend/` → `src/frontend/`. Cập nhật `docker-compose.yml` (build context), `.gitignore`, `.dockerignore` (loại `src/frontend/` khỏi image backend) và `ruff.toml` (`extend-exclude`, tránh lint `node_modules`).                                                                   | Done   | [src/frontend/](src/frontend/)                                                                                        | 06/08/2026 |
| _(điền tên)_ | Khai báo`openai` thành dependency trực tiếp trong `requirements.txt` — `vlm.py` import thẳng, không nên phụ thuộc vào việc `langchain-openai` kéo nó về.                                                                                                                                       | Done   | [requirements.txt](requirements.txt)                                                                                  | 06/08/2026 |

**Lưu ý về vận hành:** vì mock đã bỏ, một bản clone mới (weights `*.pt` bị gitignore) sẽ **không
khởi động được** cho tới khi có `src/vision/weights/yolo11n.pt`. Đây là chủ ý — hệ giám sát báo
"không thấy gì" trong khi thực chất chưa nạp model là kiểu hỏng nguy hiểm nhất.

`test_camera_id_propagates_into_detections` trước đây dựa vào mock để có detection trên frame
đen; nay chạy trên footage thật trong `data/vision/test/` và tự skip khi không có clip (thư mục
`data/` bị gitignore nên CI sẽ skip).

**Toàn bộ test:** 83 passed / ~27s. `ruff check src/ tests/` sạch. Frontend build 1.56s tại vị
trí mới.

---
