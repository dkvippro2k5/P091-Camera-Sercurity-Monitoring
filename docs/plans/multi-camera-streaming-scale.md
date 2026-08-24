# Kiểm tra tính khả thi: kiến trúc tối ưu multi-camera streaming

## Context

User đưa ra một đề xuất kiến trúc "chuẩn công nghiệp" (tách Luồng 1 Display / Luồng 2 AI, cộng
5 kỹ thuật enterprise: NVDEC zero-copy decode, TensorRT + INT8/FP16, Triton Dynamic Batching,
Cascade Inference 2 tầng, WebRTC/MediaMTX) và hỏi tính khả thi để scale multi-camera. Tài liệu
này đối chiếu từng phần với code thật đang chạy (`src/vision/stream_manager.py`,
`src/vision/detector.py`, `src/api/vision_routes.py`, `Dockerfile`), chỉ ra cái gì đã có, cái gì
chưa, và mức độ đáng làm so với quy mô dự án hiện tại.

**Bối cảnh hạ tầng thật:** deploy trên 1 GPU Tesla T4 (`Dockerfile` dòng 1: "CUDA, cuDNN, PyTorch
and Python are pinned together for the Tesla T4 deployment"), model đang dùng là file `.pt`
(PyTorch/ultralytics), **không** phải TensorRT `.engine`. Đây là dự án capstone của một team nhỏ,
không phải hệ thống VMS thương mại — điều này ảnh hưởng trực tiếp đến cái gì "khả thi" nghĩa là
gì (effort/lợi ích, không chỉ khả năng kỹ thuật thuần tuý).

## Kiến trúc hiện tại — đối chiếu với đề xuất "2 luồng"

Đề xuất giả định hệ thống đã tách Luồng 1 (display 480p) và Luồng 2 (AI). **Thực tế: chưa tách.**

- `CameraStreamWorker` (dùng bởi `MultiStreamManager`) mở RTSP bằng `cv2.VideoCapture` — decode
  CPU, không dùng NVDEC — và chạy **một** pipeline duy nhất: mỗi frame vừa bị YOLO detect vừa bị
  vẽ overlay (`draw_overlays=True`) để làm preview.
- `/api/cameras/{id}/stream` (`vision_routes.py:194-228`) phát MJPEG lấy từ đúng frame đã annotate
  đó (`service.latest_jpeg(camera_id)`) — không có sub-stream 480p riêng, không WebRTC, overlay
  vẽ ở server (`cv2.rectangle` trong `VisionPipeline`), không phải "Frontend tự vẽ bounding box từ
  metadata JSON" như đề xuất mô tả.
- Không có batching qua nhiều camera: khi có CUDA, `_default_inference_concurrency()` trả về
  **1** — nghĩa là các camera trên GPU hiện đang chạy tuần tự, không gộp batch. Docstring của
  chính class ghi rõ: *"Single mid-range GPU + batching: 20-50 cameras (needs a batched detector,
  which this class does NOT implement)"* — đây là giới hạn đã biết trước, không phải phát hiện mới.

Nói cách khác: ý tưởng "tách luồng display/AI" trong đề xuất là đúng hướng và là gap thật của hệ
thống hiện tại — không phải thứ đã có sẵn cần tinh chỉnh.

## Đánh giá từng đề xuất

| # | Đề xuất | Khả thi kỹ thuật | Effort trên codebase này | Đáng làm ở quy mô hiện tại? |
|---|---|---|---|---|
| 0 | Tách Luồng 1 (480p display) / Luồng 2 (5-10fps AI) | Cao | Vừa | **Có — ưu tiên cao nhất** |
| 1 | NVDEC zero-copy decode (PyNvVideoCodec/Decord) | Cao (có T4, hỗ trợ NVDEC) | Rất cao | Không, trừ khi CPU decode đã thành nút thắt đo được |
| 2 | TensorRT + FP16/INT8 | Cao | Vừa | **Có, sau khi có nhiều camera hơn CPU/GPU hiện chịu nổi** |
| 3 | Triton Inference Server | Cao (nhưng đổi hẳn kiến trúc deploy) | Rất cao | Không ở quy mô này |
| 4 | Cascade Inference 2 tầng | Cao | Thấp–Vừa | Đã có bản "nhẹ" tương đương, xem bên dưới |
| 5 | WebRTC qua MediaMTX/go2rtc | Cao | Vừa | Có, nhưng ưu tiên thấp hơn #0 và #2 |

### 0. Tách 2 luồng — làm trước, ROI cao nhất

Đây là thay đổi kiến trúc **không cần GPU mới, không cần thư viện mới**, chỉ cần:
- Nếu camera hỗ trợ RTSP sub-stream (hầu hết camera IP đều có 2 luồng: main + sub), pull thêm
  luồng sub 480p bằng `cv2.VideoCapture` riêng, chỉ dùng để phát MJPEG — không đưa qua YOLO.
- Luồng chính (main stream) chỉ dùng để detect ở 5-10 FPS, **không** cần vẽ overlay hay encode
  JPEG mỗi frame cho preview nữa — giảm tải CPU encode đang bị gánh chung với inference hiện nay.

Rủi ro thực tế: camera demo/test hiện dùng trong dự án là file video hoặc webcam giả lập
(`loop_finite_sources`, `CameraSetup.jsx`) — không phải camera IP thật có sẵn 2 luồng RTSP. Cần
xác nhận nguồn camera thật của team có sub-stream hay không trước khi triển khai, nếu không sẽ
phải tự downscale bằng `cv2.resize` (vẫn giảm được overhead encode, nhưng không giảm được chi phí
decode gốc).

### 1. NVDEC zero-copy decode — kỹ thuật đúng, effort không hợp lý ở đây

Đúng là `cv2.VideoCapture` decode bằng CPU và tốn copy CPU→GPU. Nhưng:
- Đây là thay đổi tầng thấp nhất của toàn bộ pipeline vision, đòi hỏi viết lại
  `CameraStreamWorker` để nhận tensor thẳng từ VRAM thay vì `np.ndarray`, kéo theo phải viết lại
  `IncidentDetector.detect()`, `MotionGate` (hiện diff trên CPU array), và toàn bộ
  `VisionPipeline` vì chúng đều giả định input là NumPy array trên CPU.
- Team hiện có 3-4 camera thực tế theo capacity đã đo (`stream_manager.py` docstring), không phải
  10-20. Ở quy mô đó, CPU decode chưa phải nút thắt — nút thắt là **YOLO inference chạy tuần tự
  từng camera** (`max_concurrent_inferences=1` trên GPU).
- **Khuyến nghị:** không làm bây giờ. Đo lại `average_inference_capacity_fps` sau khi làm #0 và
  #2 trước; nếu vẫn nghẽn ở khâu decode (không phải inference) mới xét NVDEC.

### 2. TensorRT + FP16/INT8 — khả thi và đáng làm, nhưng sau khi có nhu cầu batching thật

`ultralytics` hỗ trợ sẵn `model.export(format="engine", half=True)` để xuất TensorRT — không cần
tự viết pipeline TensorRT từ đầu như đề xuất ngụ ý. T4 hỗ trợ FP16 tốt (Tensor Cores), INT8 cần
thêm bước calibration (cần tập ảnh đại diện, tốn thêm công đoạn).

Lợi ích thật: giảm latency mỗi lần infer, nhưng **không tự động giải quyết việc hiện đang chạy
tuần tự 1 camera/lần trên GPU** — muốn tận dụng TensorRT cho nhiều camera cùng lúc vẫn cần sửa
`MultiStreamManager` để gom batch trước khi gọi model (`torch.stack` nhiều frame), tức là cần làm
cùng lúc với việc bỏ giới hạn `max_concurrent_inferences=1`.

### 3. Triton Inference Server — over-engineering ở quy mô này

Triton là lựa chọn đúng khi có hàng chục stream trở lên và cần tách server inference khỏi server
API. Ở đây toàn bộ hệ thống là 1 FastAPI process phục vụ vài camera; thêm Triton nghĩa là thêm
1 service riêng (gRPC, `config.pbtxt`, container riêng), tăng đáng kể độ phức tạp vận hành so với
lợi ích tại quy mô hiện tại. Không khuyến nghị trừ khi số camera mục tiêu tăng vượt hẳn khả năng
1 GPU T4 xử lý bằng batching đơn giản (mục #2).

### 4. Cascade Inference 2 tầng — về bản chất đã có, chỉ chưa dùng model riêng cho tầng 2

Điểm quan trọng: `RuleEngine` (`src/vision/rule_engine.py`, xem các tham số
`fall_aspect_ratio`, `fall_velocity_ratio`, `fall_optical_flow_ratio`... trong
`stream_manager.py:240-257`) **đã** đóng vai trò tầng lọc thứ 2 kiểu cascade — chỉ khi YOLO
(tầng 1) phát hiện người có dấu hiệu bất thường về hình dạng/vận tốc, hệ thống mới tính toán các
chỉ số nặng hơn và kích hoạt VLM triage. Khác biệt so với đề xuất: tầng 2 hiện là **rule-based
trên bbox**, không phải **model Pose Estimation** riêng. Thêm Pose thật là nâng cấp hợp lý về độ
chính xác, nhưng là việc "làm tinh hơn" chứ không phải "thêm kiến trúc cascade còn thiếu" —
framework cascade đã tồn tại.

### 5. WebRTC/MediaMTX — đúng hướng, làm sau #0 và #2

MJPEG hiện tại (`multipart/x-mixed-replace`) tốn băng thông hơn WebRTC đúng như đề xuất nêu, và
việc browser tự giải mã bằng GPU client cũng đúng là lợi thế thật. Nhưng đây là thay đổi ở tầng
transport, độc lập với nút thắt hiện tại (nghẽn ở compute, không phải băng thông — team đang chạy
local/LAN theo ngữ cảnh dự án, chưa phải nhiều client ở xa cùng xem). Ưu tiên thấp hơn 2 mục trên;
nên làm khi đã có nhiều viewer đồng thời hoặc cần deploy qua internet thật.

## Khuyến nghị lộ trình

1. **Tách luồng display/AI (mục #0)** — effort thấp nhất, giải quyết đúng vấn đề "AI đang gánh cả
   phần encode preview". Làm trước tiên.
2. **Bỏ giới hạn `max_concurrent_inferences=1` + gom batch nhiều camera + export TensorRT FP16**
   (mục #2, gắn liền nhau) — đây là nâng cấp thật sự tăng số camera/GPU chịu được, đúng cái đề
   xuất đang nhắm tới.
3. Đo lại `get_status()["average_inference_capacity_fps"]` sau bước 1-2 trước khi cân nhắc NVDEC
   hay Triton — cả hai đều là effort rất cao, chỉ nên làm khi có số đo cụ thể cho thấy decode
   hoặc điều phối request là nút thắt, không phải suy đoán trước.
4. WebRTC/MediaMTX khi cần phục vụ nhiều người xem cùng lúc hoặc qua mạng ngoài LAN.

**Không khuyến nghị** Triton và NVDEC ở giai đoạn hiện tại của dự án — đúng về mặt kỹ thuật cho
một VMS thương mại quy mô lớn, nhưng effort không tương xứng với quy mô camera và team hiện có.
