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

> **Trạng thái 2026-08-29:** phần mô tả bên dưới là hiện trạng *tại thời điểm viết báo cáo*. Mục
> #0 (tách luồng) và #5 (transport nén liên khung) từ đó đã được triển khai — xem mục #5 và phần
> "Khuyến nghị lộ trình" ở cuối. Phần về batching/TensorRT (#2) vẫn còn đúng nguyên.

Đề xuất giả định hệ thống đã tách Luồng 1 (display 480p) và Luồng 2 (AI). **Thực tế lúc đó: chưa
tách.**

- `CameraStreamWorker` (dùng bởi `MultiStreamManager`) mở RTSP bằng `cv2.VideoCapture` — decode
  CPU, không dùng NVDEC — và chạy **một** pipeline duy nhất: mỗi frame vừa bị YOLO detect vừa bị
  vẽ overlay (`draw_overlays=True`) để làm preview. *(Đã sửa: nay có `_display_loop` riêng dựng
  preview 480p ở 20fps, không chạy YOLO.)*
- `/api/cameras/{id}/stream` phát MJPEG lấy từ đúng frame đã annotate đó
  (`service.latest_jpeg(camera_id)`) — không có sub-stream 480p riêng, không WebRTC, overlay vẽ ở
  server (`cv2.rectangle` trong `VisionPipeline`), không phải "Frontend tự vẽ bounding box từ
  metadata JSON" như đề xuất mô tả. *(Đã sửa một phần: có `/cameras/{id}/video` phát H.264/fMP4
  qua WebSocket; MJPEG giữ làm fallback. Overlay vẫn vẽ ở server — xem ghi chú cuối mục #5.)*
- Không có batching qua nhiều camera: khi có CUDA, `_default_inference_concurrency()` trả về
  **1** — nghĩa là các camera trên GPU hiện đang chạy tuần tự, không gộp batch. Docstring của
  chính class ghi rõ: *"Single mid-range GPU + batching: 20-50 cameras (needs a batched detector,
  which this class does NOT implement)"* — đây là giới hạn đã biết trước, không phải phát hiện
  mới. **Điểm này vẫn còn nguyên, và là việc tiếp theo cần làm.**

## Đánh giá từng đề xuất

| # | Đề xuất | Khả thi kỹ thuật | Effort trên codebase này | Đáng làm ở quy mô hiện tại? |
|---|---|---|---|---|
| 0 | Tách Luồng 1 (480p display) / Luồng 2 (5-10fps AI) | Cao | Vừa | **Đã làm** |
| 1 | NVDEC zero-copy decode (PyNvVideoCodec/Decord) | Cao (có T4, hỗ trợ NVDEC) | Rất cao | Không, trừ khi CPU decode đã thành nút thắt đo được |
| 2 | TensorRT + FP16/INT8 | Cao | Vừa | **Có, sau khi có nhiều camera hơn CPU/GPU hiện chịu nổi** |
| 3 | Triton Inference Server | Cao (nhưng đổi hẳn kiến trúc deploy) | Rất cao | Không ở quy mô này |
| 4 | Cascade Inference 2 tầng | Cao | Thấp–Vừa | Đã có bản "nhẹ" tương đương, xem bên dưới |
| 5 | Transport nén liên khung (WebRTC / H.264-fMP4) | Cao | Vừa | **Đã làm** — H.264/fMP4 qua WebSocket, xem bên dưới |

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

### 5. Transport video có nén liên khung — ĐÃ TRIỂN KHAI (H.264/fMP4 qua WebSocket)

**Cập nhật 2026-08-29.** Đánh giá ban đầu ở trên xếp mục này ưu tiên thấp với lý do "nghẽn ở
compute, không phải băng thông". Điều đó đúng với *throughput của server*, nhưng bỏ sót một nút
thắt riêng biệt ở phía client: sau khi làm mục #0, server đã phát ~20fps nhưng browser chỉ render
được 10-14fps. Nguyên nhân không phải băng thông LAN mà là **chi phí giải mã**: MJPEG gửi mỗi
frame như một ảnh JPEG độc lập, và browser phải decode từng ảnh một bằng CPU. Tăng tốc inference
không sửa được triệu chứng này.

Đã triển khai (`src/vision/h264_encoder.py`, `/api/v1/cameras/{id}/video`,
`src/frontend/src/components/VideoFeed.jsx`):

- Server encode H.264 từ chính khung hình mà display loop đã dựng, xuất fMP4
  (`frag_every_frame` → mỗi frame một fragment, độ trễ ~1 frame) và đẩy qua WebSocket.
- Browser nạp vào MediaSource Extensions, giải mã bằng GPU.
- Ưu tiên `h264_nvenc` (T4), tự động lùi về `libx264` nếu không mở được.
- **MJPEG được giữ nguyên làm fallback**, không phải code chết: dùng cho browser không hỗ trợ
  MediaSource, cho host không có encoder H.264 khả dụng, khi socket rớt, và cho `/snapshot`.
  Server báo `{"type":"fallback"}` tường minh thay vì để client chờ trên một socket câm.

Đo thực tế trên 4 clip mẫu của dự án (480p, 20fps, `libx264` CPU):

| Clip | Độ phân giải | MJPEG | H.264 | Giảm |
|---|---|---|---|---|
| abandoned.mp4 | 600x480 | 4.58 Mbps | 1.25 Mbps | 3.7x |
| drop_and_go.mov | 853x480 | 11.28 Mbps | 1.30 Mbps | 8.7x |
| xamnhap_01.mov | 854x480 | 12.67 Mbps | 1.33 Mbps | 9.5x |
| xamnhap_02.mov | 854x480 | 12.84 Mbps | 1.38 Mbps | 9.3x |
| **Tổng 4 camera** | | **41.4 Mbps** | **5.3 Mbps** | **7.9x** |

Chi phí thêm: ~2.3-3.4 ms/frame encode trên CPU, chạy trong display thread nên **không** đụng
vào ngân sách inference. Trên T4 với NVENC con số này gần bằng 0.

Ghi chú về WebRTC/MediaMTX như đề xuất gốc: không chọn, vì cả hai đều đòi thêm signaling/ICE
(aiortc) hoặc thêm hẳn một service vào deploy (MediaMTX) để đổi lấy phần lợi ích còn lại là độ
trễ thấp hơn vài trăm ms — thứ chưa thành vấn đề ở quy mô LAN hiện tại. fMP4 qua WebSocket lấy
được gần hết lợi ích băng thông và GPU-decode mà không thêm dependency vận hành nào.

## Khuyến nghị lộ trình

**0. ✅ Tách luồng display / luồng AI** — luồng 1 dựng preview 480p ở 20fps
(`CameraStreamWorker._display_loop`), luồng 2 chạy vision/VLM ở 5-10fps. Đã xong.

**1. ✅ Transport video có nén liên khung** (mục #5) — H.264/fMP4 qua WebSocket, giữ MJPEG làm
fallback. Đã xong; giảm 7.9x băng thông và bỏ được chi phí decode JPEG per-frame ở client, tức
là gỡ đúng nút thắt khiến browser chỉ render 10-14fps trong khi server phát 20fps.

Hai bước trên là hai nút thắt **khác nhau** và không thay thế cho nhau: #0 gỡ nghẽn ở phía
server (inference chiếm hết ngân sách của preview), #1 gỡ nghẽn ở phía client (chi phí decode).
Đây là lý do làm #1 trước phần compute bên dưới — không có cải tiến inference nào sửa được việc
browser không kịp giải mã.

**2. Bỏ giới hạn `max_concurrent_inferences=1` + gom batch nhiều camera + export TensorRT FP16**
(mục #2) — bước tiếp theo, và là bước đầu tiên thật sự tăng *số camera* một GPU chịu được. Lưu ý
đây là trục hoàn toàn khác với #1: nó không làm client render mượt hơn, nó cho phép thêm camera.

3. Đo lại `get_status()["average_inference_capacity_fps"]` sau bước 2 trước khi cân nhắc NVDEC
   hay Triton — cả hai đều là effort rất cao, chỉ nên làm khi có số đo cụ thể cho thấy decode
   hoặc điều phối request là nút thắt, không phải suy đoán trước.
4. WebRTC thật (aiortc/MediaMTX) chỉ khi cần độ trễ dưới ~200ms hoặc phục vụ qua internet với
   bitrate thích ứng — fMP4 qua WebSocket đã lấy phần lợi ích lớn mà không thêm service nào.

