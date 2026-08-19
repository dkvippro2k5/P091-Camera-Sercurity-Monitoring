# Nghiên cứu: tối ưu stream đa camera (capacity, batching, adaptive sampling)

## Context

`MultiStreamManager` (`src/vision/stream_manager.py`) chạy mỗi camera trên 1 worker thread
riêng. Số liệu đã đo thật (docstring dòng 82-100):

```
1 camera  ->  5.5 fps aggregate
2 cameras ->  9.1 fps aggregate (4.6 each)
4 cameras -> 11.9 fps aggregate (3.0 each)
```

Throughput tổng chững lại quanh 12-15 fps bất kể thêm bao nhiêu worker. Giới hạn team tự ghi
nhận: **CPU-only, 3-4 camera** ở fps hữu dụng; **GPU + batching, ước tính 20-50 camera** (chưa
cài đặt, chưa đo thật); **hàng trăm camera** cần horizontal scaling (N process/GPU).

`MotionGate` (`src/vision/motion.py`) đã có, tắt mặc định — A/B đo được: cảnh tĩnh nhanh hơn
**~24x** (96.7% frame skip), cảnh bận gần như không đổi, cùng số sự kiện phát hiện.

## 3 lớp giải pháp — xếp chồng, không cạnh tranh nhau

1. **Shared batched detector (GPU batching)** — gộp frame nhiều camera thành 1 batch GPU
   (kiểu `nvstreammux` của NVIDIA DeepStream), quyết định 1 node xử lý được bao nhiêu camera.
   Literature: BiSwift đạt 9 stream real-time trên RTX3070 8GB.
2. **Adaptive motion-aware sampling** — mở rộng `MotionGate` từ 1 stream ra toàn hệ thống:
   camera ít chuyển động giảm tần suất infer, nhường compute cho camera có sự kiện. Không tăng
   công suất node, chỉ giảm tải đầu vào — cộng dồn với lớp 1, không thay thế.
3. **Horizontal scaling** — N process, mỗi process phụ trách 1 nhóm camera, GPU-backed. Đổi
   topology triển khai, không giới hạn kiến trúc, chỉ giới hạn chi phí hạ tầng.

Đây là kiến trúc chuẩn của các hệ NVR lớn (DeepStream, Frigate) — cả 3 lớp tồn tại đồng thời,
không phải chọn 1 trong 3.

## Đối chiếu với DeepCamera (github.com/SharpAI/DeepCamera)

DeepCamera dùng 5 nguyên tắc tối ưu (Node.js/Shinobi capture + Celery/Redis worker pool
Python). Đối chiếu với code thật: **3/5 đã có sẵn (khác công cụ, cùng ý tưởng), 1 điểm là lỗ
hổng thật, 1 điểm là đổi cả kiến trúc — chưa cấp thiết.**

| # | Nguyên tắc DeepCamera | Đối chiếu code thật |
|---|---|---|
| 1 | Tách capture nhẹ khỏi AI compute nặng (FFmpeg process + Celery worker) | ✅ Đã có, nhẹ hơn: `_reader_thread`/`_process_thread` riêng trong `CameraStreamWorker` (`stream_worker.py:87-98`), không cần IPC/Redis |
| 2 | Model dùng chung, load 1 lần cho cả pool | ❌ **Lỗ hổng thật** — `_make_default_factory` (`stream_manager.py:183-190`) tạo `IncidentDetector` mới (load lại `YOLO()`) cho **mỗi** camera worker. N camera = N bản sao model trong bộ nhớ |
| 3 | Aggressive frame dropping (1 frame in-flight, luôn lấy mới nhất) | ✅ Đã có: single-slot `_latest_frame` + `CAP_PROP_BUFFERSIZE=1` (`stream_worker.py:67-69, 166-169`); `target_fps` per-camera đã có; `MotionGate` tương đương `SAMPLING_TO_SAVE_ENERGY_MODE` |
| 4 | Backpressure/queue limit (Flower polling + Celery timeout) | ✅ Đạt cùng mục đích bằng cách đơn giản hơn — single-slot buffer tự nhiên chống tồn đọng, không cần hạ tầng giám sát queue |
| 5 | Celery/Redis worker pool, cluster routing | ⏸ Chính là lớp 3 ở trên, nhưng cả hệ sinh thái (Redis+Celery+Flower+Bull) là đổi kiến trúc lớn, không phải tinh chỉnh — hoãn tới khi đo thấy chạm trần 1 máy |

**Điểm #2 (`_make_default_factory` load model riêng mỗi camera):** comment code hiện tại lý
giải "tracker state per-stream phải tách" — đúng, nhưng **nhầm với trọng số model** (stateless,
dùng chung được). Chỉ cần tách tracker/rule-engine state (như đang làm), còn model nên load 1
lần, truyền chung vào các worker.

## Ước tính số camera tối đa — không có 1 con số chắc chắn

| Cấu hình | Số camera | Độ tin cậy |
|---|---|---|
| Hiện tại, CPU 8 core | **3-4** | Đã đo thật |
| + Lớp 1, GPU tầm trung | Ước tính nội bộ **20-50** | ⚠️ Chưa từng đo, comment tự ghi "chưa cài đặt" |
| BiSwift, RTX3070 8GB | **9** stream | Đã published, đo thật — nhưng khác model/hardware |
| + Lớp 3 | Tuyến tính theo số node | Giới hạn chi phí, không giới hạn kiến trúc |

Lý do không tin thẳng số liệu literature: project dùng `yolo26l` (biến thể "large"), nặng hơn
model thường dùng trong các benchmark trên (nano/small, hoặc EdgeTPU chuyên dụng) — model càng
lớn, camera đồng thời xử lý được càng ít. **Cách đo đúng:** dùng
`get_status()["average_inference_capacity_fps"]` đã có sẵn, tăng dần số camera, chạy `yolo26l`
thật trên hardware sẽ deploy — không trả lời được bằng lý thuyết.

## Việc tiếp theo

- [ ] **Ưu tiên cao nhất, rẻ, không cần hạ tầng mới:** refactor để dùng chung 1 model instance
      giữa các camera worker (mục đối chiếu #2), giữ nguyên tracker/rule-engine tách riêng. Đo
      lại footprint bộ nhớ trước/sau.
- [ ] Quyết định thứ tự tiếp: Lớp 2 (mở rộng `MotionGate`, rẻ) trước hay Lớp 1 (batched
      detector, cần GPU) trước.
- [ ] Benchmark `yolo26l` thật bằng `get_status()`, tăng dần số camera, trên hardware sẽ deploy.
- [ ] Ghi số liệu đo thật vào docstring `stream_manager.py`, thay ước tính 20-50 chưa kiểm chứng.
- [ ] Lớp 3 (Celery/Redis kiểu DeepCamera) — hoãn tới khi đo thấy chạm trần 1 máy.

## Nguồn tham khảo

- https://arxiv.org/pdf/2312.15740 (BiSwift)
- https://www.researchgate.net/publication/257694201_Real-time_multi-camera_video_analytics_system_on_GPU
- https://onlinelibrary.wiley.com/doi/10.1002/cpe.70174 (DualRT)
- https://arxiv.org/pdf/2603.13176 (Perceive What Matters)
- https://arxiv.org/pdf/2005.09141 (ExSample)

**Giới hạn:** tổng hợp từ snippet tìm kiếm kiểm chứng chéo, không đọc trực tiếp toàn văn (một
số PDF paywall/không trích xuất được). Số liệu của chính project đọc trực tiếp từ source code,
độ tin cậy cao nhất trong tài liệu này.
