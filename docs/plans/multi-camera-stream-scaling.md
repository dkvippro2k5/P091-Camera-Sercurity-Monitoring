# Nghiên cứu: tối ưu stream đa camera (capacity, batching, adaptive sampling)

## Context

`MultiStreamManager` (`src/vision/stream_manager.py`) chạy mỗi camera trên 1 worker thread
riêng, mỗi worker có `VisionPipeline` (detector + rule engine) độc lập. Số liệu **đã đo thật**,
ghi ngay trong docstring của class (dòng 82-100):

```
1 camera  ->  5.5 fps aggregate
2 cameras ->  9.1 fps aggregate (4.6 each)
4 cameras -> 11.9 fps aggregate (3.0 each)
```

Throughput tổng chững lại quanh 12-15 fps bất kể thêm bao nhiêu worker — per-camera fps giảm
tuyến tính sau đó. Giới hạn thực tế team tự ghi nhận (dòng 94-100):

- **CPU-only, class hiện tại:** 3-4 camera ở fps hữu dụng.
- **GPU tầm trung + batching:** 20-50 camera (**cần detector chạy batched, class này CHƯA cài
  đặt** — đây là ước tính, không phải số đã đo).
- **Hàng trăm camera:** cần horizontal scaling — N process, mỗi process phụ trách 1 nhóm
  camera, GPU-backed. Đây là đổi topology triển khai, không phải chỉnh tham số của class này.

`MotionGate` (`src/vision/motion.py`) đã có sẵn, tắt mặc định, A/B đo được (dòng 107-124):
cảnh bận không lợi gì (~6% nhiễu đo đạc), cảnh tĩnh (hành lang 3h sáng) nhanh hơn **~24x**
(136.89 fps, 96.7% frame bị skip) mà vẫn phát hiện đúng số sự kiện như cũ.

Tài liệu này research 3 hướng tối ưu, làm rõ chúng có xung đột nhau không, và ước tính số
camera khả thi theo từng cấu hình.

## 3 hướng — xếp lớp, không cạnh tranh nhau

```
Lớp 3 (horizontal scaling)   ->  N node, mỗi node chạy đầy đủ Lớp 1 + Lớp 2
Lớp 2 (adaptive motion gate) ->  giảm lượng frame cần đưa vào Lớp 1
Lớp 1 (shared batched detector) -> xử lý hiệu quả phần frame còn lại, trên 1 node
```

- **Lớp 1** quyết định 1 node xử lý được bao nhiêu camera hiệu quả nhất.
- **Lớp 2** không tăng công suất node, mà giảm tải đầu vào trước khi tới Lớp 1 — cộng dồn với
  Lớp 1, không thay thế.
- **Lớp 3** chỉ là chạy nhiều bản sao của (Lớp 1 + Lớp 2) khi 1 máy không đủ.

Đây là kiến trúc chuẩn của các hệ NVR/video-analytics lớn (DeepStream, Frigate) — cả 3 lớp tồn
tại đồng thời trong các hệ đó, không phải chọn 1 trong 3.

### Lớp 1: Shared batched detector (GPU batching)

`nvstreammux` (NVIDIA DeepStream) gộp frame từ nhiều camera stream thành 1 batch GPU duy nhất
trước khi inference, thay vì infer từng stream tuần tự — đây đúng là khoảng trống
`stream_manager.py` tự ghi nhận là chưa cài đặt ("needs a batched detector, which this class
does NOT implement"). Literature liên quan:

- **BiSwift**: đạt real-time detection trên **9 stream** đồng thời chỉ với 1 RTX3070 (8GB), dùng
  framework tối ưu 2 tầng (bandwidth + compute).
- **DeepQuery**: dynamic batch sizing, co-locate task real-time + delay-tolerant trên cùng GPU
  giới hạn tài nguyên.

### Lớp 2: Adaptive frame sampling / motion-aware scheduling

Không chạy full inference trên mọi frame của mọi stream — chỉ kích hoạt detection đầy đủ khi
có "information reward" đủ cao (có chuyển động). `MotionGate` đã implement đúng ý tưởng này ở
mức 1 stream; hướng mở rộng là áp dụng ở mức **toàn hệ thống đa luồng** (camera nào ít chuyển
động thì giảm tần suất infer, nhường compute cho camera đang có sự kiện).

- **"Perceive What Matters"** (Relevance-Driven Scheduling for Multimodal Streaming
  Perception) — kích hoạt module perception dựa trên information reward ước lượng.
- **ExSample** — adaptive sampling cho tìm kiếm hiệu quả trên kho video lớn.

### Lớp 3: Horizontal scaling

N process, mỗi process phụ trách 1 nhóm camera, GPU-backed — không đổi logic bên trong, chỉ
đổi topology triển khai. Không giới hạn về mặt kiến trúc, giới hạn bởi chi phí hạ tầng.

## Đối chiếu với kiến trúc DeepCamera (github.com/SharpAI/DeepCamera)

DeepCamera là 1 hệ multi-camera thực tế (Node.js/Shinobi capture + Celery/Redis worker pool
Python cho AI), dùng 5 nguyên tắc tối ưu. Đối chiếu từng điểm với code thật của project —
**3/5 đã có sẵn (khác công cụ, cùng ý tưởng), 1 điểm là lỗ hổng thật đáng sửa, 1 điểm là đổi cả
kiến trúc, chưa cấp thiết ở quy mô hiện tại.**

### #1 Tách "capture nhẹ" khỏi "AI compute nặng" — đã có sẵn

DeepCamera: FFmpeg process riêng + Celery worker riêng, qua Redis. Project: `CameraStreamWorker`
(`src/vision/stream_worker.py:87-98`) đã tách y hệt về ý tưởng bằng **2 thread** thay vì 2
process — `_reader_thread` (đọc RTSP) và `_process_thread` (chạy pipeline AI) riêng biệt, nhẹ
hơn DeepCamera (không cần IPC/Redis) nhưng cùng mục đích: I/O jitter RTSP không chặn tốc độ suy
luận. Không cần làm gì thêm.

### #2 Model dùng chung, load 1 lần — **lỗ hổng thật, đáng sửa nhất**

DeepCamera load model 1 lần cho cả pool worker phục vụ N camera (`concurrency=1` mỗi loại
task). Code hiện tại làm **ngược lại**: `_make_default_factory` (`stream_manager.py:183-221`)
tạo **1 `IncidentDetector` mới cho mỗi camera worker** — mỗi lần là 1 lần `YOLO(resolved)`
chạy lại, tức N camera = N lần load `yolo26l.pt` vào bộ nhớ, N bản sao model độc lập.

Comment trong code giải thích: *"YOLO's tracker state and the rule engine's dwell timers are
per-stream and would corrupt each other if shared."* — đúng cho tracker state (ByteTrack,
per-camera, phải tách) nhưng **nhầm lẫn với trọng số model** (stateless cho forward pass, dùng
chung được, không có lý do phải tách). Code hiện tại tách cả 2 cùng lúc dù chỉ 1 cái cần tách —
đúng anti-pattern mà nguyên tắc #2 của DeepCamera tránh.

→ **Việc đáng làm nhất trong đối chiếu này:** refactor `IncidentDetector` nhận 1 model instance
dùng chung (load 1 lần ở `MultiStreamManager`), còn tracker/rule-engine state vẫn tách riêng
mỗi camera như hiện tại. Giảm bộ nhớ tuyến tính theo N camera, **không cần đổi hạ tầng** (không
cần Celery/Redis).

### #3 Aggressive frame dropping/sampling — đã có sẵn, phần lớn khớp

- `_latest_frame` single-slot buffer (`stream_worker.py:67-69`) + `CAP_PROP_BUFFERSIZE=1`
  (dòng 166-169) = đúng giao thức "1 frame in-flight, luôn lấy frame mới nhất, drop frame cũ"
  mà DeepCamera dùng (và khớp `detection-protocol.md` của skill mới). Đã có.
- `target_fps` per-camera đã có sẵn trong `add_camera()` (`stream_manager.py:229`) — khớp
  "cấu hình FPS theo từng camera (0.2-15fps)".
- `MotionGate` tương đương `SAMPLING_TO_SAVE_ENERGY_MODE` + `motion_lock` — đã có, tắt mặc định.
- Các filter đặc thù khuôn mặt (`MINIMAL_FACE_RESOLUTION`, `BIGGEST_FACE_ONLY_MODE`,
  `ONE_KNOWN_PERSON_BYPASS_QUEUE_MODE`) **không áp dụng được** — DeepCamera là hệ face
  recognition, project này là phát hiện sự cố (fall/intrusion/abandoned), khác domain, không có
  khái niệm "known person" hay "face resolution" để lọc.

### #4 Backpressure/queue limit — đạt mục đích bằng cách đơn giản hơn

DeepCamera cần queue-length-polling qua Flower API + timeout Celery vì kiến trúc của họ là hàng
đợi phân tán thật (nhiều producer, worker pool qua Redis). Project hiện tại **không có hàng
đợi** giữa reader và processor — chỉ có single-slot buffer (mục #3) — nên tự động có đúng hiệu
ứng backpressure mong muốn (processor chậm thì reader ghi đè frame mới nhất, không tồn đọng) mà
không cần cơ chế phức tạp như Flower/timeout/queue-depth polling. Đây là ưu điểm của kiến trúc
đơn giản hơn (in-process, chưa phân tán), không phải thiếu sót.

### #5 Celery/Redis worker pool, cluster — đúng là Lớp 3, nhưng chi phí lớn

Đây chính là "Lớp 3: Horizontal scaling" ở trên — cùng nguyên tắc (N node/worker chia sẻ camera
qua broker chung), nhưng DeepCamera hiện thực bằng cả hệ sinh thái (Node.js + Python + Redis +
Celery + Flower + Bull) — không phải 1 tinh chỉnh nhỏ, mà đổi hẳn từ "1 process, N thread"
(hiện tại) sang hệ phân tán nhiều process/worker qua message broker. Chi phí triển khai (thêm
Redis, thêm Celery, deploy phức tạp hơn, thêm giám sát Flower) lớn hơn hẳn lợi ích ở quy mô
hiện tại (vài camera, dự án hackathon/demo). Chỉ đáng làm khi đã chạm trần capacity của 1 máy
(đo được qua `get_status()`, xem mục dưới) và cần scale ra nhiều máy thật.

### Kết luận đối chiếu

Fit về **nguyên tắc**, không fit về **công cụ cụ thể**. Cả 5 nguyên tắc DeepCamera dùng đều hợp
lý và là best-practice chuẩn cho multi-camera — nhưng project này (FastAPI + thread, không phải
Node.js + Celery/Redis) đã tự implement độc lập 3/5 điểm (#1, #3, #4) theo cách nhẹ hơn, phù
hợp quy mô nhỏ hơn. #2 (shared model) là khoảng trống thật, đáng sửa ngay vì rẻ và không cần
đổi hạ tầng. #5 là đúng hướng dài hạn nhưng nên hoãn tới khi đo thấy cần scale ra nhiều máy.

## Ước tính số camera tối đa — có điều kiện, không phải 1 con số chắc chắn

| Cấu hình | Số camera | Độ tin cậy |
|---|---|---|
| Hiện tại, chưa sửa gì (CPU 8 core) | **3-4 camera** | Đã đo thật, có số liệu trong code |
| + Lớp 1, vẫn CPU | Chưa có số đo | Batching trên CPU lợi ích khiêm tốn hơn GPU (chủ yếu giảm tranh chấp thread + tránh load N model trùng lặp) — không có cơ sở đoán con số cụ thể, cần benchmark sau khi code |
| + Lớp 1, GPU tầm trung | Team tự ước tính **20-50 camera** (comment `stream_manager.py:95`) | ⚠️ Chưa từng đo thật — chính comment ghi rõ "needs a batched detector, which this class does NOT implement" |
| Số liệu published thật (BiSwift, RTX3070 8GB) | **9 stream real-time** | Đã published, đo thật — nhưng khác hardware, khác model so với setup hiện tại |
| + Lớp 3 | Tuyến tính theo số node | Không giới hạn kiến trúc, giới hạn chi phí/hạ tầng |

**Vì sao không đưa 1 con số chắc chắn:** 2 lý do khiến số liệu tham khảo không áp thẳng được:

1. **Kích thước model khác biệt lớn** — project đang dùng `yolo26l` (biến thể "large"), nặng
   hơn đáng kể so với model thường dùng trong các benchmark đa camera (BiSwift và đa số hệ NVR
   thực tế dùng biến thể nano/small, hoặc phần cứng chuyên dụng như Coral EdgeTPU của Frigate).
   Model càng lớn, số camera đồng thời xử lý được càng thấp — số 9 (RTX3070) hay 20-50 (ước
   tính chung) nhiều khả năng là cận trên lạc quan cho `yolo26l`.
2. **Chưa có bản đo nào sau khi sửa kiến trúc** — mọi con số "sau khi sửa" ở trên là suy luận
   từ literature, chưa chạy thật trên code + hardware của team.

## Cách đo đúng, thay vì đoán lý thuyết

`get_status()["average_inference_capacity_fps"]` đã có sẵn trong `MultiStreamManager` — dùng
đúng phương pháp A/B team đã áp dụng cho Motion Gate (dòng 107-124 trong docstring): tăng dần
số camera, chạy `yolo26l` thật trên đúng hardware sẽ deploy, đo tới khi fps/camera tụt dưới mức
chấp nhận được. Đây không phải câu hỏi trả lời được bằng lý thuyết — phải đo thực nghiệm.

## Việc tiếp theo

- [ ] **Ưu tiên cao nhất, rẻ, không cần hạ tầng mới:** refactor `IncidentDetector`/
      `_make_default_factory` để dùng chung 1 model instance (load 1 lần) giữa các camera
      worker, tách riêng phần tracker state + rule-engine dwell timers như hiện tại (mục
      "Đối chiếu với DeepCamera" #2). Đo lại footprint bộ nhớ trước/sau với N camera.
- [ ] Quyết định thứ tự triển khai tiếp: Lớp 2 (motion-aware, mở rộng `MotionGate` có sẵn)
      trước vì chi phí thấp nhất — hay Lớp 1 (batched detector, cần GPU) trước.
- [ ] Nếu có GPU khả dụng để test: benchmark `yolo26l` batched trên số camera tăng dần, dùng
      `get_status()["average_inference_capacity_fps"]`.
- [ ] Nếu chưa có GPU: đo Lớp 2 mở rộng trên CPU hiện tại trước — chi phí triển khai thấp nhất
      trong 3 lớp, tận dụng được `MotionGate` đã có.
- [ ] Ghi lại số liệu đo thật vào docstring `stream_manager.py`, thay cho ước tính 20-50 chưa
      kiểm chứng hiện tại.
- [ ] Lớp 3 (Celery/Redis worker pool kiểu DeepCamera) — hoãn tới khi đo thấy chạm trần
      capacity 1 máy, không triển khai trước.

## Nguồn tham khảo

- https://arxiv.org/pdf/2312.15740 (BiSwift: Bandwidth Orchestrator for Multi-Stream Video
  Analytics on Edge)
- https://www.researchgate.net/publication/257694201_Real-time_multi-camera_video_analytics_system_on_GPU
  (batching kiểu DeepStream)
- https://onlinelibrary.wiley.com/doi/10.1002/cpe.70174 (DualRT: QoS-Aware Real-Time Video
  Analytics on Dual-Stage GPU-CPU Edge)
- https://arxiv.org/pdf/2603.13176 (Perceive What Matters: Relevance-Driven Scheduling for
  Multimodal Streaming Perception)
- https://arxiv.org/pdf/2005.09141 (ExSample: Efficient Searches on Video Repositories
  through Adaptive Sampling)

**Giới hạn cần biết:** thông tin trên tổng hợp từ snippet tìm kiếm đã kiểm chứng chéo nhiều
nguồn, không phải đọc trực tiếp toàn văn các bài báo (một số bị paywall/không trích xuất được
PDF trong môi trường này). Phần số liệu của chính project (`stream_manager.py`) đọc trực tiếp
từ source code, độ tin cậy cao nhất trong tài liệu này.
