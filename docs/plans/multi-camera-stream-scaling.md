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

- [ ] Quyết định thứ tự triển khai: Lớp 2 (motion-aware, mở rộng `MotionGate` có sẵn) trước vì
      chi phí thấp nhất, không cần đổi hạ tầng — hay Lớp 1 (batched detector, cần GPU) trước.
- [ ] Nếu có GPU khả dụng để test: benchmark `yolo26l` batched trên số camera tăng dần, dùng
      `get_status()["average_inference_capacity_fps"]`.
- [ ] Nếu chưa có GPU: đo Lớp 2 mở rộng trên CPU hiện tại trước — chi phí triển khai thấp nhất
      trong 3 lớp, tận dụng được `MotionGate` đã có.
- [ ] Ghi lại số liệu đo thật vào docstring `stream_manager.py`, thay cho ước tính 20-50 chưa
      kiểm chứng hiện tại.

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
