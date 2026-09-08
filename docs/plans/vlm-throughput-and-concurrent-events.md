# Kế hoạch: tăng thông lượng triage và xử lý nhiều sự kiện đồng thời

Trạng thái: **ĐÃ THỰC THI — giai đoạn 1, 2, 3 và phần dọn dẹp giai đoạn 5**
Nhánh hiện tại: `fix/vlm-triage-robustness`
Ngày: 2026-09-08

## Kết quả thực thi

| Giai đoạn | Trạng thái | Commit |
| --- | --- | --- |
| 1 — Sửa `.env` (pre/post-roll, max_frames) | ✅ Xong | không commit code — `.env` bị gitignore, áp dụng runtime-only |
| 2 — Tách post-roll wait khỏi triage pool | ✅ Xong | `fix(vlm): stop the post-roll wait from occupying a triage worker` |
| 3 — Crop theo `track_id` | ✅ Xong | `feat(vlm): crop the triage clip to the alerting track` |
| Dọn dẹp GĐ5 — default weights `yolo26n.pt` không tồn tại | ✅ Xong | `fix(vision): stop defaulting to a YOLO checkpoint that is not in the repo` |
| Dọn dẹp GĐ5 — `DETECTOR_IMGSZ` biến chết | ✅ Xong | comment cảnh báo trong `.env`, chưa nối dây thật |
| 4 — Gộp alert đồng thời | Hoãn theo plan | — |
| 5 — Giảm tải YOLO (MotionGate, imgsz thật) | Chưa làm, cần đo trước | — |

Toàn bộ 271 test xanh (15 skip do thiếu footage/GPU) sau mỗi giai đoạn. 20 test mới được
thêm: 5 cho giai đoạn 2 (post-roll timer), 6 cho crop trong `_collect_clip`, 2 cho
`VisionPipeline.last_tracks`, cộng các test đã sửa để dùng `RingFrame` thay vì tuple thô.

**Phát hiện phụ ngoài dự kiến:** khi chạy full suite sau giai đoạn 2, phát hiện
`IncidentDetector.DEFAULT_WEIGHTS_PATH` (hằng số riêng trong `detector.py`, độc lập với
`config.py`) cũng trỏ vào `yolo26n.pt` không tồn tại — đây là default thực sự mà một
`VisionPipeline()`/`IncidentDetector()` không tham số dùng, và là lý do 5 test trong
`test_vision.py` fail ngay cả sau khi `config.py` đã được sửa. Đã sửa cả hai hằng số về
`yolo26s.pt` (file thực sự có trong repo).

---

## 0. Tóm tắt chẩn đoán

Hai triệu chứng người dùng báo cáo có **nguyên nhân khác nhau**, và cả hai đều không nằm ở
"VLM chạy chậm trên frame path" — VLM không hề nằm trên frame path.

| Triệu chứng báo cáo | Nguyên nhân thực tế | Vị trí |
| --- | --- | --- |
| "VLM drop <1fps, bỏ qua frame" | YOLO pipeline chậm; VLM chỉ chạy 1 lần/alert, không theo frame | `stream_worker._process_loop` |
| "Không triage được" | Clip lấy 4 frame trải trên 18s → khoảnh khắc ngã lọt giữa hai mẫu | `.env` + `_collect_clip` |
| "Khó detect >2 sự kiện đồng thời" | `_collect_clip(camera_id)` không nhận `track_id`; N alert cùng camera nhận cùng một bộ frame toàn cảnh | `monitoring._collect_clip` |
| Hàng đợi bão hòa | `_wait_for_post_roll` dùng `time.sleep()` chiếm worker của `_triage_pool` | `monitoring._wait_for_post_roll` |

### 0.1 Bằng chứng số học

Cấu hình `.env` hiện tại (`VLM_PRE_ROLL_SEC=12`, `VLM_POST_ROLL_SEC=6`, `VLM_MAX_FRAMES=4`):

```
Cửa sổ clip   = 12 + 6                       = 18 giây
Số frame mẫu  = 4
Khoảng cách   = 18 / 4                       ≈ 4.5 giây/frame
Ring size     = max(16, 4.0*(12+6+2)*1.5)    = 120 slot/camera
Ring bộ nhớ   = 120 JPEG @1280px × 4 camera  = 480 JPEG thường trú RAM
```

Một cú ngã kéo dài ~1 giây. Với khoảng cách mẫu 4.5 giây, xác suất chuỗi 4 frame chứa được
khoảnh khắc chuyển tiếp là thấp — model nhận 4 ảnh "đứng", "đứng", "nằm", "nằm" cách nhau
quá xa, hoặc tệ hơn là 4 ảnh đều "nằm". Đây là lý do trực tiếp của "không triage được".

Comment trong `src/config.py:51` đã nêu đúng yêu cầu: pre-roll chỉ cần phủ
`RuleEngine.fall_dwell_sec (3.0) + cửa sổ vận tốc (0.5)` = 3.5 giây. Giá trị 12 giây trong
`.env` không mang lại thông tin gì thêm mà chỉ pha loãng mật độ mẫu.

### 0.2 Điều KHÔNG phải nguyên nhân (đã kiểm chứng, không cần sửa)

- **Rule engine không chặn sự kiện đồng thời.** Cooldown key là
  `(camera_id, label, track_id, zone_name)` tại `rule_engine.py:231`, nên 3 người ngã cùng
  lúc vẫn sinh ra 3 alert riêng biệt. Không cần đụng vào rule engine.
- **Ring không bị YOLO bỏ đói.** Commit `6f38a30` đã chuyển ring feed sang display loop
  (`_on_raw_frame`, chạy ở `display_fps` chứ không phải `target_fps`). Cơ chế này đúng.
- **Comment inline trong `.env` không gây lỗi.** Đã xác minh bằng cách load `Settings`:
  pydantic-settings strip phần comment, `camera_max_concurrent_inferences` parse ra `2`
  đúng như ý định. Không phải bug.

---

## 1. Phạm vi

### Trong phạm vi
1. Chỉnh cấu hình `.env` để khôi phục mật độ mẫu clip.
2. Tách post-roll wait ra khỏi `_triage_pool`.
3. Crop theo `track_id` để VLM biết đang được hỏi về đối tượng nào.
4. Gộp các alert đồng thời trên cùng camera thành một lần gọi API.
5. Giảm tải YOLO nếu đo được vẫn <1fps.

### Ngoài phạm vi
- Không đổi `RuleEngine` (đã xác minh không phải nguyên nhân).
- Không đổi giao thức SSE/REST hay schema `Alert` theo hướng phá vỡ tương thích.
- Không đụng vào cơ chế suppression (`vlm_auto_suppress_confidence`) — commit `9460825`
  vừa hiệu chỉnh xong.
- Không thêm face anonymization (việc riêng, PRD §3.3).

---

## 2. Các giai đoạn

Bốn giai đoạn độc lập, có thể dừng sau bất kỳ giai đoạn nào. Giai đoạn 1 và 2 rủi ro thấp
và kiểm chứng được ngay; giai đoạn 3 là thay đổi kiến trúc thực sự.

---

### Giai đoạn 1 — Sửa cấu hình (0 dòng code)

**Vấn đề:** mật độ mẫu clip quá thưa để thấy chuyển tiếp ngã.

**Thay đổi** — chỉ file `.env`:

```diff
-VLM_PRE_ROLL_SEC=12                   # nới cửa sổ cho chắc
-VLM_POST_ROLL_SEC=6
-VLM_MAX_FRAMES=4
+VLM_PRE_ROLL_SEC=5                    # phủ fall_dwell (3.0) + velocity window (0.5) + biên
+VLM_POST_ROLL_SEC=2
+VLM_MAX_FRAMES=6
```

**Kết quả tính toán:**

```
Cửa sổ clip   = 5 + 2        = 7 giây      (từ 18s)
Khoảng cách   = 7 / 6        ≈ 1.2 giây    (từ 4.5s — dày hơn 3.8 lần)
Ring size     = max(16, 4.0*(5+2+2)*1.5) = 54 slot  (từ 120 — giảm 55% RAM)
Độ trễ verdict giảm 4 giây/alert (post-roll 6s → 2s)
```

**Rủi ro:** `VLM_MAX_FRAMES` từ 4 lên 6 làm tăng 50% image token mỗi lần gọi. Với
`qwen-vl-plus` và `detail: "high"` đây là chi phí thật. Nếu latency vượt
`VLM_TIMEOUT_SEC=40`, hạ về 5. Giai đoạn 3 sẽ bù lại nhiều hơn phần tăng này nhờ crop.

**Kiểm chứng:**
- `test_pre_roll_reaches_past_the_fall_dwell` yêu cầu `_pre_roll_sec >= 3.5` — giá trị 5.0 thoả.
- `test_ring_capacity_covers_the_whole_clip_window` tự suy từ settings — thoả.
- Chạy 1 video ngã, xác nhận `triage_verdict` ra `real` thay vì `uncertain`.

---

### Giai đoạn 2 — Tách post-roll wait khỏi triage pool

**Vấn đề:** `_wait_for_post_roll` (`monitoring.py:838`) gọi `time.sleep(self._post_roll_sec)`
**bên trong** một worker của `_triage_pool`. Với 4 worker và post-roll 2 giây, mỗi alert
chiếm một worker 2 giây chỉ để ngủ, trước khi bắt đầu công việc thật (gọi API 10-30s).

Khi 3 alert nổ đồng thời: 3/4 worker đang ngủ, throughput sụt, `_MAX_TRIAGE_IN_FLIGHT=8`
chạm trần, và alert thứ 9 nhận `"triage queue saturated"` (`monitoring.py:784`).

**Thay đổi** trong `src/services/monitoring.py`:

1. Thêm một `threading.Timer` (hoặc một scheduler thread nhỏ dùng `heapq` + `Condition`)
   làm tầng chờ. `_submit_triage` lên lịch thay vì submit ngay:

```python
def _submit_triage(self, alert: Alert) -> None:
    """Lên lịch triage sau post-roll, KHÔNG chiếm worker để ngủ."""
    if self._post_roll_sec <= 0:
        self._enqueue_triage(alert)
        return
    timer = threading.Timer(self._post_roll_sec, self._enqueue_triage, args=(alert,))
    timer.daemon = True
    self._timers.add(timer)      # để shutdown() huỷ được
    timer.start()
```

2. `_enqueue_triage` là phần submit vào pool như cũ (giữ nguyên xử lý `RuntimeError`).

3. `wait_for_post_roll` trong workflow trở thành no-op (giữ node để không đổi hình dạng
   graph, hoặc bỏ node — xem mục cân nhắc bên dưới).

4. `shutdown()` huỷ mọi timer đang chờ trước khi `_triage_pool.shutdown(wait=False)`.

**Cân nhắc thiết kế — có nên bỏ node `wait_for_post_roll` khỏi graph?**
Đề xuất **giữ node nhưng biến thành no-op có chú thích**. Lý do: `test_graph.py` và cấu
trúc workflow đang phản ánh 1:1 các bước nghiệp vụ; bỏ node làm graph khác đi mà không
mang lại lợi ích runtime nào. Chú thích sẽ nói rõ việc chờ đã chuyển ra ngoài graph.

**Kết quả:** 4 worker luôn dành cho việc gọi API. Throughput triage tăng ~2-3x ở tải
đồng thời mà không thêm thread nào.

**Rủi ro:** một `threading.Timer` là một thread. Với `alert_cooldown_sec=30` và
`_MAX_TRIAGE_IN_FLIGHT=8`, số timer sống đồng thời bị chặn trên bởi số alert trong
`post_roll_sec` giây — thực tế dưới 10. Chấp nhận được. Nếu muốn chặt hơn, dùng một
scheduler thread duy nhất thay vì `Timer` per-alert (phức tạp hơn, để dành nếu đo thấy cần).

**Kiểm chứng:**
- Test mới: `_submit_triage` trả về ngay lập tức (< 50ms) dù `_post_roll_sec = 2.0`.
- Test mới: verdict vẫn tới sau khi post-roll trôi qua.
- Test mới: `shutdown()` khi còn timer chờ không để lại thread treo.
- `test_triage_runs_off_the_caller_thread` phải vẫn xanh.

---

### Giai đoạn 3 — Crop theo `track_id` *(thay đổi kiến trúc chính)*

**Vấn đề gốc của "khó detect >2 sự kiện đồng thời".** `_collect_clip(camera_id)`
(`monitoring.py:970`) chỉ nhận `camera_id`. Ba alert đồng thời trên cùng camera nhận **cùng
một bộ frame toàn cảnh**. VLM được hỏi "sự kiện này có thật không?" mà không có cách nào
biết đang được hỏi về *ai*. Với 3 người trong khung hình, câu hỏi không có câu trả lời xác
định — và ta trả tiền 3 lần cho cùng một bộ ảnh.

`Alert` đã mang `track_id` (`camera_schemas.py:186`, gán tại `monitoring.py:721`), chỉ là
nó chưa bao giờ được dùng ở tầng thu thập bằng chứng.

**Thay đổi:**

1. **Ring lưu kèm bbox.** Đổi kiểu ring từ `deque[tuple[float, bytes]]` sang
   `deque[RingFrame]` với `RingFrame = (timestamp, jpeg, tracks)`, trong đó `tracks` là
   `dict[str, BoundingBox]` — ánh xạ `track_id` → bbox tại thời điểm encode.

   Nguồn dữ liệu: `VisionPipeline._last_detections` (`pipeline.py:59`), đúng cái mà
   `overlay_last_known` đang dùng. Cần thêm một accessor read-only trên pipeline
   (`last_tracks(camera_id)`) trả về bản sao — **không** trả tham chiếu trực tiếp, vì
   display loop và process loop chạy trên hai thread khác nhau.

   Toạ độ bbox lưu ở **dạng chuẩn hoá [0,1]**, không phải pixel. Lý do: `_on_raw_frame`
   downscale frame xuống `_RING_MAX_EDGE=1280` *sau* khi lấy bbox, và
   `camera_capture_max_height` cũng có thể đổi giữa các lần khởi động. Toạ độ pixel sẽ sai
   lệch; chuẩn hoá thì không.

2. **`_collect_clip` nhận thêm `track_id`:**

```python
def _collect_clip(
    self, camera_id: str, track_id: str | None = None
) -> tuple[list[bytes], int]:
```

   Khi `track_id` không phải `None` và có mặt trong `tracks` của frame, crop vùng quanh
   bbox đó với padding **40%** mỗi chiều (đủ để thấy sàn nhà và ngữ cảnh xung quanh — một
   cú ngã cần thấy mặt sàn, crop sát người sẽ mất thông tin đó), rồi re-encode JPEG.

3. **Degrade an toàn.** Nếu `track_id` là `None`, hoặc không tìm thấy trong frame, hoặc
   bbox quá nhỏ (< 64px cạnh ngắn sau crop), **giữ nguyên frame toàn cảnh**. Hành vi hiện
   tại là fallback, không phải lỗi. Ghi log ở mức debug, không phải warning — track biến
   mất giữa các frame là chuyện bình thường của tracking.

4. **Chú thích trong prompt.** Khi clip đã crop, thêm một dòng vào
   `OpenAIVLMTriage._build_prompt` nói rõ khung hình đã được cắt quanh đối tượng được hỏi.
   Không có dòng này, model có thể suy diễn sai về việc "không thấy ai khác trong khung".
   Thêm một trường `TriageRequest.cropped: bool = False`.

**Lợi ích kép:**
- VLM biết đang được hỏi về ai → xử lý được N sự kiện đồng thời một cách có nghĩa.
- Ảnh crop nhỏ hơn nhiều → giảm image token, giảm latency, giảm chi phí. Đây là cái bù
  lại phần tăng `VLM_MAX_FRAMES` ở giai đoạn 1, và mở đường cho việc tăng tiếp nếu muốn.

**Rủi ro:**
- Ring nặng thêm phần `tracks` dict. Nhỏ (vài chục byte/frame), không đáng kể so với JPEG.
- Crop rồi re-encode tốn CPU **trên triage thread**, không phải display loop — đúng chỗ,
  vì triage thread vốn đang chờ I/O mạng.
- Nếu tracking đổi `track_id` giữa pre-roll và post-roll (stitching, `rule_engine.py:505`),
  crop sẽ mất dấu ở nửa sau. Xử lý: dùng cả `track_id` và `aliased_from` khi tra cứu.
  Cần kiểm tra `_stitch_or_create` xem `aliased_from` có tới được `_last_detections` không;
  nếu không, ghi nhận là hạn chế đã biết và fallback về toàn cảnh (vẫn đúng, chỉ kém tối ưu).

**Kiểm chứng:**
- Test: clip có `track_id` hợp lệ → frame trả về khác với frame toàn cảnh (đã crop).
- Test: `track_id=None` → hành vi y hệt hiện tại (bảo vệ tương thích ngược).
- Test: `track_id` không có trong ring → fallback toàn cảnh, không raise.
- Test: bbox chuẩn hoá được scale đúng sau khi frame bị downscale.
- Test: `cropped=True` làm prompt chứa dòng chú thích.
- Toàn bộ test hiện có trong `test_vlm_triage.py` phải xanh — đặc biệt
  `test_clip_splits_into_pre_and_post_roll` và `test_a_one_sided_clip_is_still_assessed`,
  hai test này dựng ring thủ công nên **sẽ phải cập nhật theo kiểu ring mới**.

**Ghi chú di trú:** đổi kiểu ring làm hỏng 5 test đang dựng
`deque([(stamp, b"bytes")])` trực tiếp (dòng 260, 307, 326 trong `test_vlm_triage.py`).
Cần một helper trong `conftest.py` để dựng `RingFrame` cho gọn, thay vì sửa rải rác.

---

### Giai đoạn 4 — Gộp alert đồng thời thành một lần gọi

**Chỉ làm sau khi giai đoạn 3 xong và đo được lợi ích.**

Khi N alert trên cùng camera nổ trong cửa sổ < 2 giây, gom thành **một** request:
1 bộ frame nền + N crop, yêu cầu model trả về N verdict trong một JSON array.

**Lợi ích:** N lần gọi → 1. Model thấy được tương tác giữa các đối tượng (ví dụ: một người
cúi xuống đỡ người ngã — nhìn riêng lẻ thì cả hai đều mơ hồ, nhìn chung thì rõ).

**Chi phí:** `_RESPONSE_SCHEMA` phải đổi sang mảng verdict; logic khớp verdict với alert
theo chỉ số; xử lý trường hợp model trả sai số lượng (đã có tiền lệ xử lý ở
`vlm.py:frame_observations`, dùng lại nguyên tắc "thà bỏ hết còn hơn khớp sai").

**Đề xuất:** hoãn. Giai đoạn 3 đã giải quyết vấn đề đúng đắn của câu hỏi ban đầu; giai
đoạn 4 là tối ưu chi phí. Đánh giá lại sau khi có số đo thật.

---

### Giai đoạn 5 — Giảm tải YOLO *(có điều kiện)*

**Chỉ làm nếu đo được YOLO vẫn <1fps sau các giai đoạn trên.** Cần đo trước, không đoán.

Đo bằng `GET /api/v1/monitoring/status` hoặc `worker.get_stats()`, đọc
`inference_capacity_fps` và `latency_p95_ms` (`stream_worker.py:238`).

Nếu xác nhận chậm, theo thứ tự tác động giảm dần:

1. **`MODEL_WEIGHTS_PATH=src/vision/weights/yolo26n.pt`** — `.env` đang dùng `yolo26s.pt`
   trong khi mặc định của repo là `yolo26n.pt` (`config.py:96`). Bản `n` nhanh hơn ~2.5x
   trên CPU.

   ⚠️ **ĐÃ XÁC MINH: `src/vision/weights/` chỉ chứa `yolo26s.pt`.** File `yolo26n.pt`
   **không tồn tại** — nghĩa là mặc định của `config.py` hiện đang trỏ vào một file không
   có, và `.env` đang là thứ duy nhất giữ cho hệ thống chạy được. Muốn dùng bản `n` thì
   phải tải weights về trước. Đây cũng là một phát hiện riêng: một clone mới không có
   `.env` sẽ chết ngay khi khởi động camera.

2. **Bật `MotionGate`** — hiện `enabled=False` mặc định (`pipeline.py:55`). Cảnh tĩnh sẽ
   skip inference hoàn toàn. Cần thêm một setting để bật, hiện chưa có đường nối từ
   `Settings` xuống `VisionPipeline`.

3. **`DETECTOR_IMGSZ` là biến chết.**

   ⚠️ **ĐÃ XÁC MINH: `grep -rn "imgsz" src/` không trả về kết quả nào.** `.env` đang đặt
   `DETECTOR_IMGSZ=640` nhưng không có gì trong code đọc nó, và `Settings` khai báo
   `extra="ignore"` (`config.py:12`) nên nó bị nuốt im lặng, không cảnh báo.

   Đường thật để giảm kích thước inference là `VisionPipeline.inference_size`
   (`pipeline.py:46`) — hiện **không** có đường nối từ `Settings` xuống. Muốn dùng thì
   phải nối dây, không phải chỉ đặt biến môi trường.

   Đề xuất: xoá `DETECTOR_IMGSZ` khỏi `.env`/`.env.example` (hoặc nối dây thật cho nó),
   để không ai khác tưởng nó đang có tác dụng.

---

## 3. Thứ tự thực thi đề xuất

```
Giai đoạn 1 (config)  ──▶ đo lại  ──▶ Giai đoạn 2 (tách wait)  ──▶ đo lại
                                              │
                                              ▼
                                     Giai đoạn 3 (crop track_id)  ──▶ đo lại
                                              │
                                              ▼
                                   [Giai đoạn 4 và 5: có điều kiện]
```

Giai đoạn 1 và 2 nên đi chung một commit? **Không** — tách riêng. Giai đoạn 1 là thay đổi
cấu hình có thể hoàn tác tức thì và cần được đo riêng để biết nó đóng góp bao nhiêu.

---

## 4. Kế hoạch commit

| # | Nhánh/commit | Nội dung |
| --- | --- | --- |
| 1 | `fix(vlm): restore clip sample density; 18s/4 frames missed the transition` | `.env` + cập nhật `.env.example` với chú thích giải thích tại sao pre-roll dài lại phản tác dụng |
| 2 | `fix(vlm): stop post-roll wait from occupying a triage worker` | `monitoring.py`, `incident_workflow.py`, test mới |
| 3 | `feat(vlm): crop the clip to the alerting track so N concurrent events are separable` | `monitoring.py`, `pipeline.py`, `vlm.py`, `conftest.py`, test |

Mỗi commit chạy `pytest` xanh trước khi commit.

---

## 5. Rủi ro tổng thể và cách rút lui

| Rủi ro | Khả năng | Rút lui |
| --- | --- | --- |
| `VLM_MAX_FRAMES=6` vượt timeout 40s | Trung bình | Hạ về 5 hoặc 4; hoặc chờ giai đoạn 3 giảm token nhờ crop |
| Timer thread rò rỉ khi shutdown | Thấp | Test riêng cho shutdown; huỷ timer trước khi shutdown pool |
| Crop làm mất ngữ cảnh sàn nhà → verdict tệ đi cho fall | Trung bình | Padding 40%; nếu vẫn tệ, tăng padding hoặc gửi kèm 1 frame toàn cảnh |
| Đổi kiểu ring làm hỏng test | Cao (đã biết) | Helper trong `conftest.py`; 5 vị trí đã xác định sẵn |
| `track_id` đổi do stitching giữa pre/post-roll | Trung bình | Tra cứu cả `aliased_from`; fallback toàn cảnh |

**Điểm rút lui an toàn:** sau mỗi giai đoạn. Không giai đoạn nào phụ thuộc vào giai đoạn
sau nó.

---

## 6. Cách đo (chung cho mọi giai đoạn)

Trước và sau mỗi giai đoạn, chạy cùng một bộ video và ghi lại:

1. **Tỷ lệ triage thành công** — bao nhiêu % alert có `triage_status == "complete"` và
   `triage_error is None`. Đây là chỉ số chính cho "không triage được".
2. **Phân bố verdict** — bao nhiêu `real` / `false_positive` / `uncertain`. Tỷ lệ
   `uncertain` cao là dấu hiệu clip không đủ thông tin.
3. **`latency_ms` của VLM** — từ `TriageVerdict.latency_ms`.
4. **`inference_capacity_fps` và `latency_p95_ms`** — từ `worker.get_stats()`, cho YOLO.
5. **Số lần `"triage queue saturated"`** trong log — chỉ số cho giai đoạn 2.

Đề xuất: chạy với **2 camera cùng lúc, mỗi camera 1 video có nhiều người**, để tái hiện
đúng kịch bản "nhiều sự kiện đồng thời" mà báo cáo nêu.

---

## 7. Câu hỏi cần bạn quyết trước khi thực thi

1. **`VLM_MAX_FRAMES`: 6 hay giữ 4?** Đề xuất 6 vì mật độ mẫu là vấn đề cốt lõi, nhưng nó
   tăng chi phí API 50% cho tới khi giai đoạn 3 bù lại.
   - giữ 6
2. **Giai đoạn 3 có làm ngay không?** Đây là thay đổi lớn nhất và là fix gốc cho câu hỏi
   ban đầu, nhưng cũng là cái đụng nhiều file nhất và phải sửa test hiện có.
   - thực hiện giai đoạn 3 
3. **Giai đoạn 5 có nằm trong đợt này không?** Nó cần đo trước. Hai phát hiện phụ đã xác
   minh được ở mục 5 (`DETECTOR_IMGSZ` là biến chết; `yolo26n.pt` không tồn tại nên mặc
   định của `config.py` đang hỏng) đáng sửa riêng bất kể có làm giai đoạn 5 hay không —
   bạn muốn gộp vào đợt này hay tách ra?
   - gộp vào đợt này
