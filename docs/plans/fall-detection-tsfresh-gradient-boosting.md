# Nghiên cứu: tsfresh + Gradient Boosting cho fall detection (bổ sung case "ngã dọc")

## Context

`RuleEngine._fall_indicators()` (`src/vision/rule_engine.py:477-531`) hiện dùng 3 chỉ báo
ngưỡng cố định do người tự chọn: velocity (`V_y ≥ 1.2H/s`), aspect ratio (`AR ≥ 1.2`),
height-drop (`ΔH ≥ 50%`), cần 2/3 để tính là nghi ngờ ngã. Có một **gate chặn cứng** trước khi
tính điểm (dòng 527-528):

```python
if ar < self.fall_upright_aspect_ratio:  # 0.7
    return 0, f"{_UPRIGHT_PREFIX} (AR {ar:.2f}) [{breakdown}]"
```

Nếu AR < 0.7 (bbox vẫn cao/hẹp), hàm trả về **0/3 ngay lập tức**, bất kể `V_y`/`ΔH` mạnh cỡ
nào. Đây chính là root cause khiến hệ thống **bỏ sót ngã dọc** (ngã thẳng về phía/ra xa camera,
dọc trục quang học) — trường hợp bbox không bao giờ "bẹt ra" nên AR luôn thấp, gate luôn chặn.

Tài liệu này nghiên cứu 1 trong 2 hướng khắc phục đã đề xuất: thay ngưỡng cố định bằng
classifier học từ dữ liệu (**tsfresh + Gradient Boosting**), làm rõ cơ chế, input/output, và
cách áp dụng cho nhiều người trong cùng khung hình.

Hướng còn lại (multi-feature + fall angle qua `yolo26l-seg.pt`) không nằm trong tài liệu này.

## Nguồn tham khảo chính

Vision-Based Fall Detector for Elderly Based on Sliding Window Approach and Feature
Engineering (Journal of Data Science and Intelligent Systems) — dùng `HeightWidthRatio` +
`FrameSpeed` làm 2 tham số thô, tsfresh sinh feature phụ, `auto-sklearn` chọn model, Gradient
Boosting đạt 96% accuracy trên tập gộp UR Fall Detection + CAUCAFall.

- https://ojs.bonviewpress.com/index.php/jdsis/article/view/4100
- https://tsfresh.readthedocs.io/en/stable/text/forecasting.html (cơ chế sliding window)
- https://www.sciencedirect.com/science/article/pii/S0925231218304843 (thuật toán FRESH gốc)
- https://fenix.ur.edu.pl/~mkepski/ds/uf.html (UR Fall Detection dataset)
- https://data.mendeley.com/datasets/7w7fccy7ky/4 (CAUCAFall dataset)

**Giới hạn cần biết trước:** cả hai PDF gốc không trích xuất được full-text (môi trường thiếu
`poppler`/thư viện đọc PDF) — nội dung dưới tổng hợp từ snippet tìm kiếm đã kiểm chứng chéo,
không phải đọc trực tiếp toàn văn. Phần cơ chế `tsfresh` (sliding window, FRESH algorithm) đọc
trực tiếp từ tài liệu chính thức của thư viện, độ tin cậy cao.

## Cơ chế hoạt động — 3 bước

### Bước 1: Sliding window trên chuỗi thời gian thô

Không đưa cả video vào một lần. Cắt thành cửa sổ trượt (N frame gần nhất, trượt 1 frame mỗi
lần suy luận). `tsfresh` có 3 tham số điều khiển cửa sổ: `max_timeshift` (độ dài cửa sổ tối
đa), `min_timeshift` (bỏ qua chuỗi ngắn hơn ngưỡng này — thường rơi vào đầu video), và
`rolling_direction`.

### Bước 2: Sinh ~800 feature tự động, rồi tự lọc bằng kiểm định thống kê

Đây là điểm khác biệt cốt lõi so với 3-chỉ-báo hiện tại: không cần tự nghĩ công thức. tsfresh
tự tính ~800 phép biến đổi thống kê chuẩn trên mỗi chuỗi đầu vào — mean, variance, hệ số FFT,
entropy, autocorrelation, số đỉnh cực trị, hệ số trend... Sau đó dùng thuật toán **FRESH**
(FeatuRe Extraction on basis of Scalable Hypothesis tests): kiểm định giả thuyết (p-value mỗi
feature so với nhãn) + hiệu chỉnh multiple-testing (Benjamini-Hochberg-Yekutieli) để loại bỏ
feature không liên quan. Kết quả cuối chỉ còn vài chục feature có ý nghĩa thống kê, không phải
dùng cả 800 vào classifier — tránh overfitting và giảm chi phí runtime.

**Quan trọng:** bước lọc 800→còn lại chỉ chạy **một lần lúc train**. Lúc suy luận thời gian
thực, chỉ cần tính đúng tập feature đã chọn — chi phí runtime rẻ, không phải nỗi lo hiệu năng.

### Bước 3: Gradient Boosting phân loại

Feature vector đã lọc → Gradient Boosting → nhãn fall/không-fall hoặc xác suất.

## Input / Output cụ thể

| | Bài báo gốc | Map vào `rule_engine.py` hiện tại |
|---|---|---|
| Input | 2 chuỗi thời gian thô mỗi cửa sổ: `HeightWidthRatio(t)`, `FrameSpeed(t)` | `AR(t)` và `V_y(t)` — **đã tính sẵn** mỗi frame trong `_fall_indicators`, lấy trực tiếp từ `TrackState.pos_history` (`(t, y, height)` per track, dòng ~495 trở đi) |
| Output | 1 nhãn/xác suất **cho mỗi cửa sổ**, không phải cho cả video | 1 nhãn/xác suất **cho mỗi `track_id`, mỗi cửa sổ trượt** — tần suất tương tự `_evaluate_fall_indicators` hiện chạy mỗi frame |

## Xử lý nhiều người trong 1 khung hình

**Giới hạn cần biết trước khi áp dụng:** dataset gốc bài báo dùng để train — UR Fall Detection
(5 người, mỗi video 1 người) và CAUCAFall (10 người, mỗi video 1 người, quay trong nhà) — đều
là dataset **1-người-1-video**. Bài báo gốc **không giải quyết bài toán nhiều người trong 1
khung hình** — đây không phải thứ pipeline tsfresh+GB tự động xử lý sẵn, mà là phần cần tự
thiết kế khi áp vào camera an ninh thật (nhiều người, nhiều camera).

**Cách thích ứng — tận dụng đúng cơ chế project đã có sẵn:** vì `RuleEngine` đã track riêng
từng người theo `track_id` (`TrackState`), cách thích ứng tự nhiên là chạy 1 pipeline tsfresh
+GB **độc lập cho mỗi `track_id`**, không chia sẻ gì giữa các người:

- Mỗi người có `pos_history` riêng → chuỗi `AR(t)`, `V_y(t)` riêng → cửa sổ trượt riêng →
  feature vector riêng → dự đoán riêng.
- Không cần feature "liên người" để bài toán đúng — một người ngã là sự kiện độc lập với
  người khác trong khung hình.
- Track-stitching logic đã có sẵn (`_stitch_track` — xử lý khi 1 người bị che khuất tạm thời,
  track ID nhảy) tái sử dụng được luôn, không cần viết cơ chế theo dõi mới.

**Chi phí khi nhiều người:** tăng theo số người đang track × số camera, nhưng vì mỗi lần suy
luận chỉ là tính vài chục feature đã chọn (rẻ) + 1 lần `predict()` của Gradient Boosting
(microsecond-scale) — bottleneck thật sự vẫn nằm ở YOLO detection/tracking mỗi frame như hiện
tại, **không phải** ở bước ML này. Khác với hướng pose (multi-feature + fall angle qua
`yolo26l-seg.pt`) — pose phải chạy model nặng hơn cho mỗi người, còn tsfresh+GB chỉ thêm 1
phép tính nhẹ *sau khi* bbox/tracking đã có dữ liệu sẵn. Đây là lý do hướng này hợp với bài
toán đa camera/đa người hơn về mặt chi phí runtime.

**Việc kỹ thuật cần thêm khi nhiều người:** cấp phát buffer cửa sổ trượt riêng cho từng
`track_id` đang active (bộ nhớ: số track × độ dài cửa sổ × 2 kênh — không đáng kể), và dọn
buffer khi track hết hạn — `_evict_stale` (dòng 414-425) đã có cơ chế dọn track hết hạn, chỉ
cần hook thêm việc dọn buffer tsfresh vào đúng chỗ đó.

## Khác biệt triết lý so với rule engine hiện tại

Code hiện tại dùng **ngưỡng cố định do người tự chọn** (`fall_aspect_ratio=1.2`,
`fall_velocity_ratio=1.2`, `fall_upright_aspect_ratio=0.7`) — đây chính xác là loại ngưỡng bị
gãy trước case ngoài phân bố (ngã dọc, xem Context). Hướng tsfresh+GB **học pattern từ dữ liệu
có nhãn** thay vì đoán số tay — nếu có đủ case ngã dọc trong tập benchmark, model có thể tự tìm
ra pattern phân biệt thật, không nhất thiết đơn giản như "AR flattens".

## Điều kiện tiên quyết trước khi triển khai

1. **Dữ liệu train** — cần nhiều case ngã dọc đã gắn nhãn thật/giả. Tập benchmark hiện tại
   (`tests/fixtures/benchmark/cases.jsonl`) chỉ có 6 case — quá mỏng để train một classifier
   có ý nghĩa. Việc này trùng với hạng mục "test LLM trên dataset lớn hơn" đã ghi trong họp
   17/08 — cả hai đều nghẽn ở cùng một nút thắt: **thu thập thêm footage/case gắn nhãn**.
2. **Script train offline** — không chạy trong rule engine thời gian thực. Train 1 lần dùng
   `pos_history` đã thu thập, export model nhẹ (hoặc chỉ export tập feature + threshold đã học)
   để dùng lúc suy luận.
3. **Quyết định vị trí tích hợp** — thay thế hoàn toàn `_fall_indicators`, hay chạy song song
   như một chỉ báo thứ 4 bên cạnh 3 chỉ báo hiện tại (giữ được `reason` string dễ giải thích
   cho LLM stage, vốn là yêu cầu PRD §1.2)? Cần chốt trước khi code — ảnh hưởng tới việc có giữ
   được tính "auditable" của `reason` (một scalar học máy khó giải thích bằng "AR=0.83(>=1.2)").

## Việc tiếp theo

- [ ] Quyết định: thay thế hay bổ sung chỉ báo thứ 4 song song với 3 chỉ báo hiện tại.
- [ ] Thu thập thêm case ngã dọc thật (tự quay, theo đúng cách đã làm với case intrusion —
      xem commit `d2787fd`).
- [ ] Viết script offline: đọc `pos_history` đã lưu → chạy tsfresh → train Gradient Boosting →
      export.
- [ ] Benchmark so sánh: rule engine 3-chỉ-báo hiện tại vs tsfresh+GB, trên cùng tập case ngã
      dọc mới thu thập.
