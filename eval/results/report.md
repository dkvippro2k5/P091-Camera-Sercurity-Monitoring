# Evaluation Report

- [Kết quả 5 test case thủ công](../../docs/gate2/evaluation.md)
- [Video demo 3 phút](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing)

## Benchmark té ngã

Detector + RuleEngine đúng cấu hình production, chạy offline trên hai bộ dữ liệu công khai có nhãn.
Cách chạy và giao thức đánh giá nằm trong docstring của `src/benchmark/run_fall_benchmark.py`.

- **UR Fall Detection Dataset** — 30 clip té ngã, 40 ADL, một phòng, 320×240. Dùng để chẩn đoán.
- **GMDCSA24** — 78 clip té ngã, 81 ADL, 4 người, 3 căn nhà, ngày và đêm, 720p/480p. Là tập
  kiểm chứng cho bước 1; sau đó đã được dùng để chẩn đoán nên không còn độc lập.
- **CAUCAFall** — 50 clip té ngã, 50 ADL (đi, nhảy, quỳ, ngồi, nhặt đồ), 10 người, 720×480,
  ánh sáng tự nhiên/nhân tạo/đêm. Tập kiểm chứng từ bước 2.

Cả ba đều quay trong nhà, phần lớn là ngã dàn dựng (chậm, có chủ đích). Chưa có dữ liệu ngoài
trời, đông người hay ngã thật.

Mặc định: 5 fps, giữ frame cuối 4 s. Ngoặc vuông là khoảng tin cậy 95% (Wilson).

| Phiên bản | UR Fall recall | UR Fall ADL bị báo | GMDCSA24 recall | GMDCSA24 ADL bị báo | CAUCAFall recall | CAUCAFall ADL bị báo |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 46.7% (14/30) [30–64%] | 3/40 | 44.9% (35/78) [34–56%] | 13/81 | — | — |
| 1a: cố định ByteTrack, detection yếu giữ track | 76.7% (23/30) | 6/40 | 48.7% (38/78) | 11/81 | — | — |
| 1 đầy đủ: + stitch theo cú ngã, + `recovered` phải giữ 0.6 s | 76.7% (23/30) [59–88%] | 6/40 | 51.3% (40/78) [40–62%] | 11/81 | 46.0% (23/50) [33–60%] | 10/50 |
| 2: chặn ngã có tốc độ rơi < 0.5 H/s (**tắt mặc định**) | 73.3% (22/30) | 2/40 | 42.3% (33/78) | 4/81 | 24.0% (12/50) [14–37%] | 5/50 |
| 3: + tầng "nằm kéo dài", + giữ class người, + track đổi class bắt đầu lại | 86.7% (26/30) [70–95%] | 9/40 | 79.5% (62/78) [69–87%] | 14/81 | 72.0% (36/50) [58–83%] | 14/50 |
| 4: + gộp alert theo sự cố, + tầng pose chặn khi thân thẳng (**hiện tại**) | 86.7% (26/30) [70–95%] | 7/40 | 78.2% (61/78) [68–86%] | 11/81 | 70.0% (35/50) [56–81%] | 4/50 |

Chi tiết từng clip: `fall-benchmark-{urfall,gmdcsa24,caucafall}-{baseline,step1a-tracker,step1,step2-descent,step3-static,step4-pose}.md`.

### Nguyên nhân bỏ sót ở baseline (UR Fall, 16/30)

Bản phân tích đầu tiên quy 9 clip cho "detector không thấy người"; điều đó sai. Chỉ số "người
trong khung" khi ấy bị đoạn pad chi phối, và YOLO chạy riêng vẫn thấy người nằm ở 0.4–0.8 trên
phần lớn các clip đó. Nguyên nhân thật gần như đều là tracking lúc chạm sàn:

- **Tracker mặc định không phải ByteTrack.** Code gọi `model.track()` không chỉ định tracker, nên
  ultralytics 8.4.143 dùng TrackTrack (`new_track_thresh` 0.7). Người nằm chỉ đạt 0.3–0.6: không
  tạo được track mới, còn track cũ bị Kalman ngoại suy trượt khỏi khung theo vận tốc rơi.
- **Detection dưới `DETECTOR_CONFIDENCE` (0.4) bị lọc trước khi vào tracker**, nên tầng ghép
  điểm thấp của ByteTrack không bao giờ được dùng.
- **Stitch dùng bán kính theo chiều rộng khung** (~38 px), nhỏ hơn quãng tâm bbox tụt xuống khi
  ngã (~nửa thân người), nên track mới sau khi đổi ID bắt đầu lại ở `upright` và không bao giờ arm.
- **Khoá `recovered`** (`fall-01`): một frame còn dáng đứng ngay sau khi chỉ báo arm bị coi là đã
  đứng dậy, và cú ngã thật ngay sau đó không arm lại được.

### Còn lại sau bước 1

- **UR Fall, 7 clip** (`fall-05/18/19/21/25/27/29`): người ngã về phía camera và nằm sát mép
  dưới; detection chỉ 0.1–0.3 hoặc mất hẳn trước khi kịp vào pha ngã. ByteTrack chỉ dùng
  detection yếu cho track *đang* theo dõi, không hồi sinh track đã mất. Cần detector/pose tốt hơn
  cho người nằm, không sửa được bằng logic rule.
- **GMDCSA24, 38 ca bỏ sót**: 25 ca hỏng ở máy trạng thái "cùng một track đi từ đứng sang nằm"
  — 13 đã arm nhưng chưa bao giờ được xác nhận nằm hoặc đồng hồ dwell bị reset khi người cựa quậy
  trên sàn, 6 ngã từ tư thế ngồi nên không có baseline đứng, 6 không đủ 2 chỉ báo (ngã chúi về phía
  camera, ngã chậm). 9 ca detector mất người (4 ca ban đêm); 4 ca chưa phân loại được.
- **CAUCAFall, báo nhầm 10/50**: 7 trong số đó là đi bộ hoặc nhảy, không phải nằm xuống.
- **Báo nhầm tăng trên UR Fall (3 → 6)**: tracker giữ được người nằm thì cũng giữ được người chủ
  động nằm xuống (giường, sàn).

### Bước 3 — tầng "nằm kéo dài" và lỗi class của tracker

Tầng thứ hai của rule té ngã: người có box dáng nằm (rộng/cao ≥ 1.2) liên tục 3 s thì báo, kể cả
khi không thấy chuyển từ đứng sang nằm — ngã từ tư thế ngồi, hoặc track chỉ xuất hiện khi người
đã nằm. Không báo nếu ≥ 50% box người nằm trong box `bed`/`couch` (YOLO nhận các lớp này; nhớ vị
trí 10 s vì detection chập chờn).

Lần chạy đầu làm mất 8 ca trên GMDCSA24 và lộ ra một lỗi có sẵn: tracker ghép box không phân biệt
class, mà YOLO hay gọi người nằm sàn là `bed` và đôi khi gọi giường là `person`. Track giữ nguyên
ID qua lần đổi nhãn — người biến mất giữa cú ngã, hoặc trạng thái của giường đè lên trạng thái của
người. Một số alert "đúng" trước đó chỉ là may. Sửa: track đã từng là người thì vẫn là người
(detector), và track đổi class thì bắt đầu trạng thái mới để được stitch đúng (rule engine).

Một người nằm sàn thường là nhiều track cùng lúc (box toàn thân, box một phần, box giường bị gọi
là người), nên mỗi track tự báo: 17 clip có 2–3 alert cho cùng một cú ngã (0.4–7 s cách nhau),
mỗi alert là một popup và một lần gọi VLM. Alert té ngã giờ được gộp theo sự cố: alert gần một
alert té ngã khác trên cùng camera trong cửa sổ cooldown (30 s) bị bỏ; người ngã ở chỗ khác vẫn
được báo. Các số `step3-static` trong bảng được chạy *trước* khi có phần gộp này; lần chạy đầy đủ
ở bước 4 có 0 alert trùng trên cả ba bộ (trước đó 3 / 12 / 2).

Trên tập kiểm chứng CAUCAFall recall tăng 46% → 72%. Cái giá: báo nhầm tăng 3–4 clip mỗi bộ, chủ
yếu là quỳ, ngồi xuống, nhặt đồ (CAUCAFall `kneel` ×2, `sit-down`, `pick-up-object`) — đúng rủi ro
đã biết của một luật chỉ nhìn hình dạng box. Đây là việc của tầng pose phía sau.

### Bước 4 — tầng pose: chặn alert khi thân người vẫn thẳng

Box chỉ cho biết người "rộng hơn cao"; quỳ, ngồi xuống, nhặt đồ cũng vậy. Thân người thì phân biệt
được: vai nằm trên hông là đang thẳng. Khi một track đang được đếm giờ chờ báo ngã (cả hai tầng),
pipeline chạy `yolo26s-pose` trên frame và đo góc đoạn vai–hông so với phương thẳng đứng. Đến lúc
báo, nếu ≥ 60% số frame trong 3 s trước đó *thấy được cả đầu lẫn mắt cá* có thân nghiêng < 45° thì
alert bị giữ lại, và được xét lại mỗi frame (quỳ rồi mới gục xuống vẫn được báo). Frame bị khuất
một phần không được tính; dưới 50% frame dùng được thì không chặn. Tầng này không bao giờ chặn tư
thế nằm, nên không thay đổi các ca giường. Góc thân đo được ghi vào `reason` của alert.

Ngưỡng 45° / 60% là số tròn, không tối ưu, nhưng được chọn sau khi đã xem số liệu pose của cả ba
bộ, kể cả CAUCAFall — nên với tầng này CAUCAFall *không còn là tập kiểm chứng độc lập*.

| | UR Fall | GMDCSA24 | CAUCAFall |
| --- | --- | --- | --- |
| Clip ADL bị báo | 9 → 7 | 14 → 11 | 14 → 4 |
| Ngã bắt được | 26 → 26 | 62 → 61 | 36 → 35 |
| Precision | 74.3% → 78.8% | 79.5% → 84.7% | 72.0% → 89.7% |
| F1 | 0.80 → 0.82 | 0.80 → 0.81 | 0.72 → 0.79 |

- Bị chặn đúng: ngồi xuống / ngồi dưới sàn (UR Fall `adl-01`, `adl-33`; GMDCSA24 `s2-adl-09/13/19`;
  CAUCAFall `sit-down` ×2), CAUCAFall `s08-kneel`, cùng 7 ca đi bộ/nhảy ở CAUCAFall mà tầng chuyển trạng thái báo khi box
  vẫn đứng (rộng/cao ~0.4) — lỗi đó của máy trạng thái vẫn còn, pose chỉ che nó.
- Mất 2 cú ngã: GMDCSA24 `s1-fall-13` (đầu ngoài khung) và CAUCAFall `s02-fall-sitting` (ngã ngồi
  bệt, kết thúc ở tư thế ngồi — bị chặn đúng theo thiết kế).
- Còn lại: CAUCAFall `kneel` ×2, `pick-up-object` ×2 (không đủ frame thấy toàn thân, hoặc thân cúi
  > 45°). 18 clip báo nhầm còn lại ở UR Fall và GMDCSA24 phần lớn là người nằm thật: ngủ trên giường
  (tầng chuyển trạng thái chưa kiểm tra giường), hít đất, nằm sàn. Pose không tách được các ca này.
- Chi phí: một lần pose ~89 ms trên CPU ở 720×480 (detector ~81 ms), chỉ chạy trên frame có track
  đang chờ báo ngã; thời gian chạy benchmark tăng ~10% trên hai bộ toàn cảnh có ngã.

### Bước 2 — tốc độ rơi: kết quả âm tính

Ngưỡng 0.5 chiều cao/giây được chọn trên UR Fall + GMDCSA24 (giữ 53/63 alert đúng, loại 13/17
alert nhầm). Trên CAUCAFall nó làm recall rơi từ 46% xuống 24%. Đo lại không bật bộ lọc: ngã thật
có tốc độ 0.0–1.1, alert nhầm 0.14–0.95 — gần như chồng lên nhau. Hai lý do: ngã dàn dựng trong
các bộ này chậm như chủ động nằm, và ngã về phía/ra xa camera gần như không làm mép trên bbox di
chuyển trong ảnh. Code được giữ (`fall_min_descent_speed`, mặc định 0) để dùng làm một tầng lọc
phụ phía sau tầng khác; tốc độ rơi đo được luôn có trong `reason` của alert.

**Tốc độ lấy mẫu:** trước bước 1, 15 fps đẩy recall UR Fall lên 66.7% so với 46.7% ở 5 fps
([md](fall-benchmark-urfall-padded-15fps.md)). Cấu hình 3 fps đang dùng để giảm tải CPU chưa đo.

Hạn chế: clip ngắn nên phải giữ frame cuối; báo nhầm theo giờ chỉ là ước lượng thô vì footage âm
ít (8 phút ở UR Fall); nhãn UR Fall chỉ là cửa sổ 1 s cố định quanh cú ngã.
