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

## Benchmark đồ vật bỏ quên

Detector + RuleEngine đúng cấu hình production, chạy offline, chấm theo **sự kiện** (mỗi lần bỏ
quên), không theo clip. Giao thức và nguồn dữ liệu nằm trong docstring của
`src/benchmark/run_abandoned_benchmark.py`; kiến trúc ở [docs/abandoned-object.md](../../docs/abandoned-object.md).

- **ABODA** — 11 clip, 640/720×480, trong nhà và ngoài trời, ngày và đêm (hồng ngoại), một sảnh
  đông người. 11 lần bỏ quên + 1 âm tính khó (`video11`: chủ đứng cạnh túi suốt clip).
- **AVSS 2007** (i-LIDS) — 3 clip sân ga tàu điện ngầm, 720×576, ~3 phút, đông người, tàu ra vào.
  3 lần bỏ quên.
- **CAVIAR INRIA** — 25 clip sảnh, 384×288, camera mắt cá. 4 lần bỏ quên + 21 clip không có (đi,
  xem, gặp nhau, ngồi, đánh nhau) để đo báo nhầm.

Tổng 18 lần bỏ quên, 46 phút video. Mọi lần đo dùng cùng bộ file MP4 (`src.benchmark.video_convert`);
số đo trên `.mpg` gốc không so được với bảng dưới.

**Nhãn tự gán, không phải ground truth gốc:**
- Toàn bộ nhãn ABODA (`ABODA_EVENTS`): bộ dữ liệu không kèm annotation. Mình gán bằng cách xem
  từng clip và trừ ảnh trước/sau; người dùng đã kiểm tra lại (`video5` sửa thành đặt túi 31 s, rời
  đi 67 s).
- Mốc "chủ rời đi" của AVSS (`AVSS_OWNER_LEFT_SEC`): ground truth AVSS chỉ có lúc đặt, lúc lấy và
  một mốc "abandoned" = lúc đặt + 30 s. MEDIUM sửa thành 84 s (chủ là người phụ nữ đặt túi lúc 47 s).
- CAVIAR: lúc đặt/lấy là nhãn gốc; "chủ rời đi" được suy ra là lúc người gần vật nhất khi vật xuất
  hiện ở xa hơn 25% bề rộng khung quá 1 s. `LeftBag_BehindChair` và `Fight_OneManDown` bị loại.

Một cảnh báo trên vật trước `chủ rời đi − 3 s` là báo nhầm ("early"). `ABANDON_SEC` = 2 s (mặc định
demo), 5 fps. Ngoặc vuông là khoảng tin cậy 95% (Wilson).

| Phiên bản | Recall (3 bộ) | Báo nhầm | Precision | ABODA | AVSS 2007 | CAVIAR INRIA |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline: trạng thái theo track, chủ là người gần nhất | 5/18 [12–51%] | 10 (13/h) | 33% | 4/11, 3 BN | 1/3, 7 BN | 0/4, 0 BN |
| 1a: túi theo vị trí, chủ là mọi người quanh túi, bắt buộc có chủ | 7/18 [20–61%] | 1 (1.3/h) | 88% | 4/11, 0 BN | 3/3, 1 BN | 0/4, 0 BN |
| 1b: 1a + nhận lại chủ bằng màu áo (**tắt mặc định**) | 7/18 | 0 | 100% | 4/11, 0 BN | 3/3, 0 BN | 0/4, 0 BN |
| 3: 1a + background model (**hiện tại, bật mặc định**) | 16/18 [67–97%] | 10 (13/h) | 62% | 10/11, 2 BN | 3/3, 8 BN | 3/4, 0 BN |

BN = báo nhầm. Chi tiết từng clip: `abandoned-benchmark-{aboda,avss2007,caviar-inria}-{baseline,step1a,step1b,step3,step3-check}.md`.
Độ trễ từ lúc chủ rời đi đến lúc báo ở bước 3: trung vị 3.2 s (ABODA), 7.2 s (AVSS), 2.0 s (CAVIAR).

### Baseline: trạng thái gắn với track

Rule gốc lưu đồng hồ và chủ theo track ID của túi, chủ là người gần nhất lúc túi xuất hiện. Trên
sân ga AVSS một vali lần lượt là `suitcase`, `handbag`, rồi bị người đi qua che 10 s — mỗi lần là
một track mới, nên túi quên chủ và nhận người đứng gần nhất lúc đó. AVSS MEDIUM: vali của người đàn
ông đứng ngay cạnh bị báo 4 lần. Ở ABODA, báo nhầm là túi đã nằm sẵn khi camera bắt đầu quay.

### Bước 1a — túi theo vị trí, bắt buộc có chủ

- Trạng thái lưu theo **vật ở một chỗ**: detection nối vào túi có cùng track, hoặc túi nằm đúng chỗ
  đó; túi chỉ bị quên sau 30 s không thấy. Đổi nhãn, đổi ID, bị che đều giữ nguyên chủ và đồng hồ.
- **Chủ là mọi người** trong bán kính (25% bề rộng khung) khi túi đang được mang hoặc trong 3 s đầu
  sau khi nằm yên — không chỉ người gần nhất.
- Túi chưa từng có chủ thì không báo (`ABANDON_REQUIRE_OWNER`).
- Track mới đứng đúng chỗ (IoU ≥ 0.3) của một chủ vừa mất trong 2 s là chủ đó bị đánh số lại.
- Túi nằm trong khung người không báo ở frame đó, nhưng đồng hồ không reset.

Báo nhầm còn lại (AVSS MEDIUM 141 s): chủ ngồi trên ghế bị che, đổi ID và lệch chỗ đứng.

### Bước 1b — nhận lại chủ bằng màu áo: tắt mặc định

Histogram màu thân trên và chân của người quanh túi. Bỏ được báo nhầm cuối cùng ở 0.85, nhưng 0.8
mất ca AVSS HARD và 0.9 làm báo nhầm quay lại, và ngưỡng được chọn trên chính dữ liệu đo — quá mong
manh để bật. Giữ code cho đến khi có ReID thật (ví dụ OSNet).

### Bước 3 — background model cho vật YOLO không thấy

Sau bước 1a, **cả 11 ca sót đều do YOLO không đóng khung vật**: nhỏ, xa, quay hồng ngoại, hoặc không
phải lớp COCO (túi ni lông trên bàn, hộp giấy). Vật vẫn hiện rõ khi so với cảnh trống, nên
`StaticObjectDetector` giữ một ảnh nền chỉ cập nhật ở pixel khớp nền, và báo vùng khác nền liên tục
8 s. Vùng tĩnh vào rule như một túi không bao giờ di chuyển; chủ lấy từ vị trí người quanh thời điểm
vùng xuất hiện. Cơ chế và các bộ lọc: [docs/abandoned-object.md](../../docs/abandoned-object.md) mục 4.

- Bắt thêm 9 lần bỏ quên: ABODA `video2/3/5/9`, cả 2 ca `video6`; CAVIAR `LeftBag`, `LeftBox`,
  `LeftBag_PickedUp`. Còn sót: ABODA `video4` (túi sát hộp điện, đường viền lẫn vào cạnh hộp) và
  CAVIAR `LeftBag_AtChair` (túi ~11×9 px cạnh ghế).
- 9 báo nhầm mới: AVSS 7 (tàu vào/rời ga, cửa tàu, mép sân ga), ABODA 2 (`video5` báo sớm 0.4 s so
  với dung sai; `video11` người xếp hàng YOLO chỉ thấy một phần). CAVIAR 21 clip âm tính vẫn sạch.
- Bật mặc định theo quyết định "thà báo nhầm còn hơn bỏ sót": hợp với sảnh, hành lang, lớp học (ABODA
  5.7 BN/h, CAVIAR 0), chưa hợp với sân ga có tàu (AVSS ~52 BN/h).
- Chi phí ~10 ms/frame trên CPU (720p–1080p), tức ~5% một nhân mỗi camera ở 5 fps.

Chưa có bộ lọc nào, background model báo nhầm 103 lần (đo bằng replay detection đã cache, khớp
benchmark chính thức). Các bước đưa về 10, mỗi bước chọn sau khi xem ảnh các ca báo nhầm:

1. Bỏ vùng có đường viền thuộc về nền chứ không phải vật — bóng ma của thứ đã rời đi, đổi sáng:
   103 → 25.
2. Bắt buộc có khung người chạm vào vùng lúc nó xuất hiện — tàu, màn hình: → 16. Sửa một lỗi
   (`since` của vùng thành NaN khi phép đóng hình thái thêm pixel chưa tĩnh, khiến vùng không bao
   giờ có chủ) làm con số thật lên lại 27.
3. `static_sec` 5 → 8 s, lề khung người 10% → 25%, che vị trí người trong 2 s sau khi YOLO mất họ: → 21.
4. Bỏ hysteresis của ngưỡng "khác nền": khi nền đã lệch nhẹ, người đi qua đẩy pixel vượt ngưỡng và
   hysteresis giữ chúng lại, tạo một "vật" là vệt sàn sau lưng mỗi người: → 11.
5. Giới hạn diện tích 2% khung, bỏ thân tàu: → 9 trên replay, 10 với module cuối cùng.

**Các ngưỡng của bước 3 được chọn trên chính 18 sự kiện này** (tỷ lệ cạnh mới 0.4, ngưỡng cạnh 60,
diện tích tối đa 2%, `static_sec` 8 s, lề khung người 25%, che người 2 s) — không có tập kiểm chứng
độc lập, nên số liệu bước 3 lạc quan hơn thực tế.

### YOLOE ghi chú cảnh báo của background model (`step3-check`)

Mỗi cảnh báo từ background model được YOLOE (open-vocabulary, bộ từ cố định: túi các loại, hộp,
người, tàu, cửa, màn hình) xem trên ảnh cắt, và kết quả được ghi vào `reason`. Không chặn cảnh báo,
nên số phát hiện không đổi (16/18, 10 BN).

| YOLOE ghi | Cảnh báo | Thực tế |
| --- | --- | --- |
| giống túi, ≥ 0.25 | 5 | 5 túi thật |
| giống tàu / cửa / người, ≥ 0.25 | 5 | 5 báo nhầm |
| không nhận rõ (< 0.25 hoặc không thấy gì) | 13 | 9 túi thật (nhỏ, xa, hồng ngoại), 4 báo nhầm |

Không dùng để chặn vì chính những túi background model được thêm vào để bắt lại là những túi YOLOE
không nhận ra. Ngưỡng 0.25 được đặt sau khi thấy hai túi hồng ngoại ở ABODA bị gọi là "person" với
0.06 và 0.22.

Ảnh trước/sau gửi cho VLM (mục 7 của tài liệu kiến trúc) chưa được đo: benchmark không chạy VLM.
