# Evaluation Report

- [Kết quả 5 test case thủ công](../../docs/gate2/evaluation.md)
- [Video demo 3 phút](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing)

## Benchmark té ngã

Detector + RuleEngine đúng cấu hình production, chạy offline trên hai bộ dữ liệu công khai có nhãn.
Cách chạy và giao thức đánh giá nằm trong docstring của `src/benchmark/run_fall_benchmark.py`.

- **UR Fall Detection Dataset** — 30 clip té ngã, 40 ADL, một phòng, 320×240. Dùng để chẩn đoán.
- **GMDCSA24** — 78 clip té ngã, 81 ADL, 4 người, 3 căn nhà, ngày và đêm, 720p/480p. Tập kiểm
  chứng: các thay đổi bên dưới không được chỉnh theo nó.

Mặc định: 5 fps, giữ frame cuối 4 s. Ngoặc vuông là khoảng tin cậy 95% (Wilson).

| Phiên bản | UR Fall recall | UR Fall ADL bị báo | GMDCSA24 recall | GMDCSA24 ADL bị báo |
| --- | --- | --- | --- | --- |
| Baseline | 46.7% (14/30) [30–64%] | 3/40 | 44.9% (35/78) [34–56%] | 13/81 |
| 1a: cố định ByteTrack, detection yếu giữ track | 76.7% (23/30) | 6/40 | 48.7% (38/78) | 11/81 |
| 1 đầy đủ: + stitch theo cú ngã, + `recovered` phải giữ 0.6 s | 76.7% (23/30) [59–88%] | 6/40 | 51.3% (40/78) [40–62%] | 11/81 |

Chi tiết từng clip: `fall-benchmark-{urfall,gmdcsa24}-{baseline,step1a-tracker,step1}.md`.

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
- **GMDCSA24, recall chỉ 51%**: kiểu lỗi chủ đạo ở bộ này khác UR Fall và chưa được phân tích.
- **Báo nhầm tăng trên UR Fall (3 → 6)**: tracker giữ được người nằm thì cũng giữ được người chủ
  động nằm xuống (giường, sàn). Phân biệt ngã với chủ động nằm là bước tiếp theo (vận tốc rơi).

**Tốc độ lấy mẫu:** trước bước 1, 15 fps đẩy recall UR Fall lên 66.7% so với 46.7% ở 5 fps
([md](fall-benchmark-urfall-padded-15fps.md)). Cấu hình 3 fps đang dùng để giảm tải CPU chưa đo.

Hạn chế: clip ngắn nên phải giữ frame cuối; báo nhầm theo giờ chỉ là ước lượng thô vì footage âm
ít (8 phút ở UR Fall); nhãn UR Fall chỉ là cửa sổ 1 s cố định quanh cú ngã.
