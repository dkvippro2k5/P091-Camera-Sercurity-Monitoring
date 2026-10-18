# Evaluation Report

- [Kết quả 5 test case thủ công](../../docs/gate2/evaluation.md)
- [Video demo 3 phút](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing)

## Benchmark té ngã trên UR Fall Detection Dataset

Detector + RuleEngine đúng cấu hình production, chạy offline trên 30 clip té ngã và 40 clip sinh
hoạt bình thường (ADL) có nhãn từng frame. Cách chạy và giao thức đánh giá nằm trong docstring của
`src/benchmark/run_fall_benchmark.py`.

| Lượt chạy | Recall | Specificity | Precision | F1 | Báo cáo |
| --- | --- | --- | --- | --- | --- |
| 5 fps, giữ frame cuối 4 s (chính) | 46.7% (14/30) [30–64%] | 92.5% [80–97%] | 82.4% [59–94%] | 59.6% | [md](fall-benchmark-urfall-padded.md) |
| 15 fps, giữ frame cuối 4 s | 66.7% (20/30) [49–81%] | 92.5% [80–97%] | 87.0% [68–96%] | 75.5% | [md](fall-benchmark-urfall-padded-15fps.md) |
| 5 fps, không pad | 0% (0/30) | 95.0% | — | — | [md](fall-benchmark-urfall-unpadded.md) |

Ngoặc vuông là khoảng tin cậy 95% (Wilson). Recall không pad bằng 0 là do dataset: mọi clip té ngã
kết thúc 0.1–2.1 s sau khi người chạm sàn, ngắn hơn ngưỡng nằm yên 3 s của rule.

**Nguyên nhân bỏ sót (16/30):**

- **9 clip — detector hầu như không thấy người** (người được phát hiện ở < 40% frame). Ở các clip
  đã xem (`fall-19`, `fall-27`), người ngã về phía camera, thân bị co ngắn theo phối cảnh và bị cắt
  ở mép dưới khung hình. Độ phân giải 320×240 của bản mp4 có thể làm nặng thêm.
- **6 clip — đổi track ID đúng lúc chạm sàn** (`fall-02/07/09/13/15/17`): detector mất người
  0.4–1.8 s quanh cú ngã, người xuất hiện lại với track mới ở trạng thái `upright`. Rule chỉ bắn
  khi *cùng một track* đi từ đứng → nằm, nên một track "sinh ra" đã nằm không bao giờ arm được.
  Đây chính là rủi ro ID switch mà `docs/plans/benchmark-criteria.md` (Nhóm 2) đã cảnh báo.
- **1 clip — khoá `recovered`** (`fall-01`): chỉ báo kích hoạt ở frame đầu tiên khi người bắt đầu
  cúi; frame kế tiếp bbox vẫn còn dáng đứng nên bị coi là đã đứng dậy (`recovered`). Trạng thái đó
  chỉ được xoá bởi một frame đứng thẳng không kích hoạt chỉ báo, nên cú ngã thật ngay sau không arm
  lại được; track mới sau khi mất dấu được stitch và thừa hưởng luôn trạng thái khoá.

**Tốc độ lấy mẫu quyết định nhóm lỗi tracking:** ở 15 fps, 6/7 clip trên được bắt (chỉ còn
`fall-09`); 9 clip detector hụt vẫn hụt. Tức là ở 5 fps (mặc định `CAMERA_TARGET_FPS`) khoảng
cách giữa hai frame quá lớn để ByteTrack giữ ID qua một cú ngã. Cấu hình 3 fps đang dùng để giảm
tải CPU sẽ còn tệ hơn — chưa đo.

**Báo động giả:** 3 clip ADL (`adl-01`, `adl-10`, `adl-34`), tương đương ~22 lần/giờ trên 8.3
phút footage âm. Con số theo giờ chỉ là ước lượng thô vì lượng footage âm quá ít.

Hạn chế: một dataset, một phòng, một góc camera, độ phân giải thấp. Chưa phải đánh giá tổng quát.
