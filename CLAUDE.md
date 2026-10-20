# CLAUDE.md

Hệ thống giám sát camera P091: YOLO + ByteTrack, RuleEngine phát hiện té ngã, xâm nhập vùng cấm, đồ vật bỏ quên, và VLM để phân loại lại cảnh báo.

## Giao tiếp
- Luôn trả lời bằng tiếng Việt.
- Báo cáo kết quả trước khi commit, rồi chờ người dùng đồng ý.

## Commit và push
- **Không thêm dòng ghi công Claude vào commit hay PR**: không có `Co-Authored-By: Claude ...`, không có "Generated with Claude Code". Trước mỗi lần commit hoặc push, kiểm tra và xóa các dòng này, rồi tiếp tục công việc.
- Mỗi commit chỉ chứa đúng phần vừa làm, không kèm thay đổi khác.
- Không commit và không sửa `docs/interview-prep/`. Không lặp lại thông tin đăng nhập trong `.env`.
- Thông điệp commit theo dạng `feat(scope): ...`, `fix(scope): ...`, `docs: ...`, viết bằng tiếng Anh.

## Kiểm tra
- `python -m pytest -q tests/` và `ruff check src tests` phải sạch trước khi commit.
- Code và comment viết theo phong cách code xung quanh.

## Benchmark
- Mã nguồn: `src/benchmark/run_{fall,intrusion,abandoned}_benchmark.py`; kết quả lưu ở `eval/results/`.
- Dữ liệu nằm trong `data/benchmark/` (gitignored). Video đã được chuyển sang MP4 bằng `src.benchmark.video_convert`.
- Mọi so sánh trước/sau khi sửa phải đo trên cùng bộ file MP4.
- Xem bằng chứng (frame, log) trước khi sửa rule.
- Ghi rõ nhãn nào do mình tự gán, không phải ground truth gốc của bộ dữ liệu.
