# Gate 2

Hệ thống hỗ trợ nhân viên an ninh theo dõi nhiều camera cùng lúc. Camera có thể lấy
từ video mẫu, video tải lên hoặc luồng RTSP. Backend đọc từng luồng, chạy YOLO để bám theo
người và đồ vật, sau đó dùng các luật theo thời gian để phát hiện té ngã, xâm nhập và đồ vật
bị bỏ quên.

Khi có cảnh báo, hệ thống gom các khung hình trước và sau sự kiện rồi gửi sang VLM để kiểm tra
lại. Kết quả của VLM chỉ có vai trò hỗ trợ. Nhân viên trực vẫn là người xác nhận cuối cùng cảnh
báo đó là thật hay giả.

Tài liệu kiến trúc và luồng dữ liệu nằm tại [architecture.md](architecture.md).

## Video demo

- Thời lượng: 3 phút
- Link: [P-091 - Gate 2 MVP Demo](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing)
- Nội dung: thêm camera, theo dõi luồng hình, nhận cảnh báo, xem VLM đánh giá và guard xác nhận.

Báo cáo 5 test case thủ công và ảnh chụp output nằm tại [evaluation.md](evaluation.md).

## Luồng demo chính

1. Nhân viên trực đăng nhập bằng tài khoản `guard`.
2. Thêm camera từ video hoặc địa chỉ RTSP.
3. Nếu cần phát hiện xâm nhập, bật chức năng Intrusion và vẽ vùng cấm trên ảnh xem trước.
4. Backend bắt đầu đọc hình và chạy nhận diện ngay sau khi camera được thêm.
5. Khi luật phát hiện sự kiện, cảnh báo xuất hiện trên dashboard.
6. VLM xem tối đa 6 khung hình trước/sau cảnh báo và trả về một trong ba kết quả: `real`,
   `false_positive` hoặc `uncertain`.
7. Nhân viên trực chọn sự kiện thật hoặc báo động giả. Kết quả này được lưu lại để thống kê và
   làm ví dụ cho những lần đánh giá VLM sau.
8. Tài khoản `manager` xem số lượng cảnh báo, thời gian xử lý và độ chính xác trên các mẫu đã
   được con người xác nhận.

## Yêu cầu trước khi chạy

- Python 3.11
- Node.js 20
- File YOLO tại `src/vision/weights/yolo26l.pt`
- OpenAI API key nếu muốn chạy VLM thật
- MediaMTX và FFmpeg nếu muốn giả lập camera RTSP bằng video

## Cài đặt trên máy local

Tạo môi trường Python và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Tạo tài khoản đầu tiên cho nhân viên trực và quản lý:

```powershell
.\.venv\Scripts\python.exe scripts/seed_users.py
```

Script sẽ hỏi tên đăng nhập và mật khẩu. Mật khẩu được băm trước khi lưu vào
`data/users.json`. File này không được commit lên Git.

Muốn dùng VLM thật, sửa ba dòng sau trong `.env`:

```dotenv
VLM_ENABLED=true
OPENAI_API_KEY=your-key
VLM_MODEL=gpt-4o-mini
```

Không đưa API key vào README, video demo hoặc commit.

Chạy backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --port 8000
```

Mở terminal thứ hai và chạy frontend:

```powershell
Set-Location src\frontend
npm ci
npm run dev
```

Mở `http://localhost:5173`. Có thể kiểm tra backend tại `http://localhost:8000/health`.

## Các biến môi trường đang dùng

| Biến                            | Dùng để làm gì                                          | Giá trị mặc định                   |
| ------------------------------- | ------------------------------------------------------- | ---------------------------------- |
| `APP_ENV`                       | Chế độ chạy `development` hoặc `production`             | `development`                      |
| `APP_HOST`, `APP_PORT`          | Địa chỉ và cổng của backend                             | `0.0.0.0`, `8000`                  |
| `OPENAI_API_KEY`                | Khóa gọi OpenAI cho bước VLM                            | để trống                           |
| `VLM_ENABLED`                   | Bật hoặc tắt bước kiểm tra bằng VLM                     | `false`                            |
| `VLM_MODEL`                     | Model dùng để xem các khung hình sự kiện                | `gpt-4o-mini`                      |
| `VLM_PRE_ROLL_SEC`              | Số giây hình được giữ trước cảnh báo                    | `5.0`                              |
| `VLM_POST_ROLL_SEC`             | Số giây hình được giữ sau cảnh báo                      | `2.0`                              |
| `VLM_MAX_FRAMES`                | Số khung hình tối đa gửi cho VLM                        | `6`                                |
| `VLM_TIMEOUT_SEC`               | Thời gian chờ một lần gọi VLM                           | `20.0`                             |
| `VLM_HISTORY_EXAMPLES`          | Số kết quả cũ được đưa vào làm ví dụ                    | `6`                                |
| `MODEL_WEIGHTS_PATH`            | Đường dẫn YOLO mà backend nhìn thấy                     | `src/vision/weights/yolo26l.pt`    |
| `MODEL_WEIGHTS_HOST_PATH`       | Đường dẫn YOLO trên máy host khi chạy Docker            | `./src/vision/weights/yolo26l.pt`  |
| `DETECTOR_TARGET_LABELS`        | Các nhãn YOLO được giữ lại                              | `person,backpack,handbag,suitcase` |
| `DETECTOR_CONFIDENCE`           | Ngưỡng nhận diện chung của YOLO                         | `0.40`                             |
| `CAMERA_TARGET_FPS`             | Số khung hình phân tích mỗi giây cho camera mới         | `5`                                |
| `ALERT_COOLDOWN_SEC`            | Khoảng nghỉ giữa hai cảnh báo cùng loại trên một camera | `30` giây                          |
| `FALL_POSTURE_ENABLED`          | Bật model tư thế té ngã bổ sung                         | `true`                             |
| `FALL_POSTURE_CONFIDENCE`       | Ngưỡng của model tư thế té ngã                          | `0.40`                             |
| `CORS_ORIGINS`, `ALLOWED_HOSTS` | Giới hạn frontend và hostname được phép gọi backend     | cấu hình local                     |
| `SESSION_COOKIE_SECURE`         | Chỉ gửi cookie qua HTTPS                                | `false` khi chạy local             |
| `LOGIN_*`                       | Số lần đăng nhập sai và thời gian khóa                  | xem `.env.example`                 |

Dữ liệu MVP đang được lưu bằng JSON và JSONL trong thư mục `data`.

## Thêm camera

Trong giao diện, vào **Cameras -> Add a camera** rồi chọn một trong ba nguồn:

- **Prepared demo:** dùng video đã có sẵn trong dự án.
- **Upload video:** tải một video lên; backend lưu file trong `data/uploads`.
- **RTSP stream:** nhập địa chỉ dạng `rtsp://127.0.0.1:8554/fall`.

Đối với Intrusion, phải bật chức năng và vẽ polygon trước khi thêm camera. Polygon nên bao
phần nền thực sự cần bảo vệ. Với ba video quay cổng, vùng cấm nên nằm phía trong sân; không
bao gồm vỉa hè và mặt đường phía ngoài.

Luật xâm nhập hiện kiểm tra điểm chân của người. Nếu điểm chân không ổn định do che khuất,
hệ thống dùng thêm phần thân dưới làm điều kiện dự phòng. Một phần vai hoặc chiếc ô chạm vào
polygon không đủ để tạo cảnh báo.

## Giả lập RTSP bằng video

Chạy MediaMTX và bảo đảm RTSP đang nghe tại cổng `8554`. Mỗi terminal có thể phát lặp một
video bằng lệnh sau:

```powershell
.\.venv\Scripts\python.exe scripts/publish_rtsp.py `
  data/vision/test/intrusion.mp4 `
  rtsp://127.0.0.1:8554/intrusion
```

Các địa chỉ dùng cho bộ video test:

```text
rtsp://127.0.0.1:8554/fall
rtsp://127.0.0.1:8554/abandoned
rtsp://127.0.0.1:8554/intrusion
rtsp://127.0.0.1:8554/intrusion-false
rtsp://127.0.0.1:8554/intrusion-umbrella
```

FFmpeg chỉ đọc video theo thời gian thực, lặp vô hạn và copy luồng H.264 nên phần phát RTSP
không tốn nhiều tài nguyên. Phần nặng chủ yếu nằm ở YOLO. Nếu backend chạy trong Docker, thay
`127.0.0.1` bằng `host.docker.internal`.

## Một số request mẫu

Kiểm tra backend:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Đăng nhập và giữ cookie phiên làm việc:

```powershell
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
Invoke-RestMethod http://localhost:8000/api/v1/auth/login `
  -Method Post -WebSession $session -ContentType 'application/json' `
  -Body '{"username":"guard","password":"your-password"}'
```

Lấy danh sách camera và cảnh báo:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/cameras -WebSession $session
Invoke-RestMethod http://localhost:8000/api/v1/alerts -WebSession $session
```

Thêm một camera RTSP:

```powershell
$camera = @{
  name = 'Fall test'
  source_type = 'rtsp'
  source = 'rtsp://127.0.0.1:8554/fall'
  intrusion_enabled = $false
  target_fps = 5
  loop_source = $true
} | ConvertTo-Json

Invoke-RestMethod http://localhost:8000/api/v1/cameras `
  -Method Post -WebSession $session -ContentType 'application/json' -Body $camera
```

Các tài liệu Gate 2:

- [Sơ đồ kiến trúc và luồng dữ liệu](architecture.md)
- [Kết quả 5 test case thủ công](evaluation.md)
- [Video demo 3 phút](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing)
