# Gate 2

## Deliverables

- [Kiến trúc hệ thống và luồng xử lý](architecture.md)
- [Kết quả 5 test case thủ công](evaluation.md)
- [Video demo Gate 2](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing) — 3 phút

## Setup instructions

Yêu cầu: Python 3.11, Node.js 20 và model tại
`src/vision/weights/yolo26l.pt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe scripts/seed_users.py
```

Chạy backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --port 8000
```

Chạy frontend trong terminal khác:

```powershell
Set-Location src\frontend
npm ci
npm run dev
```

Mở `http://localhost:5173`. MediaMTX và FFmpeg chỉ cần thiết khi mô phỏng RTSP.

## Environment variables quan trọng

| Biến                   | Mục đích                            | Giá trị gợi ý                            |
| ---------------------- | ----------------------------------- | ---------------------------------------- |
| `MODEL_WEIGHTS_PATH`   | Đường dẫn model YOLO                | `src/vision/weights/yolo26l.pt`          |
| `DETECTOR_CONFIDENCE`  | Ngưỡng confidence của YOLO          | `0.40`                                   |
| `CAMERA_TARGET_FPS`    | FPS inference cho mỗi camera        | `5`                                      |
| `DATABASE_URL`         | Nơi lưu camera, incident và history | `sqlite:///./data/app.db` khi chạy local |
| `VLM_ENABLED`          | Bật/tắt AI assessment               | `true` hoặc `false`                      |
| `VLM_PROVIDER`         | Provider đánh giá ảnh               | `openai` hoặc `local`                    |
| `OPENAI_API_KEY`       | API key khi dùng OpenAI             | để trống nếu dùng local/tắt VLM          |
| `VLM_MODEL`            | Model OpenAI                        | `gpt-4o-mini`                            |
| `VLM_MAX_FRAMES`       | Số frame tối đa gửi cho VLM         | `6`                                      |
| `VLM_HISTORY_EXAMPLES` | Số incident cũ đưa vào context      | `6`                                      |

Ví dụ cấu hình OpenAI trong `.env`:

```dotenv
VLM_ENABLED=true
VLM_PROVIDER=openai
OPENAI_API_KEY=your-key
VLM_MODEL=gpt-4o-mini
```

Không commit file `.env` hoặc API key lên Git.

## Sample queries

Kiểm tra backend:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Đăng nhập và giữ session cookie:

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
  -Method Post -WebSession $session -ContentType 'application/json' `
  -Body $camera
```
