# P-091 Security Monitor

Real-time multi-camera incident monitoring for security control rooms. The MVP combines
YOLO/OpenCV detection, temporal incident rules, optional local or OpenAI vision triage, and a
mandatory guard verdict in a React dashboard.

## Main User Flow

1. A guard signs in and adds a prepared demo, uploaded video, or RTSP camera.
2. The backend analyzes the camera in an independent worker.
3. YOLO tracks people and relevant objects. Temporal rules produce fall, intrusion, and
   abandoned-object incidents.
4. The browser receives annotated video over MJPEG and alerts over SSE.
5. When VLM triage is enabled, the selected provider reviews before/after frames, rule evidence,
   and recent labelled examples.
6. The guard records `real` or `false`. Managers see reviewed precision and operational counts.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component and data-flow diagrams.

## Prerequisites

- Python 3.11
- Node.js 20
- YOLO checkpoint at `src/vision/weights/yolo11n.pt`
- Optional: OpenAI API key or a CUDA-capable GPU for VLM triage
- Optional: MediaMTX and FFmpeg for looping local videos as RTSP cameras

## Local Setup

Create the Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Create the initial guard and manager accounts interactively:

```powershell
.\.venv\Scripts\python.exe scripts/seed_users.py
```

For local SmolVLM2 triage, set these values in `.env` without committing the file:

```dotenv
VLM_ENABLED=true
VLM_PROVIDER=local
VLM_MAX_FRAMES=4
VLM_LOCAL_MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
VLM_LOCAL_DEVICE=auto
VLM_LOCAL_MAX_IMAGE_SIZE=512
VLM_LOCAL_MAX_NEW_TOKENS=64
```

The local provider reads model weights from the Hugging Face cache and does not make inference
API calls. Download the checkpoint once before starting the monitor if it is not already cached.
To use OpenAI instead:

```dotenv
VLM_ENABLED=true
VLM_PROVIDER=openai
OPENAI_API_KEY=your-key
VLM_MODEL=gpt-4o-mini
```

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.main:app --reload --port 8000
```

In a second terminal, start the frontend:

```powershell
Set-Location src\frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The API health check is `http://localhost:8000/health`.

Each live camera tile reports its rolling p95 inference latency beside the inference-capacity
estimate. The backend computes p50, p95, and p99 over the latest 100 successfully processed
frames and separately counts processing failures, so transient slowdowns do not disappear into
a single last-frame number.

## VLM Regression Check

The regression runner samples chronological frames from the local fall and normal test videos.
Its default mode only validates and lists cases; it cannot call the API:

```powershell
.\.venv\Scripts\python.exe eval/vlm_regression.py
```

Run the billed OpenAI check explicitly, optionally persisting the JSON result:

```powershell
.\.venv\Scripts\python.exe eval/vlm_regression.py --execute `
  --output eval/results/vlm-regression.json
```

Each case sends six high-detail images. Review OpenAI image-token pricing before running it, and
keep `data/vision/test/fall.mp4` and `normal1.mp4` available for the bundled cases.

## Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_ENV`, `APP_HOST`, `APP_PORT` | Runtime mode and bind address | `development`, `0.0.0.0`, `8000` |
| `OPENAI_API_KEY` | OpenAI credential when `VLM_PROVIDER=openai` | empty |
| `VLM_ENABLED`, `VLM_PROVIDER` | Enable evidence review and select `openai` or `local` | `false`, `openai` |
| `VLM_MODEL` | OpenAI vision model | `gpt-4o-mini` |
| `VLM_LOCAL_*` | Local model, device, image size, and generation limit | see `.env.example` |
| `VLM_PRE_ROLL_SEC`, `VLM_POST_ROLL_SEC` | Evidence window around an alert | `5.0`, `2.0` |
| `VLM_MAX_FRAMES` | Maximum sampled evidence frames | `6` |
| `VLM_TIMEOUT_SEC` | OpenAI request timeout | `20.0` |
| `VLM_HISTORY_EXAMPLES` | Prior labelled examples sent to VLM | `6` |
| `MODEL_WEIGHTS_PATH` | Detector checkpoint visible to the backend | `src/vision/weights/yolo11n.pt` |
| `MODEL_WEIGHTS_HOST_PATH` | Host checkpoint mounted by Docker Compose | `./src/vision/weights/yolo11n.pt` |
| `DETECTOR_TARGET_LABELS` | YOLO classes retained by the pipeline | `person,backpack,handbag,suitcase` |
| `DETECTOR_CONFIDENCE` | Main YOLO acceptance threshold | `0.40` |
| `CAMERA_TARGET_FPS` | Default analysis rate for new cameras | `5` |
| `CAMERA_MAX_CONCURRENT_INFERENCES` | Simultaneous camera inference limit; `0` selects a host-aware default | `0` |
| `ALERT_COOLDOWN_SEC` | Minimum gap per camera and incident label | `30` seconds |
| `CORS_ORIGINS`, `ALLOWED_HOSTS` | Browser and host security policy | local values |
| `SESSION_COOKIE_SECURE` | Require HTTPS cookies | `false` locally |
| `LOGIN_*` | Login failure and lockout limits | see `.env.example` |

`DATABASE_URL`, `CHROMA_PERSIST_DIR`, and LangSmith settings are retained for optional/template
integrations. The monitoring MVP persists to `data/*.json` and `data/*.jsonl`.

## Add A Camera

Go to **Cameras -> Add a camera** and choose one source:

- **Prepared demo:** select a bundled clip.
- **Upload video:** choose an MP4; the backend stores it under `data/uploads`.
- **RTSP stream:** enter a URL such as `rtsp://127.0.0.1:8554/fall`.

Enable intrusion only after drawing a polygon. Draw the protected ground area, not the whole
frame. For the gate clips under `data/vision/test`, exclude the road and sidewalk and include
the courtyard behind the gate.

## Loop Local Videos As RTSP

Start MediaMTX with RTSP listening on port `8554`. Publish one looping video per terminal:

```powershell
.\.venv\Scripts\python.exe scripts/publish_rtsp.py `
  data/vision/test/intrusion.mp4 `
  rtsp://127.0.0.1:8554/intrusion
```

Example stream URLs:

```text
rtsp://127.0.0.1:8554/fall
rtsp://127.0.0.1:8554/abandoned
rtsp://127.0.0.1:8554/intrusion
rtsp://127.0.0.1:8554/intrusion-false
rtsp://127.0.0.1:8554/intrusion-umbrella
```

The helper uses real-time input pacing, infinite looping, and video stream copy. Publishing is
cheap; YOLO inference is the expensive part. If the backend runs in Docker, use
`rtsp://host.docker.internal:8554/<path>` instead of `127.0.0.1`.

## Sample API Requests

Check the service:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Login and retain the HttpOnly session cookie:

```powershell
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
Invoke-RestMethod http://localhost:8000/api/v1/auth/login `
  -Method Post -WebSession $session -ContentType 'application/json' `
  -Body '{"username":"guard","password":"your-password"}'
```

List cameras and alerts:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/cameras -WebSession $session
Invoke-RestMethod http://localhost:8000/api/v1/alerts -WebSession $session
```

Register an RTSP camera:

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

The optional scaffold chat route is separate from the camera MVP. A sample query is:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/chat `
  -Method Post -WebSession $session -ContentType 'application/json' `
  -Body '{"message":"Explain the current monitoring workflow"}'
```

This route demonstrates the conversational LangGraph graph. Production monitoring uses a
separate asynchronous LangGraph incident workflow after an alert is created; YOLO and temporal
rules remain outside LangGraph for predictable real-time latency.

## Docker Compose

Set `MODEL_WEIGHTS_HOST_PATH` and production values in `.env`, then run:

```powershell
docker compose up -d --build
docker compose logs -f backend
```

The frontend is exposed on `http://localhost`; nginx proxies `/api/` to the backend. Runtime
camera configuration, uploads, incident logs, and VLM history remain under `./data`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests eval
.\.venv\Scripts\python.exe -m pytest -q
Set-Location src\frontend
npm ci
npm run build
```

## Documentation

- [Architecture and data flow](ARCHITECTURE.md)
- [Compact architecture diagram](docs/architecture_diagram.md)
- [Evaluation results](eval/results/report.md)
- [Gate 2 documentation in Vietnamese](docs/gate2/README.md)
- [Three-minute MVP demo](https://drive.google.com/file/d/1DwTT9Bh_V6_9cmfgZGWwqVtIRbyloHsS/view?usp=sharing)
