# P-091 — Hệ Thống AI Agent Giám Sát An Ninh Khu Đô Thị

AI Agent chạy trên local server, phân tích luồng camera RTSP để phát hiện 3 nhãn sự cố, dùng LLM suy luận ngữ cảnh lọc báo động giả, đẩy cảnh báo real-time lên dashboard cho bảo vệ xác nhận (HITL).

Dự án AI20K Build Phase Cohort 3, PoC 6 tuần. Repo: `AI20K-Build-Phase-Cohort-3/P-091`.

## Ràng buộc bất di bất dịch

Ba ràng buộc này đến từ PRD, không được vi phạm dù có lý do kỹ thuật gì:

1. **HITL bắt buộc** — không có đường nào từ AI Agent đi thẳng tới kích hoạt báo động phần cứng. Mọi cảnh báo phải qua bảo vệ xác nhận trên dashboard (`incidents.status: pending → confirmed`).
2. **Làm mờ mặt bắt buộc** — ảnh gốc chưa blur không bao giờ rời khỏi Khối 3 (Core Backend). Chỉ `frame_url` (đã blur, trên MinIO) được gửi ra ngoài.
3. **Dữ liệu không ra ngoài local server** — không upload video/ảnh giám sát lên cloud service bên ngoài.

## Ngoài phạm vi PoC

Không implement: Face ID / nhận diện danh tính sinh trắc học, tự động hoá phần cứng vật lý, multi-camera tracking (theo dõi chéo một đối tượng qua nhiều camera), triển khai trên phần cứng thật.

## Kiến trúc — 4 khối

```
Camera RTSP → [K1 Vision] → [K2 Agent] → [K3 Backend] → [K4 Frontend]
                YOLOv8       LangGraph     FastAPI        React
                MediaPipe    + LLM         + Postgres     Dashboard
```

Mỗi khối chỉ tôn trọng contract ở ranh giới của mình, không cần biết nội bộ khối khác. Năm contract (`Raw_Event_Payload`, `Validated_Event_JSON`, `WS_Alert_Message`, `GET /api/events`, `Resolve_Request/Response`) đặc tả đầy đủ trong `docs/database/architecture.md` — **đọc file đó trước khi sửa bất kỳ interface nào giữa các khối**.

## Quyết định kỹ thuật đã chốt

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| ORM | SQLAlchemy 2.0 + Alembic | Drizzle chỉ chạy TypeScript, không tương thích FastAPI |
| Message Queue | Redis Streams | Consumer group + replay khi Agent restart, nhẹ hơn Kafka |
| Object Storage | MinIO (S3-compatible, self-host) | Đúng ràng buộc dữ liệu không ra ngoài local server |
| Vector DB | pgvector trong Postgres | Một DB duy nhất, join trực tiếp với `incidents` |
| Snapshot trong DB | Chỉ lưu `frame_url` (TEXT) | Không lưu binary trong Postgres |

**Về "VLM":** không phải model riêng — là cách gọi LLM multimodal khi có ảnh kèm theo. Node `Assess` gọi có ảnh, node `Plan` gọi text-only. Đừng deploy hai model riêng biệt.

## Database

6 bảng, schema đầy đủ ở `docs/database/schema.sql`:

- `users` — bảo vệ (`guard`) / quản lý (`manager`)
- `cameras` — RTSP url, `restricted_zone` (polygon JSONB), `restricted_hours`
- `zone_rules` — ngưỡng theo từng camera × `event_type` (timeout đồ bỏ quên, thời gian nằm bất động, số frame xác nhận xâm nhập)
- `events` — dữ liệu thô từ Vision, `detection_data` JSONB đổi cấu trúc theo `event_type`
- `incidents` — đánh giá của Agent (`severity`, `llm_reasoning`, `status`)
- `incident_context_embeddings` — pgvector 1536 chiều, dùng cho vòng học ngược

**Ba `event_type` cố định:** `intrusion` | `fall` | `abandoned_object`. Enum này lặp lại ở `events`, `zone_rules`, và metadata embedding — đổi một chỗ phải đổi cả ba.

**Vòng học ngược:** khi bảo vệ bấm `dismiss`, Backend ghi embedding với metadata `dismissed_by_human: true`. Lần sau node `Assess` truy vấn Vector DB sẽ thấy case tương tự từng bị bác — đây là cơ chế giảm báo động giả theo thời gian, là metric đánh giá chính của dự án.

### Hai môi trường DB

| Môi trường | Kết nối | Dùng khi |
|---|---|---|
| Local Docker | `docker compose up -d db` → `localhost:5432` | Dev, test cá nhân |
| Supabase Cloud | `DATABASE_URL` trong `.env` | Cả team dùng chung |

Supabase phải dùng **Session Pooler** (IPv4), không dùng Direct connection — container Docker trên Windows không có route IPv6 nên Direct connection sẽ fail.

Sửa schema: sửa `docs/database/schema.sql` → commit → mới áp lên Supabase, để cả team đồng bộ. Không sửa trực tiếp qua Supabase Dashboard.

## Lệnh thường dùng

```bash
make run          # uvicorn src.main:app --reload --port 8000
make test         # pytest tests/ -v
make lint         # ruff check src/ tests/
make format       # ruff format src/ tests/
make typecheck    # mypy src/
make check        # lint + format + test

docker compose up -d db              # Postgres + pgvector local
docker compose logs db               # xem log
docker compose down                  # dừng, giữ data
docker compose down -v               # dừng + XOÁ data
docker exec -it p-091-db-1 psql -U postgres -d p091
```

## Quy ước code

- Python 3.11, ruff (line-length 120, double quotes), mypy
- Cấu trúc `src/`: `agents/` (LangGraph: `graph.py`, `state.py`, `nodes/`, `tools/`), `api/routes.py`, `models/schemas.py` (Pydantic), `services/`, `config.py` (Pydantic Settings)
- Mọi payload giữa các khối có `schema_version`; timestamp ISO 8601 UTC; ID dùng UUID v4
- Bên nhận bỏ qua trường lạ thay vì báo lỗi — cho phép thêm trường mới không phá bên nhận

## Tài liệu

| File | Nội dung |
|---|---|
| `docs/database/architecture.md` | 4 khối + 5 contract dữ liệu chi tiết — **file quan trọng nhất** |
| `docs/database/schema.sql` | Schema Postgres, chạy tự động khi container db khởi tạo |
| `docs/database/pipeline.md` | Pipeline 5 giai đoạn Ingestion → Serving |
| `docs/database/diagrams.md` | 3 sơ đồ mermaid: kiến trúc, data flow, sequence HITL |
| `docs/gate1/prd.md` | PRD gốc — nguồn của mọi ràng buộc |
| `docs/reports/2026-08-01.md` | Phân tích chống báo động giả theo từng nhãn |

## Lưu ý khi làm việc

- `.env` chứa secret (DATABASE_URL có password, API keys) — đã trong `.gitignore`, không commit
- Template AI20K tự log AI usage vào `.ai-log/session.jsonl` và submit khi `git push` — không xoá thư mục `.claude/`, `.cursor/`, `scripts/log_hook.py`
- `docs/architecture_diagram.md` ở gốc `docs/` là file template mẫu chưa dùng, không phải kiến trúc thật của dự án
