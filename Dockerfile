# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: Production ----
FROM python:3.11-slim

WORKDIR /app

# opencv-python links against libGL and glib, neither of which ships in -slim, so
# `import cv2` aborts at startup without these. They are kept (rather than switching the
# requirement to opencv-python-headless) so scripts/demo_rtsp_simulation.py --show still
# works from a local checkout using the same requirements.txt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Security: run as non-root user
RUN useradd -m appuser

# Copy application code
COPY . .

# Runtime state is writable; application code remains root-owned and read-only to appuser.
RUN mkdir -p /app/data /app/models && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

# Model weights are mounted read-only at runtime by docker-compose. Keeping them outside the
# image avoids a 50+ MB layer and lets operators update a checkpoint without rebuilding.
HEALTHCHECK --start-period=90s --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
