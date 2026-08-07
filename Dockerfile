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

# Create data directory with correct ownership
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Model weights are gitignored, so a fresh clone builds an image without them and the
# detector raises FileNotFoundError at startup rather than pretending to see. Mount or
# COPY src/vision/weights/yolo11n.pt into the image.
HEALTHCHECK --start-period=90s --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
