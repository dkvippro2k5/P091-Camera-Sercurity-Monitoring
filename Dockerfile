# CUDA, cuDNN, PyTorch and Python are pinned together for the Tesla T4 deployment.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1

# opencv-python links against libGL and glib, neither of which ships in the base image, so
# `import cv2` aborts at startup without these. They are kept (rather than switching the
# requirement to opencv-python-headless) so scripts/demo_rtsp_simulation.py --show still
# works from a local checkout using the same requirements.txt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch, torchvision; assert torch.__version__.startswith('2.7.1'); assert torchvision.__version__.startswith('0.22.1')"

# Security: run as non-root user
RUN useradd -m appuser

# Copy application code
COPY . .

# Runtime state and Ultralytics' settings directory are writable; application code remains
# root-owned and read-only.
RUN mkdir -p /app/data /app/models /tmp/Ultralytics \
    && chown -R appuser:appuser /app/data /tmp/Ultralytics

USER appuser

EXPOSE 8000

# Model weights are mounted read-only at runtime by docker-compose. Keeping them outside the
# image avoids a 50+ MB layer and lets operators update a checkpoint without rebuilding.
HEALTHCHECK --start-period=90s --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
