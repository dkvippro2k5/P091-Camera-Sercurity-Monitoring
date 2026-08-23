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

# Security: run as non-root user
RUN useradd -m appuser

# Copy installed packages from builder into appuser's own home, not /root. `pip install
# --user` in the builder stage put everything under /root/.local, and /root is mode 700 --
# root's own home is deliberately unreadable by anyone else. Landing straight there was
# never actually runnable as appuser: `docker compose up` failed with
# "PermissionError: /root/.local/bin/uvicorn" on the very first `USER appuser` run, because
# appuser cannot even traverse into /root to reach the copied files, regardless of what
# permissions the files themselves carry. --chown fixes both the target ownership and the
# traversal problem by not putting it under /root in the first place.
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

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
