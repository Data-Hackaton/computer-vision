FROM python:3.12-slim-trixie

# libgl1, libglib2.0-0 — для opencv из ultralytics; ffmpeg — декодирование видео
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    DATA_DIR=/data \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    YOLO_CONFIG_DIR=/cache/ultralytics \
    MPLCONFIGDIR=/cache/matplotlib

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
