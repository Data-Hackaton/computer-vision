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

# Позы людей MediaPipe (cv.pose, POSE=lite|full|heavy). Отдельными слоями после uv sync и apt выше: смена lock или
# базового apt-слоя пересобирает слой с torch (несколько ГБ) на сервере; перенести в pyproject после хакатона.
# Версии закреплены: mediapipe 1.0.1 совместим с numpy 2.5.4 и opencv 5.0 из lock; libegl1, libgles2 — его нативная
# библиотека требует libEGL.so.1 даже на CPU. Модели Pose Landmarker (Apache-2.0), версия 1, с проверкой sha256.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/*
RUN uv pip install --no-cache mediapipe==1.0.1
ADD https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task /models/
ADD https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task /models/
RUN echo "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a  /models/pose_landmarker_lite.task" | sha256sum -c - \
 && echo "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b  /models/pose_landmarker_heavy.task" | sha256sum -c - \
 && chmod 644 /models/*.task
ENV POSE_MODEL_DIR=/models
