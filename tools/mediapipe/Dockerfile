# Проба MediaPipe Pose Landmarker (cv.pose) — отдельно от образа cv: там torch и numpy 2.5, пересборка на 11 ГБ
FROM python:3.12-slim-trixie

# libgl1, libglib2.0-0 — для opencv-contrib-python, который ставит mediapipe
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mediapipe==1.0.1 pyyaml==6.0.2

# Модель Pose Landmarker Heavy (Apache-2.0), версия 1 — закреплена
ADD https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task /models/
RUN chmod 644 /models/pose_landmarker_heavy.task

ENV PYTHONPATH=/app/src DATA_DIR=/data POSE_MODEL=/models/pose_landmarker_heavy.task
WORKDIR /app
