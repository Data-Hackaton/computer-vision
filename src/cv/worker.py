"""cv-worker: активность run_cv в очереди cv-gpu (Temporal), одна задача за раз — GPU одна.

run_cv(task) — task от активности prepare бэкенда: {job_id, video_id, source_key, sha256, duration, fps, ...}.
Пишет в S3 jobs/<job_id>/events.json, boxes.json, evidence/<n>.jpg и возвращает ключи для publish.
Прогресс по этапам — в Valkey (job:<id>), как ждёт бэкенд.
"""

import asyncio
import json
import os
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import redis
from botocore.client import Config
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from cv.detectors import apply_model_dir, get_detector
from cv.pipeline import build_video_events, describe, evidence_times, frame_with_boxes, load_spec, why_by_title

S3_BUCKET = os.environ.get("S3_BUCKET", "datathon")
CV_QUEUE = "cv-gpu"

_s3 = boto3.client(
    "s3", endpoint_url=os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333"),
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "datathon"),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "change-me-please"),
    region_name="us-east-1", config=Config(s3={"addressing_style": "path"}),
)
_valkey = redis.Redis.from_url(os.environ.get("VALKEY_URL", "redis://valkey:6379/0"), decode_responses=True)
_detector = None


def stage(job_id: str, name: str, progress: float) -> None:
    try:
        _valkey.set(f"job:{job_id}", json.dumps({"stage": name, "progress": progress}), ex=24 * 3600)
    except redis.RedisError:
        pass


def put_json(key: str, data) -> None:
    _s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data, ensure_ascii=False).encode(),
                   ContentType="application/json")


@activity.defn(name="run_cv")
def run_cv(task: dict) -> dict:
    global _detector
    job_id = task["job_id"]
    if _detector is None:
        _detector = get_detector()
    spec = load_spec()
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "video"
        stage(job_id, "downloading", 0.05)
        _s3.download_file(S3_BUCKET, task["source_key"], str(video))

        stage(job_id, "detecting", 0.2)
        det = _detector.run(video, task["sha256"])

        stage(job_id, "events", 0.6)
        events = build_video_events(det, spec, task["video_id"])
        why = why_by_title(spec)

        stage(job_id, "evidence", 0.75)
        n = 0
        for e in events:
            e["evidence"] = []
            if e["status"] == "expected":
                continue
            if e["event"] != "sop_step":  # у SOP-чекера сообщение готово: «Пропущен шаг процедуры в цикле …»
                e["message"] = describe(e, why)
            if e["status"] == "violation":  # кадр на таймкоде и соседние — до 3 на нарушение
                for t in evidence_times(e["t_start"], task.get("duration")):
                    key = f"jobs/{job_id}/evidence/{n}.jpg"
                    _s3.put_object(Bucket=S3_BUCKET, Key=key, Body=frame_with_boxes(video, t, det.boxes),
                                   ContentType="image/jpeg")
                    e["evidence"].append({"t": t, "key": key})
                    n += 1

    put_json(f"jobs/{job_id}/events.json", events)
    put_json(f"jobs/{job_id}/boxes.json", det.boxes)
    verdict = "violation" if any(e["status"] == "violation" for e in events) else "no_violation"
    activity.logger.info("job %s: %s, событий %d, боксов %d (%s)", job_id, verdict, len(events), len(det.boxes), det.note)
    return {"job_id": job_id, "verdict": verdict, "detector": det.detector, "model": os.environ.get("MODEL_TAG"),
            "events_key": f"jobs/{job_id}/events.json", "boxes_key": f"jobs/{job_id}/boxes.json"}


async def main() -> None:
    global _detector
    # несколько воркеров на CPU делят ядра: по CV_THREADS потоков torch на каждый (по умолчанию 6)
    if threads := os.environ.get("CV_THREADS"):
        import torch

        torch.set_num_threads(int(threads))
    # модель — при старте, а не на первой задаче: первая задача жюри не ждёт загрузки весов
    started = time.time()
    try:
        tag = apply_model_dir()
    except FileNotFoundError as e:  # MODEL_DIR=…/current до первого релиза: анализ важнее, веса по умолчанию
        tag = None
        print(f"cv-worker: релиза модели нет ({e.filename}) — веса по умолчанию", flush=True)
    _detector = get_detector()
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "temporal:7233"),
                                      namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"))
        # тег модели в identity: виден в GET /api/v1/queue, по нему infra/model.sh проверяет выкатку
        identity = f"{os.getpid()}@{socket.gethostname()} {_detector.name}:{tag or 'default'}"
        worker = Worker(client, task_queue=CV_QUEUE, activities=[run_cv], activity_executor=ThreadPoolExecutor(1),
                        max_concurrent_activities=1, identity=identity)
        print(f"cv-worker: очередь {CV_QUEUE}, детектор {_detector.name}, модель {tag or 'по умолчанию'}, "
              f"готов за {time.time() - started:.0f} с", flush=True)
        await worker.run()

    finally:
        _detector.close()


if __name__ == "__main__":
    asyncio.run(main())
