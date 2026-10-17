"""Прогон детектора L1 по видео датасета → кэш data/features/<sha256>/l1_<детектор>.json.

На кэше отлаживается и оценивается L2+L3 (python -m decision.eval) без GPU. Видео — уникальные из labels.csv.
  python -m cv.infer --detector yolo                  # все видео
  python -m cv.infer --detector rfdetr --videos X-1 X-4
  MODEL_DIR=/data/runs/releases/<тег> python -m cv.infer   # релиз модели → кэш l1_<детектор>@<тег>.json
  DETECTOR_LOW_CONF=0.1 python -m cv.infer --detector yolo --cache-tag lowconf   # опыт → l1_yolo@lowconf.json
"""

import argparse
import csv
import json
import os
import time
from contextlib import closing

from cv.detectors import DATA_DIR, apply_model_dir, get_detector


def cache_path(sha256: str, detector: str, tag: str | None = None):
    """Кэш L1 видео; у релиза модели свой файл, чтобы его оценка не затирала кэш, на котором учат и оценивают."""
    return DATA_DIR / "features" / sha256 / (f"l1_{detector}@{tag}.json" if tag else f"l1_{detector}.json")


def detector_arg(ap: argparse.ArgumentParser, tag: str | None, detector: str | None) -> str:
    """--detector, а для релиза — детектор из его model.json (apply_model_dir уже выставил DETECTOR)."""
    if tag and detector and detector != os.environ["DETECTOR"]:
        ap.error(f"--detector {detector}, а релиз {tag} — {os.environ['DETECTOR']}")
    detector = detector or (os.environ["DETECTOR"] if tag else None)
    if detector not in ("yolo", "rfdetr"):
        ap.error("нужен --detector yolo|rfdetr или MODEL_DIR")
    return detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["yolo", "rfdetr"], help="без MODEL_DIR обязателен")
    ap.add_argument("--videos", nargs="*", help="X-1 X-4 …; по умолчанию все уникальные")
    ap.add_argument("--cache-tag", help="свой файл кэша l1_<детектор>@<метка>.json для опыта (без MODEL_DIR)")
    args = ap.parse_args()
    tag = apply_model_dir()
    detector = detector_arg(ap, tag, args.detector)
    cache_tag = tag or args.cache_tag

    with (DATA_DIR / "labels.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if not r["duplicate_of"]]
    if args.videos:
        rows = [r for r in rows if r["video"] in args.videos]
    with closing(get_detector(detector)) as det:
        total = time.time()
        for r in rows:
            started = time.time()
            boxes = det.detect(DATA_DIR / "videos" / f"{r['video']}.mov")
            out = cache_path(r["sha256"], detector, cache_tag)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"video": r["video"], "detector": detector, "model": tag, "weights": str(det.weights),
                                       "fps": det.last_fps, "conf": det.conf, "low_conf": det.low_conf, "views": det.views,
                                       "boxes": boxes}))
            took = time.time() - started
            print(f"{r['video']:>5}: {float(r['duration_sec']):6.1f} с видео → {len(boxes):5d} боксов за {took:5.1f} с",
                  flush=True)
        print(f"всего {len(rows)} видео за {time.time() - total:.0f} с")


if __name__ == "__main__":
    main()
