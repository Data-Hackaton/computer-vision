"""Прогон детектора L1 по видео датасета → кэш data/features/<sha256>/l1_<детектор>.json.

На кэше отлаживается и оценивается L2+L3 (python -m decision.eval) без GPU. Видео — уникальные из labels.csv.
  python -m cv.infer --detector yolo                  # все видео
  python -m cv.infer --detector rfdetr --videos X-1 X-4
"""

import argparse
import csv
import json
import time

from cv.detectors import DATA_DIR, get_detector


def cache_path(sha256: str, detector: str):
    return DATA_DIR / "features" / sha256 / f"l1_{detector}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--videos", nargs="*", help="X-1 X-4 …; по умолчанию все уникальные")
    args = ap.parse_args()

    with (DATA_DIR / "labels.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if not r["duplicate_of"]]
    if args.videos:
        rows = [r for r in rows if r["video"] in args.videos]
    det = get_detector(args.detector)
    total = time.time()
    for r in rows:
        started = time.time()
        boxes = det.detect(DATA_DIR / "videos" / f"{r['video']}.mov")
        out = cache_path(r["sha256"], args.detector)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"video": r["video"], "detector": args.detector, "weights": str(det.weights),
                                   "fps": det.fps, "conf": det.conf, "boxes": boxes}))
        took = time.time() - started
        print(f"{r['video']:>5}: {float(r['duration_sec']):6.1f} с видео → {len(boxes):5d} боксов за {took:5.1f} с",
              flush=True)
    print(f"всего {len(rows)} видео за {time.time() - total:.0f} с")


if __name__ == "__main__":
    main()
