"""Позы людей MediaPipe Pose Landmarker: 33 точки [x, y, видимость] в координатах кадра по боксу person.

В сервисе — PoseEstimator.from_env(): POSE=off|lite|full|heavy (модель POSE_MODEL_DIR/pose_landmarker_<вид>.task,
в образе cv — /models), FrameDetector кладёт точки в attributes.pose боксов person, L2 читает их в decision.posture.
Нет mediapipe или модели — предупреждение и работа без поз. Hand Landmarker не берём: кисть в кадре ~20 px.

Проба на кэше L1 (контрольные листы со скелетом, доля найденных поз, мс на кроп):
  docker compose -f tools/mediapipe/compose.yaml run --rm -e CACHE_TAG=native mp python -m cv.pose --videos X-1 X-34
Пишет data/features/<sha>/pose_<вид>[@<метка>].json и data/runs/pose/<видео>_<вид>.jpg; итоги — summary_<вид>.json.
"""

import argparse
from contextlib import closing
import csv
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from cv.detectors import DATA_DIR, iter_frames

log = logging.getLogger(__name__)
KINDS = ("lite", "full", "heavy")
MARGIN = 0.15  # поле вокруг бокса: руки и локти часто выходят за рамку детектора
CROP_H = 384  # кроп растягивается до этой высоты — модель рассчитана на крупного человека
VISIBLE = 0.5
# Точки BlazePose: 0 нос, 7/8 уши, 11/12 плечи, 13/14 локти, 15/16 запястья, 23/24 бёдра
WRISTS, SHOULDERS = (15, 16), (11, 12)
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 7), (0, 8)]
TILE = (120, 200)  # ширина, высота клетки контрольного листа
SHEET_COLS, SHEET_MAX = 10, 60


class PoseEstimator:
    """Pose Landmarker в режиме IMAGE на кропе человека; точки возвращаются в координатах кадра."""

    def __init__(self, model: Path, kind: str | None = None) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self.model, self.kind = Path(model), kind or Path(model).stem.rsplit("_", 1)[-1]
        opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                            running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def close(self) -> None:
        """Закрываем native dispatcher до разрушения Python-модулей, в том числе в пакетном инференсе."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    @classmethod
    def from_env(cls) -> "PoseEstimator | None":
        kind = os.environ.get("POSE", "off")
        if kind in ("", "off", "0"):
            return None
        if kind not in KINDS:
            raise ValueError(f"POSE={kind}: off, {', '.join(KINDS)}")
        model = Path(os.environ.get("POSE_MODEL_DIR", "/models")) / f"pose_landmarker_{kind}.task"
        if not model.exists():
            log.warning("нет модели позы %s — работаем без поз", model)
            return None
        try:
            return cls(model, kind)
        except ImportError as e:  # образ без mediapipe — без поз, но не падаем
            log.warning("mediapipe недоступен (%s) — работаем без поз", e)
            return None

    def landmarks(self, frame: np.ndarray, bbox: list[float]) -> list[list[float]] | None:
        crop, origin = crop_box(frame, bbox)
        return self.detect(crop, origin)

    def detect(self, crop: np.ndarray, origin: tuple[int, int, float]) -> list[list[float]] | None:
        import mediapipe as mp

        res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1])))
        if not res.pose_landmarks:
            return None
        x0, y0, scale = origin
        ch, cw = crop.shape[:2]
        return [[round(x0 + p.x * cw / scale, 1), round(y0 + p.y * ch / scale, 1), round(p.visibility or 0.0, 2)]
                for p in res.pose_landmarks[0]]


def crop_box(frame: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, tuple[int, int, float]]:
    """Кроп с полем MARGIN, растянутый до CROP_H; (кроп, (x0, y0, масштаб)) — чтобы вернуть точки в кадр."""
    h, w = frame.shape[:2]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0, y0 = max(0, int(bbox[0] - MARGIN * bw)), max(0, int(bbox[1] - MARGIN * bh))
    x1, y1 = min(w, int(bbox[2] + MARGIN * bw)), min(h, int(bbox[3] + MARGIN * bh))
    crop = frame[y0:y1, x0:x1]
    scale = CROP_H / max(1, crop.shape[0])
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), (x0, y0, scale)


# --- проба: контрольные листы и статистика по кэшу L1 ---

MODEL = os.environ.get("POSE_MODEL", "/models/pose_landmarker_heavy.task")
CACHE_TAG = os.environ.get("CACHE_TAG") or None  # кэш L1: l1_yolo[@<метка>].json; позы считаются на его частоте
MIN_CONF = 0.5  # бокс person из L1


def draw(tile: np.ndarray, pts: list[list[float]] | None, origin: tuple[int, int, float], label: str) -> np.ndarray:
    """Скелет на кропе: зелёные кости и точки при видимости ≥ VISIBLE, красные запястья — заметнее."""
    x0, y0, scale = origin
    sx, sy = TILE[0] / tile.shape[1], TILE[1] / tile.shape[0]
    out = cv2.resize(tile, TILE)
    if pts:
        xy = [((p[0] - x0) * scale * sx, (p[1] - y0) * scale * sy, p[2]) for p in pts]
        for a, b in BONES:
            if xy[a][2] >= VISIBLE and xy[b][2] >= VISIBLE:
                cv2.line(out, (int(xy[a][0]), int(xy[a][1])), (int(xy[b][0]), int(xy[b][1])), (0, 220, 0), 2)
        for i in (*SHOULDERS, 13, 14, *WRISTS, 0):
            if xy[i][2] >= VISIBLE:
                cv2.circle(out, (int(xy[i][0]), int(xy[i][1])), 4 if i in WRISTS else 3,
                           (0, 0, 255) if i in WRISTS else (0, 220, 0), -1)
    cv2.rectangle(out, (0, 0), (TILE[0] - 1, 16), (0, 0, 0), -1)
    cv2.putText(out, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255) if pts else (0, 0, 255), 1)
    return out


def sheet(tiles: list[np.ndarray]) -> np.ndarray:
    rows = [tiles[i:i + SHEET_COLS] for i in range(0, len(tiles), SHEET_COLS)]
    blank = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    return np.vstack([np.hstack(r + [blank] * (SHEET_COLS - len(r))) for r in rows])


def run_video(model: PoseEstimator, name: str, sha: str) -> dict:
    cache = json.loads((DATA_DIR / "features" / sha / f"l1_yolo{'@' + CACHE_TAG if CACHE_TAG else ''}.json").read_text())
    by_t: dict[float, list[dict]] = {}
    for b in cache["boxes"]:
        if b["cls"] == "person" and (b.get("conf") or 0) >= MIN_CONF:
            by_t.setdefault(round(b["t"], 3), []).append(b)
    items, tiles, took = [], [], []
    for t, frame in iter_frames(DATA_DIR / "videos" / f"{name}.mov", cache["fps"]):
        for b in by_t.get(round(t, 3), []):
            crop, origin = crop_box(frame, b["bbox"])
            started = time.time()
            pts = model.detect(crop, origin)
            took.append(time.time() - started)
            items.append({"t": round(t, 3), "bbox": b["bbox"], "conf": b["conf"], "landmarks": pts})
            tiles.append(draw(crop, pts, origin, f"{t:.1f}s {'ok' if pts else 'NO'}"))
    out = DATA_DIR / "features" / sha / f"pose_{model.kind}{'@' + CACHE_TAG if CACHE_TAG else ''}.json"
    out.write_text(json.dumps({"video": name, "model": model.model.name, "fps": cache["fps"], "poses": items}))
    if tiles:
        pick = np.linspace(0, len(tiles) - 1, min(SHEET_MAX, len(tiles))).round().astype(int)
        (DATA_DIR / "runs" / "pose").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DATA_DIR / "runs" / "pose" / f"{name}_{model.kind}.jpg"), sheet([tiles[i] for i in pick]))
    found = [x for x in items if x["landmarks"]]
    wrists = [x for x in found if all(x["landmarks"][i][2] >= VISIBLE for i in WRISTS)]
    shoulders = [x for x in found if all(x["landmarks"][i][2] >= VISIBLE for i in SHOULDERS)]
    return {"boxes": len(items), "found": round(len(found) / max(1, len(items)), 3),
            "both_wrists": round(len(wrists) / max(1, len(items)), 3),
            "both_shoulders": round(len(shoulders) / max(1, len(items)), 3),
            "ms_per_crop": round(1000 * sum(took) / max(1, len(took)), 1),
            "box_h": round(float(np.median([x["bbox"][3] - x["bbox"][1] for x in items])), 1) if items else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    args = ap.parse_args()
    with (DATA_DIR / "labels.csv").open() as f:
        sha = {r["video"]: r["sha256"] for r in csv.DictReader(f) if not r["duplicate_of"]}
    with closing(PoseEstimator(Path(MODEL))) as model:
        summary = {}
        for name in args.videos:
            summary[name] = run_video(model, name, sha[name])
            print(name, summary[name], flush=True)
        path = DATA_DIR / "runs" / "pose" / f"summary_{model.kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"→ {path}")


if __name__ == "__main__":
    main()
