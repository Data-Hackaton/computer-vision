"""Обучаемый детектор момента нарушения — аналог DDM у NVIDIA (модель учится на размеченных таймкодах).

Окно WINDOW_SEC с шагом STEP_SEC → признаки из сигналов сцены (decision.scene) и счётчиков классов L1 → вероятность,
что в окне нарушение. Метки: окно, центр которого ближе POS_SEC к размеченной секунде нарушения (labels.csv), —
положительное; окна чистых видео и окна видео с нарушением дальше NEG_SEC от метки — отрицательные.
Оценка — leave-one-video-out: модель для видео обучена без него.
  python -m decision.learned --detector yolo            # оценка LOO
  python -m decision.learned --detector yolo --fit      # обучить на всех и сохранить data/runs/sop/<детектор>.joblib
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from cv.detectors import DATA_DIR
from cv.infer import cache_path
from decision.scene import SIGNALS, assign_roles, frames_of, scene_states, track_people
from decision.steps import HOLD_SEC, MIN_LENGTH_SEC, SMOOTH_SEC, debounce, smooth

WINDOW_SEC, STEP_SEC = 4.0, 1.0
POS_SEC, NEG_SEC = 2.0, 6.0
CLASSES = ["person", "reflection", "hand", "banknotes", "id_card", "cash_machine", "safe", "door", "phone", "bag",
           "foreign_object", "computer"]


def frame_features(boxes: list[dict], fps: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """По кадрам: сигналы сцены (сглаженные) + число боксов каждого класса + положение кассира относительно места."""
    frames = frames_of(boxes)
    track_people(frames, fps)
    roles, seat = assign_roles(frames, fps)
    states = scene_states(frames, roles, seat)
    cols = {}
    for s in SIGNALS:
        v = smooth([st[s] for st in states], round(SMOOTH_SEC * fps))
        cols[s] = debounce(v, max(1, round(HOLD_SEC.get(s, MIN_LENGTH_SEC) * fps)))
    for c in CLASSES:
        cols[f"n_{c}"] = [sum(b["cls"] == c for b in bs) for _, bs in frames]
    dist, depth, scr = [], [], []
    for _, bs in frames:
        cash = [b["bbox"] for b in bs if b.get("attributes", {}).get("role") == "cashier"]
        if cash and seat:
            b = cash[0]
            fx, fy = (b[0] + b[2]) / 2, b[3]
            dist.append(((fx - seat["x"]) ** 2 + (fy - seat["y"]) ** 2) ** 0.5 / seat["h"])
            depth.append((seat["y"] - fy) / seat["h"])
        else:
            dist.append(2.0)
            depth.append(1.0)
        comp = [b["attributes"].get("bright", 0) for b in bs if b["cls"] == "computer" and "bright" in b.get("attributes", {})]
        scr.append(max(comp) if comp else 0.0)
    cols["cashier_dist"], cols["cashier_depth"], cols["screen_bright"] = dist, depth, scr
    names = list(cols)
    return np.array([t for t, _ in frames]), np.array([cols[n] for n in names], dtype=float).T, names


def windows(ts: np.ndarray, x: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Окна: среднее, максимум и изменение (вторая половина − первая) каждого признака; плюс то же за 4 с до окна."""
    centers, feats = [], []
    if len(ts) == 0:
        return np.array([]), np.zeros((0, 0)), []
    for c in np.arange(ts[0] + WINDOW_SEC / 2, ts[-1] - WINDOW_SEC / 2 + 1e-6, STEP_SEC):
        cur = x[(ts >= c - WINDOW_SEC / 2) & (ts < c + WINDOW_SEC / 2)]
        before = x[(ts >= c - WINDOW_SEC * 1.5) & (ts < c - WINDOW_SEC / 2)]
        if len(cur) < 2:
            continue
        half = len(cur) // 2
        prev = before.mean(0) if len(before) else cur[:1].mean(0)
        feats.append(np.concatenate([cur.mean(0), cur.max(0), cur[half:].mean(0) - cur[:half].mean(0),
                                     cur.mean(0) - prev]))
        centers.append(c)
    fnames = [f"{k}:{n}" for k in ("mean", "max", "delta", "vs_before") for n in names]
    return np.array(centers), np.array(feats), fnames


def dataset(detector: str):
    with (DATA_DIR / "labels.csv").open() as f:
        rows = list(csv.DictReader(f))
    labels_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["verdict"] == "violation":
            labels_t[r["duplicate_of"] or r["video"]].append(float(r["timecode_sec"]))
    data = {}
    for r in rows:
        if r["duplicate_of"]:
            continue
        path = cache_path(r["sha256"], detector)
        if not path.exists():
            continue
        cache = json.loads(path.read_text())
        ts, x, names = frame_features(cache["boxes"], cache["fps"])
        centers, feats, fnames = windows(ts, x, names)
        lt = labels_t.get(r["video"], [])
        y = np.array([1 if any(abs(c - t) <= POS_SEC for t in lt) else (0 if all(abs(c - t) > NEG_SEC for t in lt) else -1)
                      for c in centers])
        data[r["video"]] = {"centers": centers, "x": feats, "y": y, "label": r["verdict"], "label_t": lt}
    return data, fnames


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    data, fnames = dataset(args.detector)
    videos = sorted(data, key=lambda v: int(v[2:]))

    if args.fit:
        import joblib

        x = np.concatenate([data[v]["x"][data[v]["y"] >= 0] for v in videos])
        y = np.concatenate([data[v]["y"][data[v]["y"] >= 0] for v in videos])
        clf = model().fit(x, y)
        out = DATA_DIR / "runs" / "sop" / f"{args.detector}.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": clf, "features": fnames, "threshold": args.threshold}, out)
        print(f"→ {out}: окон {len(y)}, положительных {int(y.sum())}")
        return

    scores = {}
    for v in videos:
        train = [w for w in videos if w != v]
        x = np.concatenate([data[w]["x"][data[w]["y"] >= 0] for w in train])
        y = np.concatenate([data[w]["y"][data[w]["y"] >= 0] for w in train])
        clf = model().fit(x, y)
        scores[v] = clf.predict_proba(data[v]["x"])[:, 1] if len(data[v]["x"]) else np.array([])
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        ok = hit = pos = 0
        for v in videos:
            s, c = scores[v], data[v]["centers"]
            pred = "violation" if len(s) and s.max() >= thr else "no_violation"
            ok += pred == data[v]["label"]
            if data[v]["label"] == "violation":
                pos += 1
                hit += bool(len(s)) and any(abs(c[s.argmax()] - t) <= 3 for t in data[v]["label_t"])
        print(f"порог {thr:.1f}: точность вердикта {ok / len(videos):.2f} ({ok}/{len(videos)}), "
              f"пик окна у метки ±3 с: {hit}/{pos}")
    thr = args.threshold
    for v in videos:
        s, c = scores[v], data[v]["centers"]
        top = f"{s.max():.2f} на {c[s.argmax()]:.0f} с" if len(s) else "-"
        print(f"  {v:>5} {data[v]['label']:>12} {data[v]['label_t']} → макс {top}")


if __name__ == "__main__":
    main()
