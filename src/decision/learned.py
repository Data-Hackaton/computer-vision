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


def step_class(a: dict) -> str:
    """Класс шага для обучения: действие + участник (вход/уход) или предмет/место (остальное)."""
    if a["action"] in ("person_enter", "person_leave"):
        return f"{a['action']}:{a.get('actor')}"
    if a["action"] in ("put", "take", "give"):
        return f"{a['action']}:{a.get('object')}>{a.get('target')}"
    return a["action"]


def steps_dataset(detector: str, step_sec: float = 0.5, near_sec: float = 1.0):
    """Окна по видео с журналом (actions.yaml + violations.yaml): метка — класс шага ближе near_sec, иначе «none»."""
    import yaml

    from cv.detectors import REFERENCE_DIR

    journals = {**yaml.safe_load((REFERENCE_DIR / "actions.yaml").read_text()),
                **yaml.safe_load((REFERENCE_DIR / "violations.yaml").read_text())}
    with (DATA_DIR / "labels.csv").open() as f:
        sha = {r["video"]: r["sha256"] for r in csv.DictReader(f)}
    global STEP_SEC
    keep_step, STEP_SEC = STEP_SEC, step_sec
    data = {}
    for name, entry in journals.items():
        path = cache_path(sha[name], detector)
        if not path.exists():
            continue
        cache = json.loads(path.read_text())
        ts, x, names = frame_features(cache["boxes"], cache["fps"])
        centers, feats, fnames = windows(ts, x, names)
        acts = [a for i in entry["instances"] for a in i["actions"]]
        y = []
        for c in centers:
            near = [a for a in acts if abs(a["t"] - c) <= near_sec]
            y.append(step_class(min(near, key=lambda a: abs(a["t"] - c))) if near else "none")
        data[name] = {"centers": centers, "x": feats, "y": np.array(y),
                      "ref": [step_class(a) for a in sorted(acts, key=lambda a: a["t"])]}
    STEP_SEC = keep_step
    return data, fnames


def peaks(centers: np.ndarray, proba: np.ndarray, classes: list[str], thr: float, nms_sec: float = 2.0) -> list[tuple]:
    """Пики вероятности по классам (кроме «none») с подавлением соседних → [(t, класс)] по времени."""
    out = []
    for k, cls in enumerate(classes):
        if cls == "none":
            continue
        p = proba[:, k]
        taken = []
        for i in np.argsort(-p):
            if p[i] < thr:
                break
            if all(abs(centers[i] - t) > nms_sec for t in taken):
                taken.append(centers[i])
        out += [(t, cls) for t in taken]
    return sorted(out)


# Шаги, которых правила L2 не видят, а обученный классификатор находит (LOO на 24 видео с журналом, 17.10):
# сейф, удостоверение, передача денег клиентом, раскладка по стопкам
HYBRID_CLASSES = {"safe_open", "safe_close", "give:id_card>cashier", "inspect", "give:banknotes>cashier",
                  "sort_banknotes"}


HYBRID_THRESHOLD = 0.5  # порог пика: на LOO гибрид 0,23 против 0,19 у правил


def class_action(cls: str) -> dict:
    """Класс шага → действие журнала."""
    if cls == "give:id_card>cashier":
        return {"action": "give", "actor": "client", "object": "id_card", "target": "cashier"}
    if cls == "give:banknotes>cashier":
        return {"action": "give", "actor": "client", "object": "banknotes", "target": "cashier"}
    if cls == "inspect":
        return {"action": "inspect", "actor": "cashier", "object": "id_card"}
    return {"action": cls, "actor": "cashier"}


def steps_model_path(detector: str):
    return DATA_DIR / "runs" / "sop" / f"steps_{detector}.joblib"


def learned_actions(boxes: list[dict], fps: float, model_path) -> list[dict]:
    """Действия по шагам HYBRID_CLASSES от обученного классификатора (STEP_CLASSIFIER=hybrid)."""
    import joblib

    saved = joblib.load(model_path)
    ts, x, names = frame_features(boxes, fps)
    global STEP_SEC
    keep_step, STEP_SEC = STEP_SEC, saved.get("step_sec", 0.5)
    try:
        centers, feats, _ = windows(ts, x, names)
    finally:
        STEP_SEC = keep_step
    if not len(feats):
        return []
    proba = saved["model"].predict_proba(feats)
    out = []
    for t, cls in peaks(centers, proba, list(saved["model"].classes_), saved["threshold"]):
        if cls in saved["classes"]:
            out.append({"t": round(float(t), 2), **class_action(cls), "source": "learned"})
    return out


def steps_fit(detector: str) -> None:
    import joblib

    data, _ = steps_dataset(detector)
    clf = model().fit(np.concatenate([d["x"] for d in data.values()]), np.concatenate([d["y"] for d in data.values()]))
    out = steps_model_path(detector)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": sorted(HYBRID_CLASSES), "threshold": HYBRID_THRESHOLD, "step_sec": 0.5}, out)
    print(f"→ {out}: видео {len(data)}, окон {sum(len(d['y']) for d in data.values())}")


def steps_main(detector: str) -> None:
    """LOO: seq_accuracy обученного классификатора шагов против правил L2 на тех же классах."""
    from decision.sop import seq_accuracy
    from decision.steps import analyze

    data, _ = steps_dataset(detector)
    videos = sorted(data, key=lambda v: int(v[2:]))
    learned_cls = sorted({c for v in videos for c in data[v]["y"]} - {"none"})
    with (DATA_DIR / "labels.csv").open() as f:
        sha = {r["video"]: r["sha256"] for r in csv.DictReader(f)}
    probs = {}
    for v in videos:
        train = [w for w in videos if w != v]
        clf = model().fit(np.concatenate([data[w]["x"] for w in train]), np.concatenate([data[w]["y"] for w in train]))
        probs[v] = (clf.predict_proba(data[v]["x"]), list(clf.classes_))
    rules_seq, rules_tc = {}, {}
    for v in videos:
        cache = json.loads(cache_path(sha[v], detector).read_text())
        acts = analyze(cache["boxes"], cache["fps"])["actions"]
        rules_seq[v] = [step_class(a) for a in acts]
        rules_tc[v] = [(a["t"], step_class(a)) for a in acts]
    for thr in (0.3, 0.4, 0.5, 0.6):
        acc_h = []
        for v in videos:
            proba, classes = probs[v]
            ref = [c for c in data[v]["ref"] if c in learned_cls]
            rules_t = [(a_t, c) for a_t, c in rules_tc[v] if c in learned_cls]
            hyb = sorted(rules_t + [(t, c) for t, c in peaks(data[v]["centers"], proba, classes, thr) if c in HYBRID_CLASSES])
            acc_h.append(seq_accuracy([c for _, c in hyb], ref))
        print(f"порог {thr:.1f}: гибрид (правила + модель по {len(HYBRID_CLASSES)} шагам) {np.mean(acc_h):.2f}")
    for thr in (0.3, 0.4, 0.5, 0.6):
        acc_l, acc_r = [], []
        per_cls = defaultdict(lambda: [0, 0, 0])  # верно найдено, предсказано, в эталоне
        for v in videos:
            proba, classes = probs[v]
            pred = [c for _, c in peaks(data[v]["centers"], proba, classes, thr)]
            ref = [c for c in data[v]["ref"] if c in learned_cls]
            acc_l.append(seq_accuracy(pred, ref))
            acc_r.append(seq_accuracy([c for c in rules_seq[v] if c in learned_cls], ref))
            for c in learned_cls:
                n_p, n_r = pred.count(c), ref.count(c)
                per_cls[c][0] += min(n_p, n_r); per_cls[c][1] += n_p; per_cls[c][2] += n_r
        print(f"порог {thr:.1f}: seq_accuracy обученный {np.mean(acc_l):.2f}, правила {np.mean(acc_r):.2f}")
        if thr == 0.5:
            for c, (hit, n_p, n_r) in sorted(per_cls.items(), key=lambda kv: -kv[1][2]):
                print(f"    {c:<40} эталон {n_r:3d}  предсказано {n_p:3d}  совпало {hit:3d}")


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--target", choices=["violations", "steps"], default="violations",
                    help="violations — момент нарушения по labels.csv; steps — шаги по эталонному журналу (аналог DDM)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if args.target == "steps":
        steps_fit(args.detector) if args.fit else steps_main(args.detector)
        return
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
