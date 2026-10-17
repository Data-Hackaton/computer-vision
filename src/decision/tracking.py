"""Трекинг людей трекерами Ultralytics: ByteTrack и BoT-SORT (TRACKER=bytetrack|botsort, см. scene.track_people).

Отличия от жадного трекера scene.py: фильтр Калмана (рамка предсказывается на пропущенных кадрах), оптимальное
сопоставление вместо жадного, новый трек подтверждается вторым кадром (однокадровые вспышки трека не получают),
слабые боксы (attributes.low, DETECTOR_LOW_CONF в L1) только продлевают треки — второй проход ByteTrack.
BoT-SORT — без компенсации движения камеры (камера неподвижна) и без ReID (нужны кадры, в кэше L1 их нет).
TRACKER_ARGS — JSON поверх cfg/trackers/<трекер>.yaml, например '{"match_thresh": 0.7, "track_buffer": 20}'.
Трекеры Ultralytics — AGPL-3.0, как и YOLO.
"""

import json
import os
import sys
import types

import numpy as np

from cv.geometry import iou, overlap_share

TRACKERS = ("bytetrack", "botsort")
ORIG_SHAPE = (1080, 1920)  # Boxes нужен размер кадра только для нормированных координат, трекер их не читает


def _install_lap() -> None:
    """lap (lapjv) нет в образе — трекеры Ultralytics импортируют его в trackers/utils/matching.py. Замена на scipy
    с той же семантикой cost_limit: расширенная матрица, где «без пары» стоит cost_limit / 2 с каждой стороны, так что
    пара дороже cost_limit никогда не выгодна. Если lap поставят в образ, возьмётся он."""
    try:
        import lap  # noqa: F401

        return
    except ImportError:
        pass
    from scipy.optimize import linear_sum_assignment

    def lapjv(cost, extend_cost=True, cost_limit=np.inf):
        cost = np.asarray(cost, dtype=float)
        n, m = cost.shape
        limit = cost_limit if np.isfinite(cost_limit) else (np.nanmax(cost[np.isfinite(cost)], initial=0.0) + 1) * 2
        ext = np.zeros((n + m, n + m))
        ext[:n, :m] = np.where(np.isfinite(cost), np.minimum(cost, limit * 2), limit * 2)
        ext[:n, m:] = limit / 2
        ext[n:, :m] = limit / 2
        rows, cols = linear_sum_assignment(ext)
        x, y = np.full(n, -1, dtype=int), np.full(m, -1, dtype=int)
        for r, c in zip(rows, cols):
            if r < n and c < m and ext[r, c] <= limit:
                x[r], y[c] = c, r
        return float(sum(cost[r, x[r]] for r in range(n) if x[r] >= 0)), x, y

    sys.modules["lap"] = types.SimpleNamespace(__version__="scipy", lapjv=lapjv)


def tracker_args(kind: str, fps: float, lost_sec: float) -> types.SimpleNamespace:
    """cfg/trackers/<kind>.yaml + наши значения + TRACKER_ARGS. Буфер потерянного трека в Ultralytics 8.4 — в кадрах
    (max_frames_lost = track_buffer), поэтому lost_sec × fps. Первый проход — боксы от DETECTOR_CONF, второй — слабые."""
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    args = dict(YAML.load(check_yaml(f"{kind}.yaml")))
    conf = float(os.environ.get("DETECTOR_CONF", "0.35"))
    args.update(track_buffer=max(1, round(lost_sec * fps)), track_high_thresh=conf, new_track_thresh=conf)
    if kind == "botsort":
        args.update(gmc_method="none", with_reid=False)
    args.update(json.loads(os.environ.get("TRACKER_ARGS") or "{}"))
    return types.SimpleNamespace(**args)


def candidates(boxes: list[dict]) -> list[dict]:
    """person и reflection кадра без вложенных и дублей: сначала обычные, потом слабые; person раньше reflection."""
    out = []
    order = sorted((b for b in boxes if b["cls"] in ("person", "reflection")),
                   key=lambda b: (bool(b.get("attributes", {}).get("low")), b["cls"] != "person", -(b.get("conf") or 0)))
    for b in order:
        if not any(overlap_share(b["bbox"], k["bbox"]) > 0.7 or overlap_share(k["bbox"], b["bbox"]) > 0.7
                   or iou(b["bbox"], k["bbox"]) > 0.5 for k in out):
            out.append(b)
    return out


def track_people_ultra(frames: list[tuple[float, list[dict]]], fps: float, kind: str, lost_sec: float = 3.0,
                       low: list[dict] | None = None) -> None:
    """track_id в боксы person/reflection (и в слабые боксы low, продлившие трек). Шаги сетки t × fps без боксов
    подаются пустыми: так потерянный трек стареет по времени, а не по числу кадров с детекциями. Трекер видит и
    отражения — иначе фильтр Калмана кассира, вставшего перед стеклом (в L1 он похож на отражение), уплывает и id
    перескакивает на клиента. Но в боксах reflection, как в жадном трекере, только продолжает трек: после бокса
    person этого трека и при IoU > 0,6 с его предыдущей рамкой не старше секунды — настоящее отражение клиентом не
    становится."""
    _install_lap()
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers import BOTSORT, BYTETracker

    tracker = {"bytetrack": BYTETracker, "botsort": BOTSORT}[kind](tracker_args(kind, fps, lost_sec))
    by_step: dict[int, list[dict]] = {}
    for t, boxes in frames:
        by_step.setdefault(round(t * fps), []).extend(boxes)
    for b in low or []:
        if b["cls"] == "person":
            by_step.setdefault(round(b["t"] * fps), []).append(b)
    if not by_step:
        return
    cands: dict[int, list[dict]] = {}
    for step in range(min(by_step), max(by_step) + 1):
        cands[step] = candidates(by_step.get(step, []))
        rows = [[*b["bbox"], b.get("conf") or 0.0, 0 if b["cls"] == "person" else 1] for b in cands[step]]
        out = tracker.update(Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), ORIG_SHAPE))
        for r in out:
            cands[step][int(r[7])]["track_id"] = int(r[4])
    # новый трек подтверждается на втором кадре — первый кадр подтверждённого трека добираем по IoU
    first: dict[int, tuple[int, dict]] = {}
    for step, bs in cands.items():
        for b in bs:
            if "track_id" in b and b["track_id"] not in first:
                first[b["track_id"]] = (step, b)
    for tid, (step, b) in first.items():
        prev = [p for p in cands.get(step - 1, []) if "track_id" not in p and not p.get("attributes", {}).get("low")]
        best = max(prev, key=lambda p: iou(p["bbox"], b["bbox"]), default=None)
        if best is not None and iou(best["bbox"], b["bbox"]) > 0.5:
            best["track_id"] = tid
    # reflection только продолжает трек: после его бокса person и при IoU > 0,6 с предыдущей рамкой трека ≤ 1 с
    last: dict[int, tuple[int, list[float]]] = {}  # трек → (шаг, рамка) последнего оставленного бокса person
    for step, bs in cands.items():
        for b in bs:
            if "track_id" not in b:
                continue
            k = b["track_id"]
            if b["cls"] == "reflection" and not (k in last and step - last[k][0] <= fps and iou(last[k][1], b["bbox"]) > 0.6):
                del b["track_id"]
                continue
            last[k] = (step, b["bbox"])
