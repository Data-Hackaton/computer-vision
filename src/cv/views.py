"""Ансамбль видов кадра: один детектор прогоняется по исходному кадру, по кадру с резкостью и с контрастом (cv.channels),
рамки сливаются WBF с весами видов. Без переобучения — те же веса модели.

DETECTOR_VIEWS — JSON: [{"name": "orig", "weight": 0.76}, {"name": "unsharp", "weight": 0.12},
{"name": "clahe", "weight": 0.12}] (веса нормируются к сумме 1). Вес 0,76 подобран золотым сечением по AP50-95 на
половине валидации det_l1f. Пусто — один проход по исходному кадру.
"""

import json

from cv.channels import contrast, sharpen
from cv.geometry import iou

TRANSFORMS = {"orig": lambda frame: frame, "unsharp": sharpen, "clahe": contrast}
# Порог каждого вида до слияния: слабая рамка одного вида добирает уверенность у других
COMPONENT_CONF = 0.05
WBF_IOU = 0.55


def parse_views(raw: str | None) -> list[tuple[str, float]]:
    """DETECTOR_VIEWS → [(вид, вес)] с суммой весов 1; пусто — []."""
    if not raw or not raw.strip():
        return []
    views = [(v["name"], float(v["weight"])) for v in json.loads(raw)]
    unknown = [n for n, _ in views if n not in TRANSFORMS]
    if unknown:
        raise ValueError(f"неизвестные виды кадра {unknown}, есть: {sorted(TRANSFORMS)}")
    total = sum(w for _, w in views)
    if not views or total <= 0 or any(w < 0 for _, w in views):
        raise ValueError(f"веса видов должны быть неотрицательны и не все нули: {raw}")
    return [(n, w / total) for n, w in views]


def wbf(per_view: list[list[tuple[str, list[float], float]]], weights: list[float],
        iou_thr: float = WBF_IOU) -> list[tuple[str, list[float], float]]:
    """Слияние рамок одного кадра по видам. Внутри класса рамки по убыванию уверенности присоединяются к кластеру,
    с рамкой которого IoU больше iou_thr; рамка кластера — среднее, взвешенное по уверенность × вес вида.
    Уверенность кластера — Σ вес вида × лучшая уверенность этого вида в кластере: рамка, которую видит только один
    вид, получает долю его веса."""
    items = [(c, b, s, k) for k, dets in enumerate(per_view) for c, b, s in dets]
    out = []
    for cls in sorted({it[0] for it in items}):
        clusters = []  # [рамка кластера, [(рамка, уверенность, вид)]]
        for _, b, s, k in sorted((it for it in items if it[0] == cls), key=lambda it: -it[2]):
            best = max(range(len(clusters)), key=lambda j: iou(clusters[j][0], b), default=None)
            if best is None or iou(clusters[best][0], b) <= iou_thr:
                clusters.append([list(b), [(b, s, k)]])
                continue
            members = clusters[best][1]
            members.append((b, s, k))
            ws = [ss * weights[kk] + 1e-9 for _, ss, kk in members]
            clusters[best][0] = [sum(bb[i] * w for (bb, _, _), w in zip(members, ws)) / sum(ws) for i in range(4)]
        for box, members in clusters:
            top: dict[int, float] = {}
            for _, s, k in members:
                top[k] = max(top.get(k, 0.0), s)
            out.append((cls, box, sum(weights[k] * s for k, s in top.items())))
    return out
