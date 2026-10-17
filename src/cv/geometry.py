"""Геометрия боксов [x1, y1, x2, y2] в пикселях и зона стекла — общая для предразметки и детекторов сервиса.

Стекло: камера неподвижна, поэтому зона — медиана боксов стекла по видео. Внутри зоны мелкие предметы не на
кассовом столе выбрасываются (это отражения), а человек целиком внутри, не в дверном проёме и не у низа стекла —
отражение (reflection).
"""

import statistics

SMALL_IN_ZONE = {"hand", "banknotes", "id_card", "phone", "bag", "foreign_object", "computer"}


def center_in(b: list[float], zone: list[float]) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def inside(b: list[float], zone: list[float], tol: float = 4.0) -> bool:
    return b[0] >= zone[0] - tol and b[1] >= zone[1] - tol and b[2] <= zone[2] + tol and b[3] <= zone[3] + tol


def intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def overlap_share(a: list[float], b: list[float]) -> float:
    """Доля площади a, лежащая внутри b."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-6, union)


def median_box(boxes: list[list[float]]) -> list[float] | None:
    return [statistics.median(b[k] for b in boxes) for k in range(4)] if boxes else None


def glass_filter(chosen: list[tuple[str, list[float]]], zone: list[float] | None) -> list[tuple[str, list[float]]]:
    """(класс, бокс) кадра → без отражений в стекле; человек в стекле становится reflection."""
    if not zone:
        return chosen
    tables = [b for name, b in chosen if name == "table"]
    doors = [b for name, b in chosen if name == "door"]
    out = []
    for name, b in chosen:
        if name in SMALL_IN_ZONE and center_in(b, zone) and not any(intersects(b, t) for t in tables):
            continue  # предмет в стекле, а не на кассовом столе
        in_doorway = any(overlap_share(b, d) > 0.5 for d in doors)
        if name == "person" and inside(b, zone) and not in_doorway and b[3] < zone[3] - 0.05 * (zone[3] - zone[1]):
            name = "reflection"  # человек в стекле, не в двери и не у низа стекла
        out.append((name, b))
    return out
