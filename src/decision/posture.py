"""Признаки позы для L2 из 33 точек MediaPipe Pose (attributes.pose боксов person, cv.pose): где кисти кассира
относительно столешницы и сейфа, встал ли клиент. Только геометрия относительно зон сцены — абсолютных координат нет.

Точки BlazePose: 11/12 плечи, 13/14 локти, 15/16 запястья, 23/24 бёдра; [x, y, видимость] в пикселях кадра.
"""

SHOULDER, ELBOW, WRIST, HIP = (11, 12), (13, 14), (15, 16), (23, 24)
VISIBLE = 0.5
POSE_HOLD_SEC = 0.5  # поза бокса без своей (POSE_EVERY > 1) — последняя поза того же трека не старше этого
TOGETHER = 0.5  # кисти «вместе» — ближе половины ширины плеч
RISE = 0.15  # клиент «встал» — бёдра поднялись на 15 % его роста относительно посадки


def seen(pose: list, i: int) -> list | None:
    p = pose[i]
    return p if p[2] >= VISIBLE else None


def hand_low(pose: list, desk_bottom: float) -> bool:
    """Одна кисть ушла под столешницу (y ниже её нижнего края), вторая осталась над ней. Скрытая кисть с локтем,
    опущенным ниже середины корпуса, тоже считается ушедшей вниз (рука у колен детектору позы не видна)."""
    low = []
    for side in range(2):
        w, e, s, h = pose[WRIST[side]], pose[ELBOW[side]], pose[SHOULDER[side]], pose[HIP[side]]
        if w[2] >= VISIBLE:
            low.append(w[1] > desk_bottom)
        elif e[2] >= VISIBLE and s[2] >= VISIBLE and h[2] >= VISIBLE:
            low.append(e[1] > s[1] + 0.6 * (h[1] - s[1]))
        else:
            low.append(None)
    return (low[0] is True and low[1] is False) or (low[1] is True and low[0] is False)


def hands_together(pose: list, desk_bottom: float, desk_x: tuple[float, float]) -> bool:
    """Обе кисти видны, рядом друг с другом, над столешницей и в её ширине — перебирает купюры руками."""
    lw, rw, ls, rs = (seen(pose, i) for i in (*WRIST, *SHOULDER))
    if not (lw and rw and ls and rs):
        return False
    width = max(1.0, abs(ls[0] - rs[0]))
    dist = ((lw[0] - rw[0]) ** 2 + (lw[1] - rw[1]) ** 2) ** 0.5
    return (dist < TOGETHER * width and all(w[1] <= desk_bottom for w in (lw, rw))
            and all(desk_x[0] <= w[0] <= desk_x[1] for w in (lw, rw)))


def wrist_in(pose: list, box: list[float] | None, margin: float = 0.2) -> bool:
    """Видимая кисть внутри бокса (сейфа), расширенного на margin его размера."""
    if not box:
        return False
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0, x1, y1 = box[0] - margin * w, box[1] - margin * h, box[2] + margin * w, box[3] + margin * h
    return any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (seen(pose, i) for i in WRIST) if p)


def hip_y(pose: list) -> float | None:
    """Середина бёдер по вертикали; None — бёдра не видны."""
    l, r = seen(pose, HIP[0]), seen(pose, HIP[1])
    return (l[1] + r[1]) / 2 if l and r else None


def risen(pose: list, seated_hip_y: float | None, height: float) -> bool:
    """Бёдра выше посадочного уровня на RISE роста — человек встал."""
    y = hip_y(pose)
    return y is not None and seated_hip_y is not None and y < seated_hip_y - RISE * height
