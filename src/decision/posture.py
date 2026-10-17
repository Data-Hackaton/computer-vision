"""Признаки позы для L2 из 33 точек MediaPipe Pose (attributes.pose боксов person, cv.pose): где кисти кассира
относительно столешницы и сейфа, встал ли клиент. Только геометрия относительно зон сцены — абсолютных координат нет.

Точки BlazePose: 11/12 плечи, 13/14 локти, 15/16 запястья, 23/24 бёдра; [x, y, видимость] в пикселях кадра.
"""

SHOULDER, ELBOW, WRIST, HIP = (11, 12), (13, 14), (15, 16), (23, 24)
VISIBLE = 0.5
POSE_HOLD_SEC = 0.5  # поза бокса без своей (POSE_EVERY > 1) — последняя поза того же трека не старше этого
TOGETHER = 0.5  # кисти «вместе» — ближе половины ширины плеч
RISE = 0.15  # клиент «встал» — бёдра поднялись на 15 % его роста относительно посадки
LOW = 0.1  # кисть «ушла вниз» — ниже бёдер на 10 % роста
# Пробовали и убрали: «одна кисть скрыта, вторая видна» (в X-1 правая рука уходит за тело при локте выше бёдер) —
# на чистых X-3, X-35, X-39 кассир так же поворачивается боком после денег, одни ложные «присвоения»


def seen(pose: list, i: int) -> list | None:
    p = pose[i]
    return p if p[2] >= VISIBLE else None


def hand_low(pose: list, height: float) -> bool:
    """Одна кисть ушла ниже бёдер (к коленям), вторая осталась выше. Относительно своего тела, а не столешницы:
    камера смотрит из-за спины кассира, и руки на коленях в кадре тоже «под столом» — это обычная поза, а ниже бёдер
    руки не держат. Скрытая кисть с локтем ниже бёдер тоже считается ушедшей вниз (у колен её не видно).
    На X-1 не срабатывает: там правая рука уходит за тело, локоть остаётся выше бёдер."""
    hips = hip_y(pose)
    if hips is None:
        return False
    low = []
    for side in range(2):
        w, e = pose[WRIST[side]], pose[ELBOW[side]]
        if w[2] >= VISIBLE:
            low.append(w[1] > hips + LOW * height)
        elif e[2] >= VISIBLE:
            low.append(e[1] > hips)
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


def hip_y(pose: list) -> float | None:
    """Середина бёдер по вертикали; None — бёдра не видны."""
    l, r = seen(pose, HIP[0]), seen(pose, HIP[1])
    return (l[1] + r[1]) / 2 if l and r else None


def risen(pose: list, seated_hip_y: float | None, height: float) -> bool:
    """Бёдра выше посадочного уровня на RISE роста — человек встал."""
    y = hip_y(pose)
    return y is not None and seated_hip_y is not None and y < seated_hip_y - RISE * height
