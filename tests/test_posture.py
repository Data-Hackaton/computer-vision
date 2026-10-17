"""Сигналы L2 из позы (decision.posture, scene._hand_hidden/_counting/_reach_safe/_client_up, steps.pose_actions)
на синтетических точках и кадрах.

Запуск — как test_decision.py:
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_posture.py
"""

from decision.posture import hand_low, hands_together, risen, wrist_in
from decision.steps import analyze

FPS = 4
TABLE = [100, 400, 400, 550]  # столешница в scene: y 392.5–482.5
DESK_BOTTOM = 400 + 0.55 * 150
SAFE = [500, 300, 600, 450]
CASHIER = [150, 250, 260, 600]
CLIENT = [250, 150, 330, 420]


def pose(pts):
    """33 невидимые точки, видимые — из pts: {индекс: (x, y)}."""
    out = [[0.0, 0.0, 0.0] for _ in range(33)]
    for i, (x, y) in pts.items():
        out[i] = [float(x), float(y), 0.95]
    return out


def cashier_pose(lw=(180, 450), rw=(230, 450), hips=True):
    pts = {11: (170, 300), 12: (240, 300), 13: (165, 380), 14: (245, 380), 15: lw, 16: rw}
    if hips:
        pts.update({23: (180, 470), 24: (230, 470)})
    return pose(pts)


def test_hand_low_one_wrist_under_desk():
    assert hand_low(cashier_pose(lw=(180, 560)), DESK_BOTTOM)  # левая под столешницей, правая над
    assert not hand_low(cashier_pose(), DESK_BOTTOM)  # обе над
    assert not hand_low(cashier_pose(lw=(180, 560), rw=(230, 560)), DESK_BOTTOM)  # обе под — не «спрятал»
    hidden = cashier_pose()
    hidden[15] = [0, 0, 0.0]  # левая кисть не видна, локоть опущен ниже середины корпуса
    hidden[13] = [165, 440, 0.9]
    assert hand_low(hidden, DESK_BOTTOM)


def test_hands_together_over_desk():
    assert hands_together(cashier_pose(lw=(200, 450), rw=(215, 452)), DESK_BOTTOM, (100, 400))
    assert not hands_together(cashier_pose(lw=(150, 450), rw=(260, 450)), DESK_BOTTOM, (100, 400))  # далеко
    assert not hands_together(cashier_pose(lw=(200, 520), rw=(215, 522)), DESK_BOTTOM, (100, 400))  # под столом


def test_wrist_in_safe_and_risen():
    assert wrist_in(cashier_pose(rw=(550, 380)), SAFE)
    assert not wrist_in(cashier_pose(), SAFE)
    assert risen(pose({23: (280, 300), 24: (300, 300)}), seated_hip_y=360, height=270)
    assert not risen(pose({23: (280, 350), 24: (300, 350)}), seated_hip_y=360, height=270)


def video(seconds, cashier_pose_at):
    """Кассир с деньгами в руках первые 3 с, клиент всё время; поза кассира — cashier_pose_at(t) или None."""
    boxes = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        boxes.append({"t": t, "cls": "table", "bbox": TABLE, "conf": 0.9})
        boxes.append({"t": t, "cls": "safe", "bbox": SAFE, "conf": 0.9})
        p = cashier_pose_at(t)
        boxes.append({"t": t, "cls": "person", "bbox": CASHIER, "conf": 0.9, "attributes": {"pose": p} if p else {}})
        boxes.append({"t": t, "cls": "person", "bbox": CLIENT, "conf": 0.8})
        if t < 3:
            boxes.append({"t": t, "cls": "banknotes", "bbox": [200, 440, 230, 460], "conf": 0.7})
            boxes.append({"t": t, "cls": "hand", "bbox": [195, 435, 235, 465], "conf": 0.7})
    return boxes


def acts(boxes):
    return [(a["t"], a["action"], a.get("target")) for a in analyze(boxes, FPS)["actions"] if a.get("source") == "pose"]


def test_keep_after_money_then_hand_under_desk():
    got = acts(video(10, lambda t: cashier_pose(lw=(180, 560)) if 3 <= t < 5 else cashier_pose()))
    assert [(a, tgt) for _, a, tgt in got] == [("keep", None)] and 2.5 <= got[0][0] <= 3.5
    # без денег перед этим — не событие
    boxes = [b for b in video(10, lambda t: cashier_pose(lw=(180, 560)) if 3 <= t < 5 else cashier_pose())
             if b["cls"] not in ("banknotes", "hand")]
    assert acts(boxes) == []


def test_count_by_hand_and_put_safe():
    got = acts(video(16, lambda t: cashier_pose(lw=(200, 450), rw=(215, 452)) if 1 <= t < 3
                     else cashier_pose(rw=(550, 380)) if 5 <= t < 6.5 else cashier_pose()))
    assert [(a, tgt) for _, a, tgt in got] == [("count_by_hand", None), ("put", "safe")]
    assert 0.5 <= got[0][0] <= 1.5 and 4.5 <= got[1][0] <= 5.5


def test_no_pose_no_actions():
    boxes = video(8, lambda t: None)
    res = analyze(boxes, FPS)
    assert not any(a.get("source") == "pose" for a in res["actions"])
    assert not any(st["_hand_hidden"] or st["_counting"] or st["_reach_safe"] or st["_client_up"] for st in res["states"])


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
