"""Политика L2 temporal_v2 (decision.temporal) на синтетических кадрах: без DECISION_POLICY поведение прежнее;
клиент на месте через reflection не уходит; исчезновение денег не даёт take; split_bundle сохраняется;
сумка, из которой достают деньги, — не личная вещь.

Запуск — как test_decision.py:
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_temporal.py
"""

import os

from decision.steps import analyze
from decision.temporal import stable_presence

FPS = 4
TABLE = [100, 400, 400, 550]
MACHINE = [420, 420, 520, 520]
CASHIER = [150, 450, 260, 800]
CLIENT = [250, 150, 330, 420]
NOTES = [300, 470, 330, 490]
BAG = [320, 430, 380, 480]


def env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return old


def box(t, cls, bbox, conf=0.9):
    return {"t": t, "cls": cls, "bbox": list(bbox), "conf": conf}


def video(seconds, client_at=(), notes_at=(), client_cls="person", extra=()):
    boxes = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        boxes.append(box(t, "table", TABLE))
        boxes.append(box(t, "cash_machine", MACHINE))
        boxes.append(box(t, "person", CASHIER))
        for a, b in client_at:
            if a <= t < b:
                boxes.append(box(t, client_cls if t >= a + 2 else "person", CLIENT, 0.8))
        for a, b in notes_at:
            if a <= t < b:
                boxes.append(box(t, "banknotes", NOTES, 0.7))
        for a, b, cls, bb in extra:
            if a <= t < b:
                boxes.append(box(t, cls, bb, 0.8))
    return boxes


def run(boxes, policy):
    old = env(DECISION_POLICY=policy, TRACKER="greedy", STEP_CLASSIFIER="rules", ECL="off")
    try:
        return analyze(boxes, FPS)
    finally:
        env(**old)


def acts(res, *names):
    return [(round(a["t"], 2), a["action"], a.get("actor"), a.get("target")) for a in res["actions"]
            if not names or a["action"] in names]


def test_stable_presence_bridges_bounded_gaps_only():
    ts = [i / FPS for i in range(20)]
    seen = [True] * 4 + [False] * 4 + [True] * 4 + [False] * 8
    out = stable_presence(seen, ts, FPS, confirm=1.0, gap=2.0)
    assert out[:12] == [True] * 12 and not any(out[12:])  # разрыв в 1 с перекрыт, хвост без наблюдений — нет
    assert not any(stable_presence([True, False] + [False] * 18, ts, FPS, 1.0, 2.0))  # одиночный кадр — не серия


def test_legacy_is_default_and_unchanged():
    boxes = video(10, client_at=[(2, 8)], notes_at=[(3, 5)])
    old = env(DECISION_POLICY=None, TRACKER="greedy", STEP_CLASSIFIER="rules", ECL="off")
    try:
        default = analyze([dict(b) for b in boxes], FPS)
    finally:
        env(**old)
    legacy = run([dict(b) for b in boxes], "legacy")
    assert acts(default) == acts(legacy)
    assert not any(a.get("evidence", {}).get("policy") == "temporal_v2" for a in legacy["raw_actions"])


def test_client_stays_when_seen_as_reflection_at_seat():
    # клиент пришёл человеком, потом L1 метит его отражением на том же месте: уход — только когда исчез
    boxes = video(16, client_at=[(1, 13)], client_cls="reflection")
    res = run(boxes, "temporal_v2")
    leaves = [t for t, a, actor, _ in acts(res, "person_leave") if actor == "client"]
    assert leaves and leaves[0] >= 12.5, acts(res, "person_enter", "person_leave")


def test_disappearance_is_not_take():
    boxes = video(12, client_at=[(1, 11)], notes_at=[(3, 6)])  # деньги на столе 3 с, потом детектор их не видит
    res = run(boxes, "temporal_v2")
    assert [a for _, a, _, tgt in acts(res, "put", "take") if tgt == "desk"] == ["put"]


def test_split_bundle_survives_money_filter():
    from decision.temporal import journal  # фильтр денег в journal пропускает split_bundle
    import inspect

    assert 'a["action"] == "split_bundle"' in inspect.getsource(journal)


def test_collection_bag_is_not_personal_item():
    # сумка на столешнице сначала у тела кассира (перекрытие > 0,5), потом сдвинута (один трек: IoU ≥ 0,5)
    bag = [(1, 2, "bag", [180, 440, 250, 480]), (2, 12, "bag", [195, 440, 265, 480])]
    res = run(video(14, extra=bag), "temporal_v2")
    personal = [x for x in res["actions"] if x.get("object") == "foreign_object"]
    assert personal and 1.0 < personal[0]["t"] <= 2.0, acts(res)
    res = run(video(30, notes_at=[(12, 20)], extra=bag), "temporal_v2")  # через 10 с из сумки достали деньги
    assert not [x for x in res["actions"] if x.get("object") == "foreign_object"]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
