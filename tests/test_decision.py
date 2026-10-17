"""L2 на синтетических кадрах: сигналы → действия → экземпляры → события L3 (движок data_prepare).

Запуск (pytest в образе cv нет — он на 11 ГБ, пересобирать ради тестов незачем):
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_decision.py
"""

from cv.detectors import Detection
from cv.pipeline import build_video_events, load_spec
from decision.steps import analyze, debounce, to_instances

FPS = 4
TABLE = [100, 400, 400, 550]
CASHIER = [150, 450, 260, 800]  # сидит у стола, весь ролик
CLIENT = [250, 150, 330, 420]  # за столом, у стекла
NOTES = [300, 470, 330, 490]  # на столе, не в руке


def video(seconds: float, client_at: list[tuple[float, float]], notes_at: list[tuple[float, float]],
          cashier_until: float | None = None) -> list[dict]:
    boxes = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        boxes.append({"t": t, "cls": "table", "bbox": TABLE, "conf": 0.9})
        if cashier_until is None or t < cashier_until:
            boxes.append({"t": t, "cls": "person", "bbox": CASHIER, "conf": 0.9})
        if any(a <= t < b for a, b in client_at):
            boxes.append({"t": t, "cls": "person", "bbox": CLIENT, "conf": 0.8})
        if any(a <= t < b for a, b in notes_at):
            boxes.append({"t": t, "cls": "banknotes", "bbox": NOTES, "conf": 0.7})
    return boxes


def events(boxes):
    res = analyze(boxes, FPS)
    return res, build_video_events(Detection("test", instances=res["instances"]), load_spec(), "test")


def test_debounce_merges_short_runs():
    assert debounce([0, 0, 1, 0, 0, 1, 1, 1, 1], 3) == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_roles_and_clean_service():
    # клиент пришёл, положил деньги, кассир их убрал, клиент ушёл — нарушений нет
    res, ev = events(video(30, client_at=[(5, 20)], notes_at=[(8, 12)]))
    assert set(res["roles"].values()) == {"cashier", "client"}
    kinds = [(a["action"], a.get("target")) for a in res["actions"]]
    assert ("person_enter", None) in kinds and ("put", "desk") in kinds and ("take", "desk") in kinds
    assert not [e for e in ev if e["status"] == "violation"]


def test_money_of_previous_client_on_desk():
    # деньги первого клиента остались на столе, входит второй — №231 п.20 (как X-4)
    res, ev = events(video(40, client_at=[(2, 12), (22, 35)], notes_at=[(6, 40)]))
    viol = [e for e in ev if e["status"] == "violation"]
    assert [e["title"] for e in viol] == ["Деньги предыдущего клиента на кассовом столе"]
    assert abs(viol[0]["t_start"] - 22) <= 1 and viol[0]["norm"] == "n231_p20"


def test_cashier_leaves_money_on_desk():
    # кассир одна, деньги на столе, уходит — №231 п.20 (как X-26)
    res, ev = events(video(30, client_at=[], notes_at=[(3, 30)], cashier_until=20))
    assert [i["scenario"] for i in res["instances"]][-1] == "leave_workplace"
    viol = [e for e in ev if e["status"] == "violation"]
    assert [e["title"] for e in viol] == ["Ценности на столе при уходе кассира"]


def test_new_instance_per_client():
    actions = [{"t": 1, "action": "person_enter", "actor": "client"}, {"t": 5, "action": "person_leave", "actor": "client"},
               {"t": 9, "action": "person_enter", "actor": "client"}]
    assert [i["scenario"] for i in to_instances(actions)] == ["client_service", "client_service"]


def test_sop_checker_missing_count_on_machine():
    # SOP-чекер NVIDIA на ядре эталона «пришёл клиент → деньги в машинку»: во втором цикле пересчёта нет
    from decision.sop import core_steps, sequence_check

    spec = load_spec()
    assert [(s["action"], s.get("target")) for s in core_steps(spec)] == [("person_enter", None), ("put", "cash_machine")]
    acts = [{"t": 1, "action": "person_enter", "actor": "client"},
            {"t": 5, "action": "put", "actor": "cashier", "object": "banknotes", "target": "cash_machine"},
            {"t": 12, "action": "person_enter", "actor": "client"}]
    found = sequence_check(acts, spec)
    assert [(f["kind"], f["step"]["target"], f["cycle"]) for f in found] == [("missing", "cash_machine", 1)]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
