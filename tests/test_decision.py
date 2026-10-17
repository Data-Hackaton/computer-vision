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
    # событие называет шаг и охватывает отрезок обслуживания, где шаг ожидался (второй клиент — с 12 с до конца)
    from cv.pipeline import sequence_events

    acts.append({"t": 20, "action": "person_leave", "actor": "client"})
    [e] = sequence_events([acts], spec)
    assert e["title"] == "Пропущен шаг: кассир кладёт купюры в счётную машинку"
    assert (e["t_start"], e["t_end"], e["status"]) == (12, 20, "warning")
    assert "Обслуживание клиента 2 (12.0–20.0 с)" in e["message"] and "пересчитываются" in e["message"]


def test_evidence_times():
    # первый кадр — на таймкоде, соседние — в пределах видео
    from cv.pipeline import evidence_times

    assert evidence_times(7.26, 16.5) == [7.26, 5.76, 8.76]
    assert evidence_times(0.5, 16.5) == [0.5, 2.0]
    assert evidence_times(16.0, 16.5) == [16.0, 14.5]
    assert evidence_times(3.0, None) == [3.0, 1.5]


def test_rule_explanations():
    # пояснения правил для описаний событий (worker.describe): из scenarios.yaml с основаниями из норм
    from cv.pipeline import why_by_title

    why = why_by_title(load_spec())
    assert "основание (условно)" in why["Компьютер с активной сессией без присмотра"]


def test_fsm_matches_legacy_rules_on_synthetic_segments():
    # все фасеты сцены по очереди включаются и выключаются; автоматы дают то же, что прежние правила
    from decision.fsm import instances_of, scene_actions
    from decision.scene import SIGNALS
    from decision.steps import RulesClassifier, to_instances

    base = {s: 0 for s in SIGNALS} | {"cashier_present": 1}
    seq = [{}, {"client_present": 1, "visitors": 1}, {"client_present": 1, "visitors": 2},
           {"client_present": 1, "visitors": 1, "money_on_desk": 1, "bundles_at_cashier": 1},
           {"client_present": 1, "visitors": 1, "bundles_at_cashier": 2, "money_in_machine": 1, "screen_on": 1},
           {"client_present": 1, "visitors": 1, "phone_at_head": 1, "two_at_cashier": 1},
           {"foreign_on_desk": 1, "bag_on_desk": 1}, {"cashier_present": 0, "bag_on_desk": 1}, {"cashier_present": 1}]
    segs = [{"t_start": float(i), "t_end": i + 1.0, "state": base | d} for i, d in enumerate(seq)]
    clf = RulesClassifier()
    legacy = clf.initial(segs[0])
    for p, q in zip(segs, segs[1:]):
        legacy += clf.classify(p, q)
    new = scene_actions(segs)
    assert new == legacy and {a["action"] for a in new} >= {"person_enter", "split_bundle", "phone_call", "_cashier_back"}
    assert instances_of([dict(a) for a in new]) == to_instances([dict(a) for a in legacy])


def test_client_present_from_first_frame():
    """Клиент у кассы с первого кадра — пришёл до записи: экземпляр — обслуживание, а не работа без посетителей."""
    from decision.fsm import instances_of, scene_actions
    from decision.scene import SIGNALS
    from decision.steps import RulesClassifier

    base = {s: 0 for s in SIGNALS} | {"cashier_present": 1}
    segs = [{"t_start": 0.0, "t_end": 5.0, "state": base | {"client_present": 1, "visitors": 1, "screen_on": 1}},
            {"t_start": 5.0, "t_end": 9.0, "state": base | {"client_present": 1, "visitors": 1, "money_on_desk": 1}},
            {"t_start": 9.0, "t_end": 12.0, "state": base}]
    actions = scene_actions(segs)
    assert actions[0] == {"t": 0.0, "action": "person_enter", "actor": "client"}
    assert [i["scenario"] for i in instances_of(actions)] == ["client_service"]
    clf = RulesClassifier()
    assert clf.initial(segs[0]) == scene_actions(segs[:1])


def test_instance_machine_cases():
    from decision.fsm import instances_of

    a = lambda t, action, actor="cashier": {"t": t, "action": action, "actor": actor}
    got = instances_of([a(0, "use_laptop"), a(1, "person_leave"), a(5, "_cashier_back"), a(6, "person_enter", "client"),
                        a(7, "person_leave"), a(8, "person_leave", "client"), a(9, "person_leave")])
    # работа без клиента → уход (переименование), возврат, клиент (кассир ушёл при клиенте — часть обслуживания), уход
    assert [(i["scenario"], [x["action"] for x in i["actions"]]) for i in got] == [
        ("leave_workplace", ["use_laptop", "person_leave"]),
        ("client_service", ["person_enter", "person_leave", "person_leave"]),
        ("leave_workplace", ["person_leave"])]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
