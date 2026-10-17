"""L2 по схеме NVIDIA SOP Monitoring: отрезки по границам событий → шаг процедуры → журнал действий.

У NVIDIA модель DDM режет поток на отрезки (threshold, min_length_sec, max_length_sec), а VLM выбирает для отрезка
номер шага из actions.json. Здесь граница отрезка — смена символьного состояния сцены (decision.scene), а шаг по
переходу между соседними отрезками выбирает классификатор STEP_CLASSIFIER: rules — правила по признакам; место
под VLM — тот же интерфейс. Выход — журнал в формате reference/actions.yaml (экземпляры сценариев с действиями),
его проверяет L3 data_prepare (check_instance / check_video). Экземпляры помечены partial: L2 видит не все шаги
(осмотр удостоверения, бумаги), и пропуск шага не должен становиться замечанием.
"""

import os
import statistics

from decision.scene import SIGNALS, assign_roles, frames_of, scene_states, track_people

SMOOTH_SEC = 1.0  # threshold: окно сглаживания большинством
MIN_LENGTH_SEC = 1.0  # min_length_sec: сигнал короче — шум
MAX_LENGTH_SEC = 10.0  # max_length_sec: длинный отрезок режется (для будущего VLM, правилам не мешает)
# Сколько сигнал должен держаться, чтобы считаться событием (по умолчанию MIN_LENGTH_SEC).
# Кассир «ушёл» — только если его нет дольше 3 с: клиент или стойка могут перекрыть его на секунду-две.
# Клиент «ушёл» — если его нет дольше 2 с (трек рвётся в дверях); второй посетитель — дольше 3 с.
LEAVE_SEAT_D = 0.3  # кассир «у места» — ближе 0,3 роста к своему месту
HOLD_SEC = {"cashier_present": 3.0, "client_present": 2.0, "visitors": 3.0, "foreign_on_desk": 3.0,
            "phone_at_head": 2.0, "bag_on_desk": 1.5, "screen_on": 1.5}


def smooth(values: list[int], win: int) -> list[int]:
    half = max(0, win // 2)
    return [int(statistics.median(values[max(0, i - half): i + half + 1])) for i in range(len(values))]


def debounce(values: list[int], min_run: int) -> list[int]:
    """Серии короче min_run кадров (кроме первой) сливаются с предыдущей."""
    out, i = list(values), 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if i > 0 and j - i < min_run:
            out[i:j] = [out[i - 1]] * (j - i)
            i = max(0, i - 1)
            while i > 0 and out[i - 1] == out[i]:
                i -= 1
            continue
        i = j
    return out


def segments(states: list[dict], fps: float) -> list[dict]:
    """Сглаженные сигналы → отрезки постоянного состояния {t_start, t_end, state}."""
    if not states:
        return []
    series = {}
    for s in SIGNALS:
        v = smooth([st[s] for st in states], round(SMOOTH_SEC * fps))
        series[s] = debounce(v, max(1, round(HOLD_SEC.get(s, MIN_LENGTH_SEC) * fps)))
    ts = [st["t"] for st in states]
    dt = 1 / fps
    # кассир пропал в конце ролика (ушёл, и запись кончилась) — это уход, даже если пропал меньше чем на HOLD_SEC
    raw = smooth([st["cashier_present"] for st in states], round(SMOOTH_SEC * fps))
    tail = next((i for i in range(len(raw) - 1, -1, -1) if raw[i]), -1) + 1
    if len(raw) - tail >= fps and tail > 0:
        series["cashier_present"][tail:] = [0] * (len(raw) - tail)
    out: list[dict] = []
    for i, t in enumerate(ts):
        state = {s: series[s][i] for s in SIGNALS}
        if out and out[-1]["state"] == state and t - out[-1]["t_start"] < MAX_LENGTH_SEC:
            out[-1]["t_end"] = round(t + dt, 2)
        else:
            out.append({"t_start": round(t, 2), "t_end": round(t + dt, 2), "state": state})
    return out


def act(t: float, action: str, actor: str = "cashier", **kw) -> dict:
    return {"t": t, "action": action, "actor": actor, **{k: v for k, v in kw.items() if v is not None}}


class RulesClassifier:
    """Переход между отрезками → действия журнала. Аналог шага VLM «отрезок → номер шага»."""

    # сигнал «предмет лежит» → (объект, куда)
    PLACED = {"money_on_desk": ("banknotes", "desk"), "money_in_machine": ("banknotes", "cash_machine"),
              "foreign_on_desk": ("foreign_object", "desk"), "bag_on_desk": ("bag", "desk")}

    def initial(self, seg: dict) -> list[dict]:
        """Что уже лежит в первом кадре — «положено» до начала видео. Кроме денег: на столе с начала ролика лежат
        бумаги и удостоверения, которые детектор путает с купюрами (ложные «деньги прошлого клиента» в X-37, X-39)."""
        t, st = seg["t_start"], seg["state"]
        out = [act(t, "put", object=o, target=g) for s, (o, g) in self.PLACED.items()
               if s not in ("money_in_machine", "money_on_desk") and st[s]]
        if st["screen_on"] and st["cashier_present"]:
            out.append(act(t, "use_laptop"))
        return out

    def classify(self, prev: dict, cur: dict) -> list[dict]:
        p, c, t = prev["state"], cur["state"], cur["t_start"]
        out = []
        if not p["client_present"] and c["client_present"]:
            out.append(act(t, "person_enter", "client"))
        if p["client_present"] and not c["client_present"]:
            out.append(act(t, "person_leave", "client"))
        if c["visitors"] >= 2 and p["visitors"] < 2 and p["client_present"]:
            out.append(act(t, "person_enter", "visitor"))
        if p["cashier_present"] and not c["cashier_present"]:
            out.append(act(t, "person_leave", "cashier"))
        if not p["cashier_present"] and c["cashier_present"]:
            out.append(act(t, "_cashier_back"))  # служебное: новый экземпляр, в журнал не идёт
        for s, (obj, target) in self.PLACED.items():
            if c[s] and not p[s]:
                out.append(act(t, "put", object=obj, target=target))
            if p[s] and not c[s]:
                out.append(act(t, "take", object=obj, target=target))
        if c["bundles_at_cashier"] >= 2 and p["bundles_at_cashier"] == 1 and c["client_present"]:
            out.append(act(t, "split_bundle", object="banknotes"))
        if c["screen_on"] and not p["screen_on"] and c["cashier_present"]:
            out.append(act(t, "use_laptop"))
        if p["screen_on"] and not c["screen_on"]:
            out.append(act(t, "screen_lock"))
        if c["phone_at_head"] and not p["phone_at_head"]:
            out.append(act(t, "phone_call", t_end=cur["t_end"]))
        return out


def get_classifier():
    kind = os.environ.get("STEP_CLASSIFIER", "rules")
    if kind != "rules":
        raise ValueError(f"неизвестный классификатор шагов: {kind} (есть: rules)")
    return RulesClassifier()


def to_instances(actions: list[dict]) -> list[dict]:
    """Действия → экземпляры сценариев. Новый client_service — на каждый приход клиента (иначе L3 спутает деньги
    двух клиентов); уход кассира без клиента — leave_workplace; остальное без клиента — cash_handling."""
    instances: list[dict] = []
    client_here = False

    def start(scenario):
        instances.append({"scenario": scenario, "partial": True, "actions": []})

    for a in sorted(actions, key=lambda a: a["t"]):
        if a["action"] == "person_enter" and a["actor"] == "client":
            start("client_service")
            client_here = True
        elif a["action"] == "person_leave" and a["actor"] == "cashier" and not client_here:
            # всё, что кассир делал без клиента перед уходом, — часть ухода (как в эталоне: экран, сейф, свет)
            if instances and instances[-1]["scenario"] == "cash_handling":
                instances[-1]["scenario"] = "leave_workplace"
            else:
                start("leave_workplace")
        elif a["action"] == "_cashier_back":
            if not client_here:
                start("cash_handling")
            continue
        elif not instances:
            start("cash_handling")
        if a["action"] == "person_leave" and a["actor"] == "client":
            client_here = False
        instances[-1]["actions"].append(a)
    return [i for i in instances if i["actions"]]


def analyze(boxes: list[dict], fps: float) -> dict:
    """Боксы L1 → треки и роли (в боксах), отрезки, действия, экземпляры."""
    frames = frames_of(boxes)
    track_people(frames, fps)
    roles, seat = assign_roles(frames, fps)
    states = scene_states(frames, roles, seat)
    segs = segments(states, fps)
    clf = get_classifier()
    actions = clf.initial(segs[0]) if segs else []
    for prev, cur in zip(segs, segs[1:]):
        actions += clf.classify(prev, cur)
    # уход кассира — с момента, когда он последний раз был у своего места (встал и пошёл), а не когда пропал из
    # кадра; но не раньше его последнего действия со столом (положил деньги и ушёл — деньги остались на столе)
    for a in actions:
        if a["action"] == "person_leave" and a["actor"] == "cashier":
            at_seat = [st["t"] for st in states if st["t"] <= a["t"] and st["_cashier_d"] is not None
                       and st["_cashier_d"] < LEAVE_SEAT_D]
            last_desk = [b["t"] for b in actions if b.get("target") == "desk" and b["t"] < a["t"]]
            if at_seat:
                a["t"] = round(max([at_seat[-1]] + [t + 0.1 for t in last_desk]), 2)
    return {"roles": roles, "segments": segs, "actions": [a for a in actions if not a["action"].startswith("_")],
            "instances": to_instances(actions)}


def build_journal(boxes: list[dict], fps: float) -> list[dict]:
    return analyze(boxes, fps)["instances"]
