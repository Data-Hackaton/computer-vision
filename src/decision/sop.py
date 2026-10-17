"""Процедура в формате NVIDIA SOP Monitoring и метрика seq_accuracy.

actions.json (как configs/actions.json у NVIDIA): пронумерованные шаги «(N) …» всех сценариев scenarios.yaml
плюс последний «действие вне процедуры»; actions_can_be_skipped — необязательные шаги и он. Это же — список
вариантов для будущего VLM-классификатора шагов.
  python -m decision.sop > ../data_prepare/reference/actions.json
"""

import json

import yaml

from cv.detectors import REFERENCE_DIR
from cv.pipeline import load_spec
from data_prepare.reference_actions import step_matches
from decision.sop_checker import MissingNumberDetector

# Шаг обязателен для чекера, если он есть в 75% чистых циклов обслуживания эталона: из 9 циклов это приход
# клиента (9/9) и пересчёт на машинке (7/9); уход клиента — 5/9 (ролик часто кончается раньше)
CORE_SHARE = 0.75

ACTOR = {"cashier": "кассир", "client": "клиент", "collector": "инкассатор", "visitor": "посторонний"}


def numbered_steps(spec: dict) -> list[tuple[int, str, dict, bool]]:
    """[(номер, сценарий, шаг, можно пропустить)] в порядке scenarios.yaml."""
    out = []
    for name, sc in spec["scenarios"].items():
        for step in sc["steps"]:
            out.append((len(out) + 1, name, step, not step.get("required", False)))
    return out


def step_text(spec: dict, scenario: str, step: dict) -> str:
    what = spec["vocabulary"]["actions"].get(step["action"], step["action"])
    parts = [spec["scenarios"][scenario]["title"] + ":", ACTOR.get(step.get("actor"), step.get("actor") or ""), "—", what]
    for key in ("object", "target"):
        if step.get(key):
            parts.append(f"[{key}: {', '.join(step[key]) if isinstance(step[key], list) else step[key]}]")
    return " ".join(p for p in parts if p)


def actions_json(spec: dict) -> dict:
    steps = numbered_steps(spec)
    actions = [f"({n}) {step_text(spec, sc, st)}" for n, sc, st, _ in steps]
    other = f"({len(steps) + 1}) действие вне процедуры"
    return {"actions": actions + [other],
            "actions_can_be_skipped": [a for a, (_, _, _, skip) in zip(actions, steps) if skip] + [other]}


def seq_accuracy(pred: list[tuple], ref: list[tuple]) -> float:
    """1 − (редакционное расстояние / длина эталона): совпадение последовательностей шагов без учёта времени."""
    if not ref:
        return 1.0 if not pred else 0.0
    d = list(range(len(ref) + 1))
    for i, p in enumerate(pred, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (p != r))
    return max(0.0, 1 - d[len(ref)] / len(ref))


def core_steps(spec: dict) -> list[dict]:
    """Шаги client_service, которые есть почти в каждом чистом цикле эталона (actions.yaml). В шаблоне обязателен
    один шаг — приход клиента, и чекер NVIDIA на нём вырождается; обязательность берём из данных."""
    steps = spec["scenarios"]["client_service"]["steps"]
    journal = yaml.safe_load((REFERENCE_DIR / "actions.yaml").read_text())
    cycles = [i["actions"] for v in journal.values() for i in v["instances"] if i["scenario"] == "client_service"]
    if not cycles:
        return []
    return [s for s in steps if sum(any(step_matches(s, a) for a in c) for c in cycles) >= CORE_SHARE * len(cycles)]


def sequence_check(actions: list[dict], spec: dict, core: list[dict] | None = None) -> list[dict]:
    """Действия видео → пропущенные и переставленные обязательные шаги по циклам (MissingNumberDetector NVIDIA).

    Возвращает [{t, kind: missing|misordered, step, cycle}]; t — момент, когда чекер это понял."""
    core = core if core is not None else core_steps(spec)
    if len(core) < 2:
        return []
    checker = MissingNumberDetector(len(core), {i + 1: i + 1 for i in range(len(core))})
    found = []
    last_t = 0.0
    for a in sorted(actions, key=lambda a: a["t"]):
        number = next((i + 1 for i, s in enumerate(core) if step_matches(s, a)), None)
        if number is None:
            continue
        last_t = a["t"]
        res = checker.process_number(number)
        for kind in ("missing", "misordered"):
            for n in res.get(f"{kind}_detected", []):
                found.append({"t": a["t"], "kind": kind, "step": core[n - 1], "cycle": res.get("cycle")})
    final = checker.finalize_processing()
    for kind in ("missing", "misordered"):
        for n in final.get(f"final_{kind}_detected", []):
            found.append({"t": last_t, "kind": kind, "step": core[n - 1], "cycle": checker.current_cycle - 1})
    return found


if __name__ == "__main__":
    print(json.dumps(actions_json(load_spec()), ensure_ascii=False, indent=2))
