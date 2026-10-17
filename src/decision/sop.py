"""Процедура в формате NVIDIA SOP Monitoring и метрика seq_accuracy.

actions.json (как configs/actions.json у NVIDIA): пронумерованные шаги «(N) …» всех сценариев scenarios.yaml
плюс последний «действие вне процедуры»; actions_can_be_skipped — необязательные шаги и он. Это же — список
вариантов для будущего VLM-классификатора шагов.
  python -m decision.sop > ../data_prepare/reference/actions.json
"""

import json

from cv.pipeline import load_spec

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


if __name__ == "__main__":
    print(json.dumps(actions_json(load_spec()), ensure_ascii=False, indent=2))
