"""L2 на конечных автоматах (pytransitions): сцена и экземпляры процедуры.

SceneModel — по автомату на фасет сцены (model_attribute на сигнал): клиент, посетители, кассир, что лежит на столе
и в машинке, пачки у кассира, экран, телефон у головы, двое у места кассира. Триггер — смена сглаженного сигнала на
границе отрезка (decision.steps.segments); колбэки переходов выдают действия журнала. Порядок фасетов и условия —
как в прежних правилах RulesClassifier (tests: test_decision.py сверяет на синтетике, decision.eval — на 40 видео).

InstanceMachine — экземпляры процедуры по действиям: новый client_service на каждый приход клиента, уход кассира
без клиента → leave_workplace (из cash_handling — переименованием), возврат кассира → cash_handling.

  python -m decision.fsm --diagrams   → data_prepare/docs/procedures.md (mermaid: процедуры, экземпляры, сцена)
"""

import argparse
from pathlib import Path

from transitions import Machine

# фасет → (сигнал из decision.scene, состояния по значению сигнала)
FACETS = {
    "client": ("client_present", ["absent", "present"]),
    "visitors": ("visitors", ["none", "one", "many"]),
    "pair": ("two_at_cashier", ["alone", "two"]),
    "cashier": ("cashier_present", ["away", "here"]),
    "desk_money": ("money_on_desk", ["empty", "placed"]),
    "machine_money": ("money_in_machine", ["empty", "placed"]),
    "desk_foreign": ("foreign_on_desk", ["empty", "placed"]),
    "desk_bag": ("bag_on_desk", ["empty", "placed"]),
    "bundles": ("bundles_at_cashier", ["none", "one", "two"]),
    "screen": ("screen_on", ["off", "on"]),
    "phone": ("phone_at_head", ["down", "at_head"]),
}
# «предмет лежит» → (объект, куда) — порядок как в прежних правилах
PLACED = {"desk_money": ("banknotes", "desk"), "machine_money": ("banknotes", "cash_machine"),
          "desk_foreign": ("foreign_object", "desk"), "desk_bag": ("bag", "desk")}


def act(t: float, action: str, actor: str = "cashier", **kw) -> dict:
    return {"t": t, "action": action, "actor": actor, **{k: v for k, v in kw.items() if v is not None}}


def level(facet: str, value: int) -> str:
    states = FACETS[facet][1]
    return states[min(int(value), len(states) - 1)]


class SceneModel:
    """Автоматы фасетов сцены на одной модели; observe(prev, cur) — граница отрезков, выдаёт действия."""

    def __init__(self, first_state: dict) -> None:
        self.out: list[dict] = []
        self.machines = {}
        for facet, (signal, states) in FACETS.items():
            m = Machine(model=self, states=states, initial=level(facet, first_state[signal]), model_attribute=facet,
                        auto_transitions=False, send_event=True)
            for src in states:  # переход в состояние по новому значению сигнала
                for dst in states:
                    if src != dst:
                        m.add_transition(f"set_{facet}", src, dst, conditions=[self._targets(facet, dst)],
                                         after=self._emitter(facet, src, dst))
                m.add_transition(f"set_{facet}", src, None)  # без изменений
            self.machines[facet] = m

    @staticmethod
    def _targets(facet: str, dst: str):
        return lambda e: level(facet, e.kwargs["cur"][FACETS[facet][0]]) == dst

    def _emitter(self, facet: str, src: str, dst: str):
        def emit(e) -> None:
            p, c, t = e.kwargs["prev"], e.kwargs["cur"], e.kwargs["t"]
            if facet == "client":
                self.out.append(act(t, "person_enter" if dst == "present" else "person_leave", "client"))
            elif facet == "visitors" and dst == "many" and p["client_present"]:
                self._visitor = True  # второй посетитель при клиенте
            elif facet == "pair" and dst == "two":
                self._visitor = True  # двое на месте кассира
            elif facet == "cashier":
                self.out.append(act(t, "person_leave", "cashier") if dst == "away" else act(t, "_cashier_back"))
            elif facet in PLACED:
                obj, target = PLACED[facet]
                self.out.append(act(t, "put" if dst == "placed" else "take", object=obj, target=target))
            elif facet == "bundles" and dst == "two" and src == "one" and c["client_present"]:
                self.out.append(act(t, "split_bundle", object="banknotes"))
            elif facet == "screen":
                if dst == "on" and c["cashier_present"]:
                    self.out.append(act(t, "use_laptop"))
                elif dst == "off":
                    self.out.append(act(t, "screen_lock"))
            elif facet == "phone" and dst == "at_head":
                self.out.append(act(t, "phone_call", t_end=e.kwargs["t_end"]))
        return emit

    def initial(self, seg: dict) -> list[dict]:
        """Что уже лежит в первом кадре — «положено» до начала видео, кроме денег (бумаги и удостоверения на столе
        детектор путает с купюрами); экран включён при кассире — работает за компьютером."""
        t, st = seg["t_start"], seg["state"]
        out = [act(t, "put", object=o, target=g) for facet, (o, g) in PLACED.items()
               if facet not in ("desk_money", "machine_money") and st[FACETS[facet][0]]]
        if st["screen_on"] and st["cashier_present"]:
            out.append(act(t, "use_laptop"))
        return out

    def observe(self, prev: dict, cur: dict) -> list[dict]:
        """Граница отрезков: фасеты в порядке прежних правил; посторонний — одно действие на границу."""
        self.out, self._visitor = [], False
        kw = {"prev": prev["state"], "cur": cur["state"], "t": cur["t_start"], "t_end": cur["t_end"]}
        for facet in ("client", "visitors", "pair"):
            getattr(self, f"set_{facet}")(**kw)
        if self._visitor:
            self.out.append(act(cur["t_start"], "person_enter", "visitor"))
        for facet in ("cashier", *PLACED, "bundles", "screen", "phone"):
            getattr(self, f"set_{facet}")(**kw)
        return self.out


def scene_actions(segs: list[dict]) -> list[dict]:
    """Отрезки → действия журнала через автоматы сцены."""
    if not segs:
        return []
    scene = SceneModel(segs[0]["state"])
    actions = scene.initial(segs[0])
    for prev, cur in zip(segs, segs[1:]):
        actions += scene.observe(prev, cur)
    return actions


class InstanceMachine:
    """Экземпляры процедуры: none → cash_handling / client_service / leave_workplace по действиям."""

    states = ["none", "cash_handling", "client_service", "leave_workplace"]

    def __init__(self) -> None:
        self.instances: list[dict] = []
        self.client_here = False
        m = Machine(model=self, states=self.states, initial="none", auto_transitions=False, send_event=True)
        m.add_transition("client_enters", "*", "client_service", after=self._start)
        m.add_transition("cashier_leaves", "cash_handling", "leave_workplace", unless=[self._client], after=self._rename)
        m.add_transition("cashier_leaves", "*", "leave_workplace", unless=[self._client], after=self._start)
        m.add_transition("cashier_leaves", "*", None)  # при клиенте — часть обслуживания
        m.add_transition("cashier_back", "*", "cash_handling", unless=[self._client], after=self._start)
        m.add_transition("cashier_back", "*", None)
        m.add_transition("other", "none", "cash_handling", after=self._start)
        m.add_transition("other", "*", None)
        self.machine = m

    def _client(self, e) -> bool:
        return self.client_here

    def _start(self, e) -> None:
        self.instances.append({"scenario": self.state, "partial": True, "actions": []})

    def _rename(self, e) -> None:
        self.instances[-1]["scenario"] = "leave_workplace"

    def observe(self, a: dict) -> None:
        if a["action"] == "person_enter" and a["actor"] == "client":
            self.client_enters()
            self.client_here = True
        elif a["action"] == "person_leave" and a["actor"] == "cashier":
            self.cashier_leaves()
        elif a["action"] == "_cashier_back":
            self.cashier_back()
            return  # служебное действие в журнал не идёт
        else:
            self.other()
        if a["action"] == "person_leave" and a["actor"] == "client":
            self.client_here = False
        self.instances[-1]["actions"].append(a)


def instances_of(actions: list[dict]) -> list[dict]:
    machine = InstanceMachine()
    for a in sorted(actions, key=lambda a: a["t"]):
        machine.observe(a)
    return [i for i in machine.instances if i["actions"]]


def clean(code: str, title: str) -> str:
    """Вывод GraphMachine → mermaid, который рендерят GitHub и локальные рендереры: без frontmatter, без раскраски активного
    состояния (classDef/Class — это снимок текущего состояния, для схемы процедуры не нужен), id не с цифры."""
    import re

    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith(("classDef", "Class "))]
    if lines and lines[0] == "---":  # заголовок-frontmatter не везде рендерится; название — в заголовке раздела
        lines = lines[3:]
    return re.sub(r"\b(\d+_\w+)", r"s\1", "\n".join(lines))


def diagrams() -> str:
    """Mermaid: процедуры scenarios.yaml (шаги вперёд, пропуск необязательных), экземпляры, фасеты сцены."""
    from transitions.extensions import GraphMachine

    from cv.pipeline import load_spec

    def graph(states, transitions, initial, title):
        model = type("M", (), {})()
        m = GraphMachine(model=model, states=states, initial=initial, auto_transitions=False, graph_engine="mermaid",
                         title=title)
        for trigger, src, dst in transitions:
            m.add_transition(trigger, src, dst)
        return clean(model.get_graph().draw(None), title)

    spec = load_spec()
    parts = ["# Процедуры и автоматы L2/L3", "",
             "Сгенерировано `python -m decision.fsm --diagrams` из `reference/scenarios.yaml` и `decision/fsm.py`.", ""]
    for name, sc in spec["scenarios"].items():
        steps = sc["steps"]
        names = [f"{i + 1}_{s['action']}" + (f"_{s['actor']}" if s.get("actor") else "") +
                 (f"_{s['target']}" if s.get("target") else "") for i, s in enumerate(steps)]
        trans = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                trans.append(("далее", names[i], names[j]))
                if steps[j].get("required"):
                    break
        parts += [f"## {sc.get('title', name)} (`{name}`)", "",
                  "Обязательные шаги: " + ", ".join(n for n, s in zip(names, steps) if s.get("required")) + ".", "",
                  "```mermaid", graph(names, trans, names[0], name).strip(), "```", ""]
    inst = [("client_enters", "*", "client_service"), ("cashier_leaves", "cash_handling", "leave_workplace"),
            ("cashier_leaves", "client_service", "leave_workplace"), ("cashier_back", "leave_workplace", "cash_handling"),
            ("other", "none", "cash_handling")]
    expanded = [(tr, s, d) for tr, src, d in inst for s in (InstanceMachine.states if src == "*" else [src])]
    parts += ["## Экземпляры процедуры (L2)", "", "```mermaid",
              graph(InstanceMachine.states, expanded, "none", "instances").strip(), "```", ""]
    for facet, (signal, states) in FACETS.items():
        trans = [(f"{signal}", s, d) for s in states for d in states if s != d]
        parts += [f"## Сцена: {facet} (`{signal}`)", "", "```mermaid", graph(states, trans, states[0], facet).strip(),
                  "```", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagrams", type=Path, nargs="?", const=Path("/opt/data_prepare/docs/procedures.md"))
    args = ap.parse_args()
    if args.diagrams:
        args.diagrams.write_text(diagrams())
        print(f"→ {args.diagrams}")
