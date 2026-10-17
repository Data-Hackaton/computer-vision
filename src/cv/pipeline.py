"""Видео → события со статусами → кадры-доказательства. Движок сверки с процедурой — data_prepare (тот же,
что проверяет эталон): автоматы процедур sop_fsm + sop_events.build_events и reference/scenarios.yaml."""

import io
import subprocess
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from cv.detectors import REFERENCE_DIR, Detection
from data_prepare.reference_actions import step_matches
from data_prepare.sop_events import build_events
from data_prepare.sop_fsm import check_instance_fsm, check_video_fsm

ACTOR_RU = {"cashier": "Кассир", "client": "Клиент", "collector": "Инкассатор", "visitor": "Второй посетитель"}
ACTION_RU = {
    "person_enter": "входит в помещение", "person_leave": "выходит из помещения", "give": "передаёт",
    "inspect": "рассматривает", "put": "кладёт", "take": "берёт", "sort_banknotes": "раскладывает купюры по стопкам",
    "count_by_hand": "пересчитывает купюры руками", "safe_open": "открывает сейф", "safe_close": "закрывает сейф",
    "open_bag": "вскрывает инкассаторскую сумку", "use_laptop": "работает за компьютером",
    "screen_lock": "блокирует экран", "paperwork": "оформляет бумаги", "phone_call": "говорит по телефону",
    "lights_off": "выключает свет", "split_bundle": "делит пачку купюр на две части",
    "keep": "оставляет купюры у себя", "safe_left_open": "оставляет сейф открытым",
}
OBJECT_RU = {"banknotes": "купюры", "id_card": "удостоверение", "bag": "сумку", "phone": "телефон",
             "document": "документ", "foreign_object": "личные вещи", "computer": "компьютер"}
TARGET_RU = {"cash_machine": "в счётную машинку", "safe": "в сейф", "safe_top": "на сейф", "desk": "на кассовый стол",
             "client": "клиенту", "cashier": "кассиру", "bag": "из сумки", "door": "к двери"}
WITH_OBJECT = {"give", "put", "take", "inspect"}
STEP_WHY = {  # зачем обязательный шаг цикла обслуживания — для описания SOP-чекера
    ("put", "cash_machine"): "принятые деньги пересчитываются на счётной машинке при клиенте",
    ("person_enter", None): "цикл обслуживания начинается с прихода клиента",
}  # у остальных глаголов объект уже в тексте действия
HIDDEN_ON_EVIDENCE = {"ignore_zone", "reflection", "door"}


def load_spec() -> dict:
    return yaml.safe_load((REFERENCE_DIR / "scenarios.yaml").read_text())


def build_video_events(det: Detection, spec: dict, video_name: str = "video") -> list[dict]:
    """Журнал действий детектора → события со статусами expected / warning / violation."""
    scenarios = spec["scenarios"]
    hits, instances, instance_actions = [], [], []
    for idx, inst in enumerate(det.instances, start=1):
        name = inst["scenario"]
        actions = sorted(inst["actions"], key=lambda a: a["t"])
        instance_actions.append(actions)
        first = len(hits)
        steps = check_instance_fsm(f"{video_name}#{idx}", scenarios[name], actions, [], [], [], hits, inst.get("partial", False))
        for h in hits[first:]:
            h.update(instance=idx, scenario=name)
        instances.append((name, actions, steps))
    check_video_fsm(video_name, instance_actions, spec.get("video_invariants", []), [], hits)
    events = build_events(video_name, instances, hits)
    p_by_t = {(a["t"], a["action"]): a.get("p") for acts in instance_actions for a in acts}
    for e in events:
        uncertain(e, p_by_t.get((e["t_start"], e["event"])))
    return events + sequence_events(instance_actions, spec)


def uncertain(e: dict, p: float | None) -> None:
    """Нарушение по действию, которому ECL (decision.evidence) не верит (certain: False), — предупреждение: детекция
    могла ошибиться, человек проверит по записи. Как шаг такое действие по-прежнему засчитано (L3 его видит)."""
    if e.get("certain", True) or e["status"] != "violation":
        return
    e["status"] = "warning"
    note = f"Модель не уверена в самом действии (оценка {p:.2f})" if p is not None else "Модель не уверена в самом действии"
    e["message"] = f"{e['message']} {note} — проверьте по записи." if e.get("message") else f"{note} — проверьте по записи."


def sequence_events(instance_actions: list[list[dict]], spec: dict) -> list[dict]:
    """Пропущенные и переставленные обязательные шаги по циклам обслуживания — SOP-чекер NVIDIA
    (decision.sop.sequence_check). Предупреждения для интерфейса, на вердикт не влияют. Название говорит, какой шаг,
    интервал события — отрезок обслуживания, в котором шаг ожидался."""
    from decision.sop import core_steps, sequence_check

    actions = sorted((a for acts in instance_actions for a in acts), key=lambda a: a["t"])
    core = core_steps(spec)
    starts = [a["t"] for a in actions if core and step_matches(core[0], a)]  # начало каждого цикла обслуживания
    end_of_video = actions[-1]["t"] if actions else 0.0
    out = []
    for f in sequence_check(actions, spec):
        step, cycle = f["step"], f["cycle"]
        what = " ".join(x for x in (ACTOR_RU.get(step.get("actor") or "", ""), ACTION_RU.get(step["action"], step["action"]),
                                    OBJECT_RU.get(step.get("object") or "", "") if step["action"] in WITH_OBJECT else "",
                                    TARGET_RU.get(step.get("target") or "", "")) if x)
        what = what[:1].lower() + what[1:]
        start = starts[cycle] if 0 <= cycle < len(starts) else f["t"]
        end = starts[cycle + 1] if cycle + 1 < len(starts) else max(end_of_video, f["t"])
        why = STEP_WHY.get((step["action"], step.get("target")), "шаг обязателен в каждом цикле обслуживания")
        if f["kind"] == "missing":
            title = f"Пропущен шаг: {what}"
            message = (f"Обслуживание клиента {cycle + 1} ({start:.1f}–{end:.1f} с): не видно, что {what}. По процедуре "
                       f"{why}. Проверьте по записи: шаг мог быть, но его закрыли руки или люди в кадре.")
        else:
            title = f"Не по порядку: {what}"
            message = (f"Обслуживание клиента {cycle + 1} ({start:.1f}–{end:.1f} с): {what} — раньше, чем положено "
                       f"по процедуре ({why}).")
        out.append({"t_start": start, "t_end": end, "event": "sop_step", "actor": step.get("actor"),
                    "object": step.get("object"), "target": step.get("target"), "scenario": "client_service",
                    "instance": cycle, "step": None, "status": "warning", "title": title,
                    "rules": ["nvidia_sop_checker", f"sop:{f['kind']}"], "norm": None, "message": message})
    return sorted(out, key=lambda e: e["t_start"])


def load_norms() -> dict:
    """Нормы: корпус (norms_corpus.yaml, полные тексты) → аудит пакета проверки (norms_review.yaml) → выверенные
    вручную norms.yaml; поля следующего файла поверх полей предыдущего."""
    out: dict[str, dict] = {}
    for name in ("norms_corpus.yaml", "norms_review.yaml", "norms.yaml"):
        path = REFERENCE_DIR / name
        for key, norm in ((yaml.safe_load(path.read_text()) or {}) if path.exists() else {}).items():
            out[key] = {**out.get(key, {}), **norm}
    return out


def why_by_title(spec: dict, norms: dict | None = None) -> dict[str, str]:
    """Пояснение правила по его названию — из scenarios.yaml, без ссылок на видео эталона «(X-1)»; косвенные и
    условные основания (basis) — с пометкой: прямой норматив (norm) бэкенд добавляет в ответ жюри сам."""
    norms = load_norms() if norms is None else norms
    rules = [inv for sc in spec["scenarios"].values() for inv in sc.get("invariants", [])]
    rules += spec.get("video_invariants", [])
    out: dict[str, list[str]] = {}
    basis: dict[str, dict[str, list[str]]] = {}  # правило → отношение → нормы
    for inv in rules:
        if inv.get("title") and inv.get("why"):
            why = inv["why"].split(" (X-")[0].strip()
            if why not in out.setdefault(inv["title"], []):
                out[inv["title"]].append(why)
            for b in inv.get("basis", []):
                short = norms.get(b["norm"], {}).get("short")
                names = basis.setdefault(inv["title"], {}).setdefault(b["relation"], [])
                if short and short not in names:
                    names.append(short)
    for title, by_relation in basis.items():
        out[title] += [f"основание ({rel}): {join_points(names)}" for rel, names in by_relation.items() if names]
    return {title: "; ".join(whys)[:1].upper() + "; ".join(whys)[1:] for title, whys in out.items()}


def join_points(shorts: list[str]) -> str:
    """«Правила №14, п.8» + «Правила №14, п.16» → «Правила №14, пп.8, 16»."""
    acts: dict[str, list[str]] = {}
    for short in shorts:
        act, sep, point = short.rpartition(", п.")
        acts.setdefault(act if sep else short, []).append(point if sep else "")
    return "; ".join(f"{act}, {'пп.' if len(points) > 1 else 'п.'}{', '.join(points)}" if points[0] else act
                     for act, points in acts.items())


def describe(e: dict, why: dict[str, str]) -> str:
    """Что видно на видео и почему это может быть нарушением — текст для проверяющего и жюри."""
    who = ACTOR_RU.get(e.get("actor") or "", "")
    obj = OBJECT_RU.get(e.get("object") or "", "") if e["event"] in WITH_OBJECT else ""
    what = " ".join(x for x in (ACTION_RU.get(e["event"], e["event"]), obj, TARGET_RU.get(e.get("target") or "", "")) if x)
    seen = f"{e['t_start']:.1f} с: {who + ' ' if who else ''}{what}."
    reason = why.get(e.get("title") or "")
    return f"{seen} {reason}." if reason else seen


EVIDENCE_CONTEXT_SEC = 1.5  # соседние кадры-доказательства: до и после события


def evidence_times(t: float, duration: float | None) -> list[float]:
    """Секунды кадров-доказательства: первый — на таймкоде, затем за EVIDENCE_CONTEXT_SEC до и после (в пределах
    видео) — проверяющему виден ход действия, жюри получает 1–3 кадра."""
    end = duration if duration else t
    return [round(t, 3)] + [round(x, 3) for x in (t - EVIDENCE_CONTEXT_SEC, t + EVIDENCE_CONTEXT_SEC)
                            if 0 <= x <= end - 0.05]


def frame_with_boxes(video: Path, t: float, boxes: list[dict]) -> bytes:
    """Кадр в момент t с боксами модели в пределах ±0,5 с — кадр-доказательство."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    draw = ImageDraw.Draw(image)
    near = [b for b in boxes if abs(b["t"] - t) <= 0.5 and b["cls"] not in HIDDEN_ON_EVIDENCE]
    if near:
        nearest_t = min(near, key=lambda b: abs(b["t"] - t))["t"]
        for b in (b for b in near if b["t"] == nearest_t):
            draw.rectangle(b["bbox"], outline=(255, 64, 64), width=2)
            draw.text((b["bbox"][0] + 2, b["bbox"][1] + 1), b["cls"], fill=(255, 64, 64))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()
