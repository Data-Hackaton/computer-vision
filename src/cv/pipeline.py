"""Видео → события со статусами → кадры-доказательства. Движок сверки с процедурой — data_prepare (тот же,
что проверяет эталон): check_instance / check_video + sop_events.build_events и reference/scenarios.yaml."""

import io
import subprocess
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from cv.detectors import REFERENCE_DIR, Detection
from data_prepare.reference_actions import check_instance, check_video
from data_prepare.sop_events import build_events

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
WITH_OBJECT = {"give", "put", "take", "inspect"}  # у остальных глаголов объект уже в тексте действия
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
        steps = check_instance(f"{video_name}#{idx}", scenarios[name], actions, [], [], [], hits, inst.get("partial", False))
        for h in hits[first:]:
            h.update(instance=idx, scenario=name)
        instances.append((name, actions, steps))
    check_video(video_name, instance_actions, spec.get("video_invariants", []), [], hits)
    events = build_events(video_name, instances, hits)
    return events + sequence_events(instance_actions, spec)


def sequence_events(instance_actions: list[list[dict]], spec: dict) -> list[dict]:
    """Пропущенные и переставленные обязательные шаги по циклам обслуживания — SOP-чекер NVIDIA
    (decision.sop.sequence_check). Предупреждения для интерфейса, на вердикт не влияют."""
    from decision.sop import sequence_check

    out = []
    for f in sequence_check([a for actions in instance_actions for a in actions], spec):
        step = f["step"]
        what = " ".join(x for x in (ACTOR_RU.get(step.get("actor") or "", ""), ACTION_RU.get(step["action"], step["action"]),
                                    OBJECT_RU.get(step.get("object") or "", "") if step["action"] in WITH_OBJECT else "",
                                    TARGET_RU.get(step.get("target") or "", "")) if x)
        title = "Пропущен шаг процедуры" if f["kind"] == "missing" else "Шаг процедуры не по порядку"
        out.append({"t_start": f["t"], "t_end": f["t"], "event": "sop_step", "actor": step.get("actor"),
                    "object": step.get("object"), "target": step.get("target"), "scenario": "client_service",
                    "instance": f["cycle"], "step": None, "status": "warning", "title": title,
                    "rules": ["nvidia_sop_checker"], "norm": None,
                    "message": f"{title} в цикле обслуживания {f['cycle'] + 1}: {what}."})
    return sorted(out, key=lambda e: e["t_start"])


def why_by_title(spec: dict) -> dict[str, str]:
    """Пояснение правила по его названию — из scenarios.yaml, без ссылок на видео эталона «(X-1)»."""
    rules = [inv for sc in spec["scenarios"].values() for inv in sc.get("invariants", [])]
    rules += spec.get("video_invariants", [])
    out: dict[str, list[str]] = {}
    for inv in rules:
        if inv.get("title") and inv.get("why"):
            why = inv["why"].split(" (X-")[0].strip()
            if why not in out.setdefault(inv["title"], []):
                out[inv["title"]].append(why)
    return {title: "; ".join(whys)[:1].upper() + "; ".join(whys)[1:] for title, whys in out.items()}


def describe(e: dict, why: dict[str, str]) -> str:
    """Что видно на видео и почему это может быть нарушением — текст для проверяющего и жюри."""
    who = ACTOR_RU.get(e.get("actor") or "", "")
    obj = OBJECT_RU.get(e.get("object") or "", "") if e["event"] in WITH_OBJECT else ""
    what = " ".join(x for x in (ACTION_RU.get(e["event"], e["event"]), obj, TARGET_RU.get(e.get("target") or "", "")) if x)
    seen = f"{e['t_start']:.1f} с: {who + ' ' if who else ''}{what}."
    reason = why.get(e.get("title") or "")
    return f"{seen} {reason}." if reason else seen


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
