"""VLM на спорных моментах L2 — шаг «отрезок → ответ» у NVIDIA SOP Monitoring.

У NVIDIA Cosmos-Reason2 по кадрам отрезка выбирает номер шага из пронумерованного списка actions.json. Здесь
Qwen3-VL-2B (Cosmos-Reason2 — её дообучение) в сервисе vlm (llama.cpp, CPU) отвечает номером варианта на вопросы
о том, чего не видно в боксах L1: что осталось на столе, что на экране, открыт ли сейф, есть ли посторонний на
стороне кассира. Вопросы задаются в моменты, которые нашёл L2 (уход кассира, его возврат к месту, приход клиента);
ответ становится действием журнала, а нарушение решает L3. Нет VLM_URL, ошибка или таймаут — работают одни правила.
"""

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

VLM_URL = os.environ.get("VLM_URL", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-2b")
VLM_MAX_CALLS = int(os.environ.get("VLM_MAX_CALLS", "6"))
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "30"))
FRAME_WIDTH = 448  # длинная сторона кадра или фрагмента для VLM: ~200–360 токенов изображения у Qwen3-VL

# вопрос → (текст, варианты); ответ — номер варианта с единицы
QUESTIONS = {
    "desk": ("Look at the wooden cashier desk. Apart from the laptop, the desk telephone and the banknote counting "
             "machines, what is lying on it?",
             ["banknotes or cash", "a green cloth bag", "stamps, a notebook or documents", "nothing else"]),
    "screen": ("Look at the laptop on the desk. What does its screen show?",
               ["an open application or document", "a lock screen or desktop wallpaper",
                "the screen is off or the laptop is closed", "no laptop is visible"]),
    "safe": ("Look at the grey metal safe next to the desk. Is its door open?",
             ["the door is open", "the door is closed", "the safe is not visible"]),
    "behind": ("How many people are on the cashier's side of the desk (where the cashier's chair is)?",
               ["one", "two or more", "none"]),
}


def prompt(question: str) -> str:
    text, options = QUESTIONS[question]
    lines = [text, *[f"({i}) {o}" for i, o in enumerate(options, 1)], "Answer with the option number only."]
    return "\n".join(lines)


def frames_at(video: Path, ts: list[float], crop: list[float] | None = None) -> list[bytes]:
    """JPEG кадров на заданные секунды. crop — фрагмент [x1, y1, x2, y2] (сейф, столешница, ноутбук): на целом
    кадре в 448 px мелкие предметы — десяток пикселей, и VLM их не различает. Длинная сторона — FRAME_WIDTH."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = []
    try:
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            if crop:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = (int(max(0, crop[0])), int(max(0, crop[1])), int(min(w, crop[2])), int(min(h, crop[3])))
                frame = frame[y1:y2, x1:x2]
            h, w = frame.shape[:2]
            k = FRAME_WIDTH / max(h, w)
            frame = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                out.append(buf.tobytes())
    finally:
        cap.release()
    return out


def ask_vlm(images: list[bytes], question: str, url: str = VLM_URL, timeout: float = VLM_TIMEOUT) -> int | None:
    """Номер варианта (с единицы) или None, если ответа нет или он не номер."""
    content = [{"type": "text", "text": prompt(question)}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
                for b in images]
    body = json.dumps({"model": VLM_MODEL, "messages": [{"role": "user", "content": content}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    digit = next((ch for ch in text if ch.isdigit()), None)
    n = int(digit) if digit else None
    return n if n and 1 <= n <= len(QUESTIONS[question][1]) else None


def grow(b: list[float], ky: float, kx: float | None = None) -> list[float]:
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
    return [cx - w * (kx or ky) / 2, cy - h * ky / 2, cx + w * (kx or ky) / 2, cy + h * ky / 2]


def crops(zones: dict) -> dict[str, list[float] | None]:
    """Фрагмент кадра под вопрос по медианным боксам L1: сейф с запасом под распахнутую дверцу, столешница;
    про людей у стола — весь кадр."""
    table, safe = zones.get("table"), zones.get("safe")
    return {
        "safe": grow(safe, 1.6, 2.4) if safe else None,
        "desk": [table[0] - 10, table[1] - 0.15 * (table[3] - table[1]), table[2] + 10,
                 table[1] + 0.6 * (table[3] - table[1])] if table else None,
        "behind": None,
    }


class Asker:
    """Вопросы к VLM с лимитом на видео и кэшем ответов (для оценки: data/features/<sha>/vlm_<модель>.json).
    Один кадр на вопрос, фрагмент — по crops(zones): на шлюзе 10 из 13 известных случаев, ~2 с на вопрос."""

    def __init__(self, video: Path, zones: dict, cache_file: Path | None = None, max_calls: int = VLM_MAX_CALLS) -> None:
        self.video, self.crop, self.cache_file, self.max_calls = video, crops(zones), cache_file, max_calls
        self.cache = json.loads(cache_file.read_text()) if cache_file and cache_file.exists() else {}
        self.calls, self.log = 0, []

    def __call__(self, question: str, t: float) -> int | None:
        key = f"{question}@{t:.2f}"
        if key in self.cache:
            return self.cache[key]
        if not VLM_URL or self.calls >= self.max_calls:
            return None
        self.calls += 1
        started = time.time()
        try:
            answer = ask_vlm(frames_at(self.video, [t], self.crop.get(question)), question)
        except Exception as e:  # сеть, таймаут, сервис не поднят — дальше без VLM
            self.log.append(f"{key}: ошибка {e}")
            return None
        self.cache[key] = answer
        self.log.append(f"{key}: {answer} за {time.time() - started:.1f} с")
        if self.cache_file:
            self.cache_file.write_text(json.dumps(self.cache, indent=1))
        return answer


def moments(res: dict) -> list[tuple[str, float]]:
    """Моменты для вопросов, по приоритету: уход кассира, его возврат к месту после отлучки, приход клиента."""
    out = [("leave", a["t"]) for a in res["actions"] if a["action"] == "person_leave" and a.get("actor") == "cashier"]
    # возврат к месту: кассир был дальше 0,5 роста от места (сейф, машинка) не меньше секунды и вернулся
    away_since = None
    for st in res["states"]:
        d = st.get("_cashier_d")
        if d is None:
            continue
        if d > 0.5 and away_since is None:
            away_since = st["t"]
        elif d < 0.3 and away_since is not None:
            if st["t"] - away_since >= 1.0:
                out.append(("back", st["t"]))
            away_since = None
    out += [("enter", a["t"]) for a in res["actions"] if a["action"] == "person_enter" and a.get("actor") == "client"]
    if not out and res["states"]:
        out.append(("middle", res["states"][len(res["states"]) // 2]["t"]))
    return out


# На шлюзе и на 40 видео (17.10): про стол VLM почти всегда отвечает «штампы и документы» — вопрос не задаём;
# про экран путает заставку блокировки с активным экраном (X-29) — остаётся признак яркости; «двое на стороне
# кассира» отвечает почти при каждом приходе клиента — VLM только подтверждает признак L1 (two_at_cashier).
# «Сейф открыт» отвечает верно, но и тогда, когда кассир сам у сейфа: спрашиваем, только когда кассир сидит на
# месте или ушёл, и нужны два ответа «открыт» подряд.
SAFE_LOOKS = (1.5, 3.5)
# На 40 видео вопрос про сейф дал +1 нарушение (X-15, не тот тип) и +2–3 ложных (X-12, X-23: сейф у кресла) —
# по умолчанию выключен; VLM_SAFE=1 — для экспериментов
VLM_SAFE = os.environ.get("VLM_SAFE", "") == "1"


def refine(res: dict, ask: Asker) -> list[dict]:
    """Ответы VLM в спорных моментах → поправки журнала (поверх raw_actions L2 с маркерами)."""
    actions = list(res["raw_actions"])
    states = res["states"]

    def away_from_safe(t):  # у сейфа никого: в сцене B сейф вплотную к креслу, «кассир на месте» — не довод
        st = min(states, key=lambda st: abs(st["t"] - t)) if states else None
        return bool(st) and not st.get("_at_safe")

    added, dropped = [], set()
    for kind, t in moments(res) if VLM_SAFE else []:
        if kind in ("leave", "back"):
            looks = [t + dt for dt in SAFE_LOOKS]
            if all(away_from_safe(x) for x in looks) and all(ask("safe", x) == 1 for x in looks):
                added.append({"t": round(t, 2), "action": "safe_left_open", "actor": "cashier", "source": "vlm"})
    for a in actions:  # посторонний из L1 — подтверждение VLM; «один» — снимаем, нет ответа — оставляем
        if a["action"] == "person_enter" and a.get("actor") == "visitor" and ask("behind", a["t"] + 1.0) == 1:
            dropped.add(id(a))
    return [a for a in actions if id(a) not in dropped] + added
