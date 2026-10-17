"""L1-боксы по кадрам → треки людей, роли и символьное состояние сцены на каждом кадре (вход L2).

Только относительная геометрия: стол, машинка — медианы своих боксов по видео (камера неподвижна), роли — по
поведению трека. Абсолютных зон кадра нет: у жюри видео снято под другим углом.
  кассир — самый долгий трек и любой трек, который держится у его места (кассир вернулся — новый трек);
  клиент — остальные треки людей дольше секунды (отражения уже отсеяны в L1).
"""

import statistics
from collections import defaultdict

from cv.geometry import center_in, iou, median_box, overlap_share

SIGNALS = ["cashier_present", "client_present", "visitors", "money_on_desk", "money_in_machine", "bundles_at_cashier",
           "foreign_on_desk", "bag_on_desk", "phone_at_head", "screen_on", "two_at_cashier"]
# Экран «включён»: доля ярких пикселей в боксе ноутбука (computer.bright из L1). На датасете рабочий экран — 0,40–0,50,
# заставка блокировки или обои — 0,22–0,29, закрытая крышка или выключенный свет — ниже 0,2.
SCREEN_ON = 0.35
# Состояние сейфа по пикселям (safe.dark) не годится: открытая дверца у стены не даёт тёмного нутра (X-20),
# а при выключенном свете тёмным становится всё (X-12). Сигнала сейфа в L2 нет.
# Детектор, обученный на предразметке Grounding DINO, путает купюры и удостоверение (мелкий прямоугольник на столе):
# для L2 это один класс «деньги/бумаги». Удостоверение в правилах нарушений не участвует.
MONEY = {"banknotes", "id_card"}
# Порог уверенности для классов, по которым срабатывают нарушения без контекста (вещь, сумка, телефон)
MIN_CONF = {"foreign_object": 0.5, "bag": 0.45, "phone": 0.5}


def frames_of(boxes: list[dict]) -> list[tuple[float, list[dict]]]:
    by_t: dict[float, list[dict]] = defaultdict(list)
    for b in boxes:
        by_t[b["t"]].append(b)
    return sorted(by_t.items())


def track_people(frames: list[tuple[float, list[dict]]], fps: float, lost_sec: float = 3.0) -> None:
    """Трекинг людей: жадное сопоставление с живыми треками по IoU, а при слабом IoU — по близости центров
    (человек в комнате движется плавно); трек живёт lost_sec после последнего бокса. track_id пишется в боксы.
    В кадре 1–3 человека — ByteTrack не нужен (в supervision 0.31 его уже нет, у Ultralytics — AGPL)."""
    tracks: dict[int, tuple[float, list[float]]] = {}  # id → (когда видели, последний бокс)
    next_id = 1
    for t, boxes in frames:
        people = []
        # reflection только продолжает живой трек: вставший перед стеклом кассир в L1 похож на отражение
        cands = [b for b in boxes if b["cls"] == "person"] + [
            b for b in boxes if b["cls"] == "reflection"
            and any(iou(b["bbox"], last) > 0.6 for seen, last in tracks.values() if t - seen <= 1.0)]
        for b in sorted(cands, key=lambda b: (b["cls"] != "person", -(b.get("conf") or 0))):
            # вложенный бокс (человек по пояс) — часть уже взятого, без трека и без роли
            if not any(overlap_share(b["bbox"], k["bbox"]) > 0.7 or overlap_share(k["bbox"], b["bbox"]) > 0.7
                       for k in people):
                people.append(b)
        alive = {k: v for k, v in tracks.items() if t - v[0] <= lost_sec}
        pairs = []
        for b in people:
            for k, (_, last) in alive.items():
                h = max(1.0, last[3] - last[1])
                dist = (((b["bbox"][0] + b["bbox"][2]) - (last[0] + last[2])) ** 2
                        + ((b["bbox"][1] + b["bbox"][3]) - (last[1] + last[3])) ** 2) ** 0.5 / 2
                score = iou(b["bbox"], last) if iou(b["bbox"], last) > 0.2 else (0.2 - dist / h if dist < 0.5 * h else -1)
                if score > -1:
                    pairs.append((score, id(b), k, b))
        used_b, used_k = set(), set()
        for _, bid, k, b in sorted(pairs, key=lambda x: -x[0]):
            if bid in used_b or k in used_k:
                continue
            used_b.add(bid), used_k.add(k)
            b["track_id"], tracks[k] = k, (t, b["bbox"])
        for b in people:
            if id(b) not in used_b and b["cls"] == "person":
                b["track_id"], tracks[next_id] = next_id, (t, b["bbox"])
                next_id += 1


def assign_roles(frames: list[tuple[float, list[dict]]], fps: float) -> tuple[dict[int, str], dict | None]:
    """track_id → cashier / client и место кассира {x, y, h}; роль пишется в attributes.role боксов.

    Клиент — трек дольше секунды, который хоть раз подходил к столешнице: люди за стеклом и ушедший кассир в
    коридоре к столу не подходят."""
    tracks: dict[int, list[list[float]]] = defaultdict(list)
    for _, boxes in frames:
        for b in boxes:
            if b["cls"] in ("person", "reflection") and "track_id" in b:
                tracks[b["track_id"]].append(b["bbox"])
    table = median_box([b["bbox"] for _, boxes in frames for b in boxes if b["cls"] == "table"])
    desk_top = [table[0], table[1], table[2], table[1] + 0.55 * (table[3] - table[1])] if table else None
    if not tracks:
        return {}, None
    longest = max(tracks, key=lambda k: len(tracks[k]))
    # место — по первой четверти трека: в начале ролика кассир обычно сидит, дальше встаёт, ходит к сейфу, уходит
    early = tracks[longest][: max(1, len(tracks[longest]) // 4)]
    seat = (statistics.median((b[0] + b[2]) / 2 for b in early), statistics.median(b[3] for b in early))
    height = statistics.median(b[3] - b[1] for b in early)
    roles = {}
    for tid, bxs in tracks.items():
        fx = statistics.median((b[0] + b[2]) / 2 for b in bxs)
        fy = statistics.median(b[3] for b in bxs)
        near_seat = ((fx - seat[0]) ** 2 + (fy - seat[1]) ** 2) ** 0.5 < 0.6 * height
        if tid == longest or (near_seat and len(bxs) >= 2 * fps):
            roles[tid] = "cashier"
        elif len(bxs) >= fps and (not desk_top or any(overlap_share(b, desk_top) > 0.05 for b in bxs)):
            roles[tid] = "client"
    for _, boxes in frames:
        for b in boxes:
            if b["cls"] in ("person", "reflection") and b.get("track_id") in roles:
                b["attributes"] = {**b.get("attributes", {}), "role": roles[b["track_id"]]}
    return roles, {"x": seat[0], "y": seat[1], "h": height}


def scene_states(frames: list[tuple[float, list[dict]]], roles: dict[int, str], seat: dict | None = None) -> list[dict]:
    """Сырое состояние сцены по кадрам: {t, сигнал: значение}; сглаживание — в decision.steps.

    seat — место кассира {x, y, h} из assign_roles. Кассир на работе, пока он у места или рядом (сейф и машинка
    на той же глубине, что и стол). Ушёл — далеко от места и ушёл вглубь кадра (ноги выше места на полроста):
    дверь и коридор — за стеклом позади стола, и ушедшего там видно."""

    def at_work(b):
        if not seat:
            return True
        fx, fy = (b[0] + b[2]) / 2, b[3]
        far = ((fx - seat["x"]) ** 2 + (fy - seat["y"]) ** 2) ** 0.5 > 0.6 * seat["h"]
        return not (far and fy < seat["y"] - 0.5 * seat["h"])

    def zone(cls):
        return median_box([b["bbox"] for _, boxes in frames for b in boxes if b["cls"] == cls])

    table, machine, safe = zone("table"), zone("cash_machine"), zone("safe")
    if table:  # столешница — верхние 55% бокса стола (ниже — тумба и кресло кассира), чуть шире по краям
        w, h = table[2] - table[0], table[3] - table[1]
        table = [table[0] - 0.05 * w, table[1] - 0.05 * h, table[2] + 0.05 * w, table[1] + 0.55 * h]
    def desk_screen(boxes):
        """Яркость экрана компьютера на столешнице (computer.bright из L1) или None, если его не видно."""
        on_desk = [b for b in boxes if b["cls"] == "computer" and table and center_in(b["bbox"], table)
                   and "bright" in b.get("attributes", {})]
        return max(on_desk, key=lambda b: b.get("conf") or 0)["attributes"]["bright"] if on_desk else None

    screen_prev = 0
    states = []
    for t, boxes in frames:
        boxes = [b for b in boxes if (b.get("conf") or 0) >= MIN_CONF.get(b["cls"], 0)]
        people = [b for b in boxes if b["cls"] in ("person", "reflection") and "track_id" in b]
        cashiers = [b for b in people if b.get("attributes", {}).get("role") == "cashier" and at_work(b["bbox"])]
        clients = [b for b in people if b.get("attributes", {}).get("role") == "client"]
        hands = [b["bbox"] for b in boxes if b["cls"] == "hand"]
        notes = [b["bbox"] for b in boxes if b["cls"] in MONEY]

        def held(b):
            return any(overlap_share(b, hd) > 0.2 for hd in hands)

        def on_table(b):
            return bool(table) and center_in(b, table)

        def big(b):
            return bool(table) and (b[2] - b[0]) * (b[3] - b[1]) >= 0.03 * (table[2] - table[0]) * (table[3] - table[1])

        in_machine = [b for b in notes if machine and overlap_share(b, machine) > 0.3]
        screen = desk_screen(boxes)
        if screen is not None:  # экран загорожен — прошлое значение
            screen_prev = int(screen >= SCREEN_ON)
        at_cashier = [b for b in notes if b not in in_machine and any(overlap_share(b, c["bbox"]) > 0.5 for c in cashiers)]
        states.append({
            "t": t,
            "cashier_present": int(bool(cashiers)),
            "client_present": int(bool(clients)),
            "visitors": len(clients),
            "money_on_desk": int(any(on_table(b) and not held(b) and b not in in_machine for b in notes)),
            "money_in_machine": int(bool(in_machine)),
            "bundles_at_cashier": min(len(at_cashier), 2),
            # личная вещь — крупная (сумка, одежда): стаканы и мелочь на столе есть и в чистых видео (X-25)
            "foreign_on_desk": int(any(b["cls"] == "foreign_object" and on_table(b["bbox"]) and big(b["bbox"])
                                       and not any(overlap_share(b["bbox"], p["bbox"]) > 0.6 for p in people)
                                       for b in boxes)),
            "bag_on_desk": int(any(b["cls"] == "bag" and on_table(b["bbox"]) and not held(b["bbox"]) for b in boxes)),
            # телефон у головы кассира (верхние 40% бокса): в руках над столом детектор путает его с удостоверением
            "phone_at_head": int(any(b["cls"] == "phone" and any(
                overlap_share(b["bbox"], c["bbox"]) > 0.5
                and (b["bbox"][1] + b["bbox"][3]) / 2 < c["bbox"][1] + 0.4 * (c["bbox"][3] - c["bbox"][1])
                for c in cashiers) for b in boxes)),
            "screen_on": screen_prev,
            # двое на месте кассира — второй трек держится у места кассира (посторонний за кассовым столом, X-30);
            # на 19 чистых видео такого нет ни разу
            "two_at_cashier": int(len(cashiers) >= 2),
            # у сейфа кто-то стоит (человек перекрывает больше 30% бокса сейфа); не сигнал, а подсказка для VLM
            "_at_safe": bool(safe) and any(overlap_share(safe, b["bbox"]) > 0.3 for b in people),
            # расстояние кассира до его места в ростах (None — кассира не видно); не сигнал, а подсказка для L2
            "_cashier_d": min((((c["bbox"][0] + c["bbox"][2]) / 2 - seat["x"]) ** 2
                               + (c["bbox"][3] - seat["y"]) ** 2) ** 0.5 / seat["h"] for c in cashiers)
            if cashiers and seat else None,
        })
    return states
