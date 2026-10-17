"""Type-aware replay metrics. Unknown reference types never become guessed labels."""

EXPECTED = {
    "Присвоение части наличных": "cash_retention",
    "Приём наличных без предварительной визуальной идентификации клиента": "identity_check",
    "Деньги предыдущего клиента на кассовом столе": "previous_cash",
    "Оставление включённого оборудования после завершения работы": "active_equipment",
    "Пропуск пересчёта денег из инкассаторской сумки перед хранением": "missing_count",
    "Оставление сейфа открытым после извлечения наличных": "open_safe",
    "Предполагаемое оставление ключа в замке сейфа при уходе": "safe_key",
    "Оставление предполагаемых штампов и служебной книги на столе при уходе": "unattended_valuables",
    "Оставление наличных без присмотра вне сейфа": "unattended_valuables",
    "Предполагаемый допуск постороннего в рабочую зону кассира": "visitor",
    "Пересчёт наличных после ухода клиента": "client_absent_count",
    "Личные вещи на кассовом столе": "personal_items",
    "Напиток на кассовом рабочем месте во время работы с наличными": "food_drink",
}
PREDICTED = {
    "Присвоение части наличных": "cash_retention",
    "Деньги предыдущего клиента на кассовом столе": "previous_cash",
    "Компьютер с активной сессией без присмотра": "active_equipment",
    "Деньги в сейф без пересчёта на машинке": "missing_count",
    "Пропуск пересчёта денег из инкассаторской сумки перед хранением": "missing_count",
    "Сейф оставлен открытым": "open_safe",
    "Ценности на столе при уходе кассира": "unattended_valuables",
    "Постороннее лицо у кассы": "visitor",
    "Постороннее лицо у кассы во время обслуживания": "visitor",
    "Пересчёт наличных после ухода клиента": "client_absent_count",
    "Личные вещи на кассовом столе": "personal_items",
}


def typed_result(expected: str | None, labels: list[float], violations: list[dict]) -> dict:
    kind = EXPECTED.get(expected)
    if kind is None:
        return {"expected": None, "matched": None, "hits": [], "wrong_type": None, "wrong_events": []}
    correct = [e for e in violations if PREDICTED.get(e["title"]) == kind]
    # One event cannot claim two labelled episodes. Sorted interval matching maximizes cardinality.
    available = sorted(e["t"] for e in correct)
    hits = []
    for t in sorted(set(labels)):
        available = [x for x in available if x >= t - 3]
        if available and available[0] <= t + 3:
            hits.append(t)
            available.pop(0)
    wrong = [{"type": PREDICTED.get(e["title"], "unmapped:" + e["title"]), "t": e["t"]}
             for e in violations if PREDICTED.get(e["title"]) != kind]
    return {"expected": kind, "matched": bool(correct), "hits": hits,
            "wrong_type": bool(wrong), "wrong_events": wrong}


def typed_summary(rows: list[dict]) -> dict:
    known = [r for r in rows if r["label"] == "violation" and r["typed"]["expected"] is not None]
    return {"known_positive": len(known), "typed_tp": sum(r["typed"]["matched"] for r in known),
            "typed_hits": sum(bool(r["typed"]["hits"]) for r in known),
            "unknown_positive": sum(r["label"] == "violation" and r["typed"]["expected"] is None for r in rows),
            "wrong_type_videos": [r["video"] for r in known if r["typed"]["wrong_type"]],
            "false_positive_videos": [r["video"] for r in rows if r["label"] == "no_violation" and r["verdict"] == "violation"]}


def release_failures(candidate: list[dict], baseline: list[dict]) -> list[str]:
    """Strict temporal_v2 promotion gate; historical group metrics remain unchanged."""
    failures = []
    if len(candidate) != 40 or len(baseline) != 40 or len({r['video'] for r in candidate}) != 40 or {
            r['video'] for r in candidate} != {r['video'] for r in baseline}:
        return ["comparison must contain the same 40 unique videos"]
    right = sum(r["verdict"] == r["label"] for r in candidate)
    old_by_video = {r["video"]: r for r in baseline}
    for row in candidate:
        old_row = old_by_video[row["video"]]
        if any(row.get(k) != old_row.get(k) for k in ("label", "expected_type", "label_t", "group")):
            failures.append(f"changed reference labels for {row['video']}")
    new, old = typed_summary(candidate), typed_summary(baseline)
    if right < 28:
        failures.append(f"correct verdicts {right}/40 < 28/40")
    if len(new["false_positive_videos"]) > 2:
        failures.append("more than two false positive videos")
    if new["typed_hits"] < old["typed_hits"] + 2:
        failures.append("fewer than two additional type-and-time matches")
    for row in candidate:
        old_row = old_by_video[row["video"]]
        old_wrong = typed_result(old_row.get("expected_type"), old_row["label_t"], old_row["violations"])["wrong_events"]
        for event in typed_result(row.get("expected_type"), row["label_t"], row["violations"])["wrong_events"]:
            match = next((e for e in old_wrong if e["type"] == event["type"] and abs(e["t"]-event["t"]) <= 3), None)
            if match is None:
                failures.append(f"new wrong-type event on {row['video']}: {event['type']}")
            else:
                old_wrong.remove(match)
    by_video = {r["video"]: r for r in candidate}
    for r in baseline:
        if r["typed"]["hits"] and not set(r["typed"]["hits"]) <= set(by_video[r["video"]]["typed"]["hits"]):
            failures.append(f"lost type-and-time match on {r['video']}")
    return failures
