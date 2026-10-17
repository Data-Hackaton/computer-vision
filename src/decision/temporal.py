"""Versioned L2 policy over existing detections. No video names, labels or reference timestamps.

People are stabilized before aggregation. Objects retain identity through missing observations,
but absence is never removal and a stale memory is never fresh evidence of a violation.
The fixed learned classifier still consumes legacy features (steps.analyze calls it first).
"""

import os
from collections import defaultdict
from copy import deepcopy

from cv.geometry import center_in, iou, median_box, overlap_share

POLICIES = ("legacy", "temporal_v2")


def policy() -> str:
    value = os.environ.get("DECISION_POLICY", "legacy")
    if value not in POLICIES:
        raise ValueError(f"DECISION_POLICY must be one of {POLICIES}, got {value!r}")
    return value


def parameters() -> tuple[float, float]:
    confirm = float(os.environ.get("DECISION_CONFIRM_SEC", "1"))
    gap = float(os.environ.get("DECISION_OCCLUSION_SEC", "2"))
    if confirm not in (.5, 1., 1.5) or gap not in (1., 2., 3.):
        raise ValueError("decision timing must belong to the validated 3×3 parameter grid")
    return confirm, gap


def stable_presence(values: list[bool], ts: list[float], fps: float, confirm: float, gap: float) -> list[bool]:
    """Bridge only bounded gaps; retain supported runs including runs ending at the video boundary."""
    out = list(values)
    seen = [i for i, value in enumerate(values) if value]
    for a, b in zip(seen, seen[1:]):
        if ts[b] - ts[a] - 1 / fps <= gap + 1e-6:
            out[a:b] = [True] * (b - a)
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and out[end] == out[start]:
            end += 1
        observations = [i for i in range(start, end) if values[i]]
        if out[start] and (len(observations) < 2 or ts[observations[-1]] - ts[observations[0]] + 1 / fps < confirm):
            out[start:end] = [False] * (end - start)
        start = end
    return out


def people_states(frames, roles, seat, states, fps, confirm, gap):
    tracks = defaultdict(list)
    for t, boxes in frames:
        for b in boxes:
            if b.get("track_id") in roles:
                tracks[b["track_id"]].append((t, b))
    table = median_box([b["bbox"] for _, boxes in frames for b in boxes if b["cls"] == "table"])
    def near_table(b):
        return not table or b[3] >= table[1] - .1*(table[3]-table[1])
    anchors = {}
    for tid, obs in tracks.items():
        candidates = [b["bbox"] for _, b in obs if near_table(b["bbox"])] or [b["bbox"] for _, b in obs]
        # The shorter near-table boxes represent seated service rather than the route to the door.
        candidates.sort(key=lambda b: b[3]-b[1])
        anchors[tid] = median_box(candidates[:max(1, len(candidates)//2)])
    # Stitch only non-overlapping fragments, never two simultaneously visible people.
    canonical = {tid: tid for tid in tracks}
    ordered = sorted(tracks, key=lambda tid: tracks[tid][0][0])
    for tid in ordered:
        start, first = tracks[tid][0]
        candidates = []
        for prev in ordered:
            end, last = tracks[prev][-1]
            if prev != tid and roles[prev] == roles[tid] and 0 < start - end <= gap:
                score = iou(first["bbox"], last["bbox"])
                if score >= .35:
                    candidates.append((score, prev))
        if candidates:
            canonical[tid] = canonical[max(candidates)[1]]
    raw = {tid: [False] * len(frames) for tid in set(canonical.values())}
    role_of = {canonical[tid]: role for tid, role in roles.items() if tid in canonical}
    for i, (t, boxes) in enumerate(frames):
        accepted = []
        for b in sorted(boxes, key=lambda b: -(b.get("conf") or 0)):
            tid = b.get("track_id")
            if tid not in canonical:
                continue
            cid = canonical[tid]
            # L1's glass mask labels real seated clients as reflection for long stretches.
            # An established person track remains usable; reflection-only tracks lack a role upstream.
            if any(iou(b["bbox"], other) > .35 or overlap_share(b["bbox"], other) > .6
                   or overlap_share(other, b["bbox"]) > .6 for other in accepted):
                continue
            accepted.append(b["bbox"])
            x1, y1, x2, y2 = b["bbox"]
            if roles[tid] == "cashier":
                far = bool(seat) and (((x1+x2)/2-seat["x"])**2+(y2-seat["y"])**2)**.5 > .6*seat["h"]
                active = near_table(b["bbox"]) and not (far and y2 < seat["y"] - .5*seat["h"])
            else:
                anchor = anchors[tid]
                distance = (((x1+x2-anchor[0]-anchor[2])/2)**2 + ((y1+y2-anchor[1]-anchor[3])/2)**2)**.5
                active = distance <= .6 * (anchor[3] - anchor[1])
                end = tracks[tid][-1][0]
                # Backdate a departure to rising only when this track actually disappears before
                # the clip ends; standing at the desk or a cropped ending alone is not departure.
                if (end < frames[-1][0]-.5 and 0 <= end-t <= 6 and y2-y1 > 1.4*(anchor[3]-anchor[1])
                        and distance > .25*(anchor[3]-anchor[1])):
                    active = False
            raw[cid][i] |= active
        # Fill an interior tracker gap only with a visible person/reflection at the same service
        # position. This does not extend a disappeared client beyond their last observed track.
        for b in boxes:
            if b.get("track_id") in roles or b["cls"] not in ("person", "reflection") or b.get("conf", 0) < .5:
                continue
            candidates = [(iou(b["bbox"], anchors[tid]), tid) for tid in tracks if roles[tid] == "client"
                          and tracks[tid][0][0] <= t <= tracks[tid][-1][0]]
            if candidates and max(candidates)[0] >= .35:
                raw[canonical[max(candidates)[1]]][i] = True
    ts = [t for t, _ in frames]
    stable = {tid: stable_presence(vals, ts, fps, confirm, gap) for tid, vals in raw.items()}
    service = 0
    was_client = False
    for i, st in enumerate(states):
        cash = [tid for tid, vals in stable.items() if vals[i] and role_of[tid] == "cashier"]
        clients = [tid for tid, vals in stable.items() if vals[i] and role_of[tid] == "client"]
        if clients and not was_client:
            service += 1
        was_client = bool(clients)
        st.update(cashier_present=int(bool(cash)), client_present=int(bool(clients)), visitors=len(clients),
                  two_at_cashier=int(len(cash) >= 2), _people=cash+clients, _service=service if clients else 0,
                  _last_service=service)


def object_actions(frames, states, zones, fps, confirm, gap, learned=()):
    table, machine, safe = (zones.get(k) for k in ("table", "cash_machine", "safe"))
    if table:
        w, h = table[2]-table[0], table[3]-table[1]
        table = [table[0]-.05*w, table[1]-.05*h, table[2]+.05*w, table[1]+.55*h]
    records = []
    for (t, boxes), state in zip(frames, states):
        used = set()
        hands = [b["bbox"] for b in boxes if b["cls"] == "hand"]
        for b in sorted((b for b in boxes if b["cls"] in ("banknotes", "id_card")), key=lambda b: -b["conf"]):
            bbox = b["bbox"]
            candidates = []
            for rec in records:
                if rec["id"] in used:
                    continue
                last = rec["samples"][-1]
                score = iou(bbox, last["bbox"])
                previous = last["bbox"]
                distance = (((bbox[0]+bbox[2]-previous[0]-previous[2])/2)**2
                            + ((bbox[1]+bbox[3]-previous[1]-previous[3])/2)**2)**.5
                scale = max(1, previous[2]-previous[0], previous[3]-previous[1])
                # A long gap can reconnect a stationary object only at the same location.
                limit = 15 if last["place"] == "desk" and score >= .6 else gap
                if t-last["t"] <= limit and (score >= .15 or t-last["t"] <= gap and distance <= 3*scale):
                    candidates.append((score if score >= .15 else .14-distance/(100*scale), rec["id"]))
            if candidates:
                rec = records[max(candidates)[1]]
            else:
                rec = {"id": len(records), "samples": []}
                records.append(rec)
            used.add(rec["id"])
            held = any(overlap_share(bbox, h) > .2 for h in hands)
            place = ("cash_machine" if machine and overlap_share(bbox, machine) > .3 else
                     "safe" if safe and overlap_share(bbox, safe) > .5 else
                     "held" if held else "desk" if table and center_in(bbox, table) else "unknown")
            rec["samples"].append({"t": t, "bbox": bbox, "cls": b["cls"], "place": place,
                                   "service": state["_service"] or state["_last_service"], "conf": b["conf"]})
    actions = []
    for rec in records:
        samples = rec["samples"]
        cash_times = [s["t"] for s in samples if s["cls"] == "banknotes"]
        receipts = [a for a in learned if a["action"] == "give" and a.get("object") == "banknotes"
                    and a.get("actor") == "client" and a.get("p_learned", 0) >= .9
                    and -.5 <= samples[0]["t"]-a["t"] <= 5]
        card_receipts = [a for a in learned if a.get("object") == "id_card" and abs(a["t"]-samples[0]["t"]) <= 3]
        context = bool(receipts and not card_receipts)
        if len(cash_times) < 2 and not context:
            continue  # a card requires independently recognized cash receipt and sustained object evidence
        cash_since = min([a["t"] for a in receipts]) if context else cash_times[0]
        for s in samples:
            s["cash"] = s["t"] >= cash_since
        current = None
        owner = None
        i = 0
        while i < len(samples):
            j = i + 1
            while j < len(samples) and samples[j]["place"] == samples[i]["place"] and samples[j]["t"]-samples[j-1]["t"] <= gap:
                j += 1
            run = samples[i:j]
            place, start = run[0]["place"], run[0]["t"]
            support = [s for s in run if s["cash"]]
            established = len(support) >= 2 and support[-1]["t"]-support[0]["t"]+1/fps >= confirm
            # A card observation cannot retroactively become money because cash appeared later.
            if established and place != "unknown" and place != current:
                if owner is None:
                    owner = support[0]["service"]
                proof = {"object_id": rec["id"], "owner_service": owner, "samples": samples,
                         "level": "corroborated" if context else "observed", "policy": "temporal_v2",
                         "receipt_times": [a["t"] for a in receipts] if context else []}
                if current in ("desk", "cash_machine", "safe"):
                    actions.append({"t": start, "action": "take", "actor": "cashier", "object": "banknotes",
                                    "target": current, "evidence": proof})
                if place in ("desk", "cash_machine", "safe"):
                    actions.append({"t": start, "action": "put", "actor": "cashier", "object": "banknotes",
                                    "target": place, "evidence": proof})
                current = place
            i = j
    return actions


def personal_item_actions(frames, zones, fps, confirm, gap):
    """Repeated stationary bag observations can survive isolated low confidence frames.

    Require prior association with a cashier and later separation from their body, so a carried
    customer/collection bag or clothes on a person do not become a personal item on the desk.
    """
    table = zones.get("table")
    if not table:
        return []
    w, h = table[2]-table[0], table[3]-table[1]
    desk = [table[0]-.05*w, table[1]-.05*h, table[2]+.05*w, table[1]+.55*h]
    tracks = []
    for t, boxes in frames:
        for b in boxes:
            if b["cls"] != "bag" or not center_in(b["bbox"], desk):
                continue
            same = [tr for tr in tracks if t-tr[-1]["t"] <= 4 and iou(b["bbox"], tr[-1]["bbox"]) >= .5]
            track = same[0] if same else []
            if not same:
                tracks.append(track)
            cash = [p["bbox"] for p in boxes if p.get("attributes", {}).get("role") == "cashier"]
            hands = [p["bbox"] for p in boxes if p["cls"] == "hand"]
            track.append({"t":t, "bbox":b["bbox"], "body_overlap":max([overlap_share(b["bbox"], p) for p in cash] or [0]),
                          "held":any(overlap_share(b["bbox"], h) > .2 for h in hands),
                          "client_present":any(p.get("attributes", {}).get("role") == "client" for p in boxes),
                          "conf":b["conf"]})
    out = []
    for tid, track in enumerate(tracks):
        prior = [s for s in track if s["body_overlap"] > .5]
        if not prior:
            continue
        separate = [s for s in track if s["t"] > prior[0]["t"] and not s["held"] and not s["client_present"]]
        if len(separate) >= 2 and separate[-1]["t"]-separate[0]["t"]+1/fps >= confirm:
            out.append({"t": separate[0]["t"], "action":"put", "actor":"cashier", "object":"foreign_object", "target":"desk",
                        "evidence":{"policy":"temporal_v2", "level":"observed", "item_id":tid, "samples":track}})
    return out


def machine_interactions(frames, zones, actions, learned, fps, confirm, gap):
    """Cash receipt + observed small object + sustained cashier hand at machine = corroborated loading.

    A hand at the machine alone does not prove counting. All three observations are retained.
    """
    machine = zones.get("cash_machine")
    if not machine:
        return []
    ts, raw, observations = [], [], []
    for t, boxes in frames:
        cash = [b["bbox"] for b in boxes if b.get("attributes", {}).get("role") == "cashier"]
        expanded = [[b[0]-.35*(b[3]-b[1]), b[1]-.35*(b[3]-b[1]), b[2]+.35*(b[3]-b[1]), b[3]] for b in cash]
        hands = [b["bbox"] for b in boxes if b["cls"] == "hand" and overlap_share(b["bbox"], machine) > .3
                 and any(center_in(b["bbox"], c) for c in expanded)]
        ts.append(t)
        raw.append(bool(hands))
        observations.append({"t":t, "hands":hands})
    stable = stable_presence(raw, ts, fps, confirm, gap)
    out = []
    for i, present in enumerate(stable):
        if not present or i and stable[i-1]:
            continue
        t = ts[i]
        receipts = [a for a in learned if a["action"] == "give" and a.get("object") == "banknotes"
                    and a.get("actor") == "client" and a.get("p_learned", 0) >= .9 and 0 <= t-a["t"] <= 10]
        cash = [a for a in actions if a.get("object") == "banknotes" and 0 <= t-a["t"] <= 10
                and a.get("evidence", {}).get("samples")]
        if not receipts or not cash or any(a.get("target") == "cash_machine" and abs(a["t"]-t) <= 3 for a in actions+out):
            continue
        if any(a.get("target") in ("safe", "client") and receipts[-1]["t"] < a["t"] < t for a in actions):
            continue
        out.append({"t":t, "action":"put", "actor":"cashier", "object":"banknotes", "target":"cash_machine",
                    "evidence":{"policy":"temporal_v2", "level":"corroborated", "receipt_times":[a["t"] for a in receipts],
                                "object_ids":[a["evidence"]["object_id"] for a in cash],
                                "hand_samples":[s for s in observations if t <= s["t"] <= t+confirm+gap and s["hands"]]}})
    return out


def journal(frames, roles, seat, raw_states, legacy_segments, learned, zones, fps):
    from decision.fsm import scene_actions
    from decision.scene import SIGNALS

    confirm, gap = parameters()
    states = deepcopy(raw_states)
    # Keep existing non-person signal processing; person aggregates come from one shared state.
    j = 0
    for st in states:
        while j+1 < len(legacy_segments) and st["t"] >= legacy_segments[j+1]["t_start"]:
            j += 1
        if legacy_segments:
            st.update(legacy_segments[j]["state"])
    people_states(frames, roles, seat, states, fps, confirm, gap)
    segs = []
    for st in states:
        value = {k: st[k] for k in SIGNALS}
        if segs and segs[-1]["state"] == value:
            segs[-1]["t_end"] = round(st["t"]+1/fps, 3)
        else:
            segs.append({"t_start": st["t"], "t_end": round(st["t"]+1/fps, 3), "state": value})
    actions = [a for a in scene_actions(segs) if a.get("object") != "banknotes"]
    actions += object_actions(frames, states, zones, fps, confirm, gap, learned)
    actions += machine_interactions(frames, zones, actions, learned, fps, confirm, gap)
    actions += personal_item_actions(frames, zones, fps, confirm, gap)
    actions += learned
    for n, a in enumerate(sorted(actions, key=lambda a: a["t"])):
        st = min(states, key=lambda st: abs(st["t"]-a["t"])) if states else {}
        a["action_id"] = f"a{n}"
        a["service_id"] = st.get("_service", 0)
        a.setdefault("evidence", {"policy": "temporal_v2", "level": "inferred" if a.get("source") == "learned" else "observed",
                                  "people": st.get("_people", []), "t": a["t"]})
    return states, segs, actions


def corroborate(event, action, actions):
    """Validate the premises of L3 hits, not just confidence in the triggering action."""
    if not action or action.get("evidence", {}).get("policy") != "temporal_v2" or event["status"] != "violation":
        return
    reasons = []
    t = event["t_start"]
    title = event.get("title", "")
    support_ids = [action["action_id"]]
    if title in ("Деньги предыдущего клиента на кассовом столе", "Ценности на столе при уходе кассира"):
        supports = []
        for a in actions:
            proof = a.get("evidence", {})
            if a["t"] > t or a["action"] != "put" or a.get("target") != "desk":
                continue
            if title.startswith("Деньги предыдущего") and not (0 < proof.get("owner_service", 0) < action.get("service_id", 0)):
                continue
            samples = [s for s in proof.get("samples", []) if s["t"] <= t]
            if samples and samples[-1]["place"] == "desk" and samples[-1].get("cash") and t-samples[-1]["t"] <= parameters()[1]:
                supports.append(a)
        if not supports:
            reasons.append("Нет свежего подтверждения наличных на столе нужного обслуживания")
        support_ids += [a["action_id"] for a in supports]
    if title == "Пересчёт наличных после ухода клиента":
        leaves = [a for a in actions if a["action"] == "person_leave" and a.get("actor") == "client" and a["t"] <= t]
        if not leaves or action.get("evidence", {}).get("level") not in ("observed", "corroborated"):
            reasons.append("Не подтверждена цепочка: уход клиента и последующий пересчёт")
        else:
            support_ids.append(leaves[-1]["action_id"])
    if title in ("Деньги в сейф без пересчёта на машинке", "Пропуск пересчёта денег из инкассаторской сумки перед хранением"):
        reasons.append("Отсутствие наблюдения пересчёта не доказывает его пропуск")
    event["evidence_actions"] = sorted(set(support_ids))
    if reasons:
        event["status"] = "warning"
        event["policy_reason"] = "; ".join(reasons)


def deduplicate(events):
    """Collapse only proven identical violation episodes, not unrelated expected steps or warnings."""
    out, previous = [], {}
    for e in sorted(events, key=lambda e: e["t_start"]):
        key = (e.get("title"), e.get("instance"), tuple(e.get("rules", [])))
        prior = previous.get(key)
        if (e["status"] == "violation" and e.get("evidence_actions") and prior
                and set(e["evidence_actions"]) & set(prior["evidence_actions"])
                and e["t_start"]-prior["t_end"] <= 3):
            prior["t_end"] = max(prior["t_end"], e["t_end"])
            prior["evidence_actions"] = sorted(set(prior["evidence_actions"]+e["evidence_actions"]))
        else:
            out.append(e)
            if e["status"] == "violation" and e.get("evidence_actions"):
                previous[key] = e
    return out
