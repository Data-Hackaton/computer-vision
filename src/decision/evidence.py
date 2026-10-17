"""ECL — калибровка доказательств между L2 и L3: насколько верить каждому действию-кандидату, прежде чем применять
нормативные правила. L2 выдвигает гипотезу (действие), ECL оценивает её достоверность p, L3 решает последствия:
действие с p ниже порога класса получает certain: False и его собственные нарушения становятся предупреждениями
(data_prepare.sop_events), но шаги оно по-прежнему закрывает — пропуск детекции не превращается в «шага не было».

Признаки по группам: C — уверенность боксов участвующего класса; T — устойчивость сигнала во времени; G — геометрия
(место кассира, зоны, дубли треков); A — согласие (руки, присутствие клиента, новые треки); X — контекст (класс
действия, источник, время с прошлого такого же). Модель — StandardScaler + LogisticRegression, одна на все классы.

Разметка кандидатов — журналы actions.yaml / violations.yaml (24 видео): кандидат верен, если рядом (≤ MATCH_SEC или
внутри t..t_end) есть журнальное действие того же класса, один к одному. person_enter:visitor в журналах нет:
отрицательные — все кандидаты на чистых видео, положительные — X-18 и X-30 у секунды метки.
  python -m decision.evidence --detector yolo --fit          # LOO по видео, пороги, data/runs/sop/ecl_yolo.joblib
ECL=on + ECL_MODEL (из model.json.ecl) включают калибровку в analyze. ECL_VIDEO=X-1 выбирает внешний LOO-fold:
модель и пороги не используют это видео; пороги — по внутреннему LOO. Это проверка ECL при фиксированном L1/L2,
а не переобучение детектора или классификатора шагов на каждом fold.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from cv.detectors import DATA_DIR, REFERENCE_DIR
from cv.geometry import center_in, iou, overlap_share
from decision.learned import step_class
from decision.steps import HOLD_SEC

MONEY = {"banknotes", "id_card"}
MATCH_SEC = 2.0
VISITOR_MATCH_SEC = 5.0  # секунда метки организаторов приблизительна
VISITOR_POSITIVE = {"X-18", "X-30"}  # посторонний подтверждён проверкой; X-33, X-40 — спорные, не размечаем
WIN = 1.0  # окно признаков вокруг действия, с
RECALL_TARGET = 0.9  # порог класса — самый строгий, при котором LOO-полнота верных не ниже
# класс шага → (сигнал scene, значение после перехода, классы боксов)
SPEC = {
    "put:banknotes>desk": ("money_on_desk", 1, MONEY), "take:banknotes>desk": ("money_on_desk", 0, MONEY),
    "put:banknotes>cash_machine": ("money_in_machine", 1, MONEY), "take:banknotes>cash_machine": ("money_in_machine", 0, MONEY),
    "put:foreign_object>desk": ("foreign_on_desk", 1, {"foreign_object"}),
    "take:foreign_object>desk": ("foreign_on_desk", 0, {"foreign_object"}),
    "put:bag>desk": ("bag_on_desk", 1, {"bag"}), "take:bag>desk": ("bag_on_desk", 0, {"bag"}),
    "person_enter:client": ("client_present", 1, {"person", "reflection"}),
    "person_leave:client": ("client_present", 0, {"person", "reflection"}),
    "person_leave:cashier": ("cashier_present", 0, {"person", "reflection"}),
    "person_enter:visitor": ("two_at_cashier", 1, {"person", "reflection"}),
    "use_laptop": ("screen_on", 1, {"computer"}), "screen_lock": ("screen_on", 0, {"computer"}),
    "phone_call": ("phone_at_head", 1, {"phone"}), "split_bundle": ("bundles_at_cashier", 2, MONEY),
    "keep": ("_hand_hidden", 1, MONEY), "count_by_hand": ("_counting", 1, MONEY),
}
CLASSES = sorted(SPEC)
# Классы, которые размечающие записывали в журналы полностью: по ним «нет в журнале» значит «не было». Деньги, сумка,
# вещь на столе (put/take … desk) журналы пишут только когда это важно для сценария — там «нет в журнале» не значит
# «не было», обучаться на них нельзя: ECL научился бы не верить любому действию со столом. Такие классы ECL не трогает.
JOURNAL_CLASSES = {"person_enter:client", "person_leave:client", "person_leave:cashier", "put:banknotes>cash_machine",
                   "take:banknotes>cash_machine", "use_laptop", "screen_lock", "phone_call", "split_bundle",
                   "safe_open", "safe_close", "give:id_card>cashier", "give:banknotes>cashier", "inspect", "sort_banknotes"}
CALIBRATED = JOURNAL_CLASSES | {"person_enter:visitor"}


def context(frames, states, fps, zones, actions, duration) -> dict:
    """Что нужно признакам: сырые сигналы и боксы по кадрам (после треков и ролей), зоны, все действия."""
    return {"frames": frames, "states": states, "fps": fps, "zones": zones, "actions": actions,
            "duration": duration, "first_seen": _first_seen(frames)}


def _first_seen(frames) -> dict[int, float]:
    out: dict[int, float] = {}
    for t, bs in frames:
        for b in bs:
            if "track_id" in b and b["track_id"] not in out:
                out[b["track_id"]] = t
    return out


def _window(items, t, lo, hi):
    return [(tt, x) for tt, x in items if t + lo <= tt <= t + hi]


def features(a: dict, ctx: dict) -> dict[str, float]:
    cls = step_class(a)
    sig, target, box_cls = SPEC.get(cls, (None, None, set()))
    t, fps = a["t"], ctx["fps"]
    f: dict[str, float] = {f"cls_{c}": float(cls == c) for c in CLASSES}
    f["cls_other"] = float(cls not in SPEC)
    f["src_learned"] = float(a.get("source") == "learned")
    f["src_pose"] = float(a.get("source") == "pose")
    f["p_learned"] = float(a.get("p_learned") or 0.0)
    # C — боксы участвующего класса в окне
    win = _window(ctx["frames"], t, -WIN, WIN)
    boxes = [b for _, bs in win for b in bs if b["cls"] in box_cls]
    confs = [b.get("conf") or 0.0 for b in boxes]
    f["conf_mean"], f["conf_max"] = (float(np.mean(confs)), float(max(confs))) if confs else (0.0, 0.0)
    f["n_boxes"] = len(boxes) / max(1, len(win))
    # T — устойчивость сигнала: доля кадров со значением после перехода за время удержания и до него, длина серии
    hold = HOLD_SEC.get(sig, 1.0) if sig else 1.0
    states = ctx["states"]
    after = [st for st in states if t <= st["t"] <= t + hold]
    before = [st for st in states if t - hold <= st["t"] < t]
    if sig:
        f["sig_after"] = float(np.mean([int(st.get(sig) == target or (target == 2 and (st.get(sig) or 0) >= 2))
                                        for st in after])) if after else 0.0
        f["sig_before"] = float(np.mean([int(st.get(sig) != target) for st in before])) if before else 0.0
        run = 0
        for st in states:
            if st["t"] >= t:
                if st.get(sig) == target or (target == 2 and (st.get(sig) or 0) >= 2):
                    run += 1
                else:
                    break
        f["run_sec"] = min(10.0, run / fps)
    else:
        f["sig_after"] = f["sig_before"] = f["run_sec"] = 0.0
    dur = ctx["duration"] or (states[-1]["t"] if states else 0.0)
    f["edge_start"] = float(t < 2.0)
    f["edge_end"] = float(dur - t < 2.0)
    f["pos"] = t / dur if dur else 0.0
    # G/A — люди и треки в окне
    around = [st for st in states if t - WIN <= st["t"] <= t + WIN]
    f["client"] = float(np.mean([st["client_present"] for st in around])) if around else 0.0
    f["visitors"] = float(np.mean([st["visitors"] for st in around])) if around else 0.0
    f["two_at_cashier"] = float(np.mean([st["two_at_cashier"] for st in around])) if around else 0.0
    d = [st["_cashier_d"] for st in around if st.get("_cashier_d") is not None]
    f["cashier_d"] = float(np.mean(d)) if d else 0.0
    f["cashier_seen"] = float(bool(d))
    people = [(tt, b) for tt, bs in win for b in bs if b["cls"] in ("person", "reflection") and "track_id" in b]
    cashier_ids = {b["track_id"] for _, b in people if b.get("attributes", {}).get("role") == "cashier"}
    f["n_cashier_tracks"] = len(cashier_ids)
    f["n_person_tracks"] = len({b["track_id"] for _, b in people})
    f["refl_share"] = float(np.mean([b["cls"] == "reflection" for _, b in people])) if people else 0.0
    f["new_tracks"] = sum(1 for tid, t0 in ctx["first_seen"].items() if abs(t0 - t) <= 2 * WIN and t0 > 0.5)
    dup = 0.0
    for tt, bs in win:
        cs = [b["bbox"] for b in bs if b.get("attributes", {}).get("role") == "cashier"]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                dup = max(dup, iou(cs[i], cs[j]))
    f["dup_iou"] = dup
    # G/A — деньги: зоны и руки
    machine, table = ctx["zones"].get("cash_machine"), ctx["zones"].get("table")
    money = [(tt, b["bbox"]) for tt, bs in win for b in bs if b["cls"] in MONEY]
    f["money_machine"] = max((overlap_share(b, machine) for _, b in money), default=0.0) if machine else 0.0
    f["money_table"] = float(any(center_in(b, table) for _, b in money)) if table else 0.0
    held = 0
    for tt, bs in win:
        hands = [b["bbox"] for b in bs if b["cls"] == "hand"]
        if any(overlap_share(b["bbox"], h) > 0.2 for b in bs if b["cls"] in MONEY for h in hands):
            held += 1
    f["held_frac"] = held / max(1, len(win))
    # X — соседние действия
    same = [x["t"] for x in ctx["actions"] if x is not a and step_class(x) == cls and x["t"] < t]
    anyt = [x["t"] for x in ctx["actions"] if x is not a and x["t"] < t]
    f["dt_same"] = min(30.0, t - max(same)) if same else 30.0
    f["dt_any"] = min(30.0, t - max(anyt)) if anyt else 30.0
    return f


def vector(a: dict, ctx: dict, names: list[str]) -> list[float]:
    f = features(a, ctx)
    return [f.get(n, 0.0) for n in names]


# --- применение в analyze ---

def enabled() -> bool:
    return os.environ.get("ECL", "off") in ("on", "1")


def model_path() -> Path:
    return Path(os.environ.get("ECL_MODEL") or DATA_DIR / "runs" / "sop" / f"ecl_{os.environ.get('DETECTOR', 'yolo')}.joblib")


_cache: dict = {}


def load():
    import joblib

    path = model_path().resolve()
    try:
        st = path.stat()
    except OSError as e:
        raise RuntimeError(f"ECL включён, но модель недоступна: {path}") from e
    fingerprint = (str(path), st.st_mtime_ns, st.st_size)
    if _cache.get("fingerprint") != fingerprint:
        try:
            saved = joblib.load(path)
            validate(saved)
        except Exception as e:
            raise RuntimeError(f"Некорректная модель ECL: {path}: {e}; переобучите decision.evidence --fit") from e
        _cache.clear()
        _cache.update(fingerprint=fingerprint, saved=saved)
    return _cache["saved"]


def validate(saved: dict) -> None:
    if not isinstance(saved, dict) or saved.get("format_version") != 2:
        raise ValueError("ожидается формат ECL 2 с отдельными порогами outer folds")
    names = saved["features"]
    if not names or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("некорректные имена признаков")
    if set(saved["folds"]) != set(saved["training_videos"]):
        raise ValueError("нет outer fold для одного из обучающих видео")
    for part in [saved, *saved["folds"].values()]:
        if part.get("model") is None:
            if part is saved or not part.get("reason") or part.get("thresholds") != {}:
                raise ValueError("пропущенный fold должен иметь причину и не иметь порогов")
            continue
        model = part["model"]
        if list(model.classes_) != [0, 1] or model.n_features_in_ != len(names):
            raise ValueError("классы или число признаков модели не совпадают")
        prob = np.asarray(model.predict_proba(np.zeros((1, len(names)))))
        if prob.shape != (1, 2) or not np.isfinite(prob).all():
            raise ValueError("некорректный predict_proba")
        if any(c not in CALIBRATED or not np.isfinite(t) or not 0 <= t <= 1
               for c, t in part["thresholds"].items()):
            raise ValueError("некорректные пороги классов")


def select_fold(saved: dict, video: str | None) -> dict:
    # Неизвестное видео не участвовало в ECL fit; известное с непригодным fold НЕ получает full model.
    return saved["folds"][video] if video in saved["training_videos"] else saved


def apply(actions: list[dict], ctx: dict) -> None:
    """p каждому действию; ниже порога класса — certain: False. ECL_VIDEO — LOO-модель без этого видео."""
    saved = load()
    if not actions:
        return
    part = select_fold(saved, os.environ.get("ECL_VIDEO"))
    selected = [a for a in actions if step_class(a) in part["thresholds"]]
    if part["model"] is None or not selected:
        return
    x = np.array([vector(a, ctx, saved["features"]) for a in selected], dtype=float)
    p = part["model"].predict_proba(x)[:, 1]
    for a, pa in zip(selected, p):
        a["p"] = float(pa)
        if pa < part["thresholds"][step_class(a)]:
            a["certain"] = False


# --- обучение ---

def journals() -> dict:
    import yaml

    return {**yaml.safe_load((REFERENCE_DIR / "actions.yaml").read_text()),
            **yaml.safe_load((REFERENCE_DIR / "violations.yaml").read_text())}


def label(actions: list[dict], journal: list[dict] | None, verdict: str, label_t: list[float],
          video: str) -> list[int | None]:
    """Метка кандидата: 1 — есть журнальное действие того же класса рядом (один к одному), 0 — нет; None — класс вне
    журнала (не обучаемся). Посторонний: 0 на чистых видео, 1 у метки на подтверждённых X-18/X-30, иначе None."""
    out: list[int | None] = []
    used: set[int] = set()
    ref = sorted(journal or [], key=lambda a: a["t"])
    for a in actions:
        cls = step_class(a)
        if cls == "person_enter:visitor":
            if verdict == "no_violation":
                out.append(0)
            elif video in VISITOR_POSITIVE:
                out.append(int(any(abs(a["t"] - lt) <= VISITOR_MATCH_SEC for lt in label_t)))
            else:
                out.append(None)
            continue
        if journal is None or cls not in JOURNAL_CLASSES:
            out.append(None)
            continue
        best, best_d = None, None
        lo, hi = a["t"], a.get("t_seen", a["t"])  # уход кассира L2 сдвигает назад — к моменту, когда встал с места
        for j, r in enumerate(ref):
            if j in used or step_class(r) != cls:
                continue
            rs, re_ = r["t"] - 1, r.get("t_end", r["t"]) + 1
            d = 0.0 if rs <= hi and lo <= re_ else min(abs(r["t"] - lo), abs(r["t"] - hi))
            if d <= MATCH_SEC and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            used.add(best)
        out.append(int(best is not None))
    return out


def candidates(detector: str, cache_tag: str | None = None):
    """Кандидаты L2 (ECL выключен) с признаками и метками по всем видео с журналом и по чистым видео."""
    from cv.infer import cache_path
    from decision.steps import analyze

    with (DATA_DIR / "labels.csv").open() as f:
        rows = list(csv.DictReader(f))
    label_t = defaultdict(list)
    for r in rows:
        if r["verdict"] == "violation":
            label_t[r["duplicate_of"] or r["video"]].append(float(r["timecode_sec"]))
    rows = [r for r in rows if not r["duplicate_of"]]
    jr = journals()
    data = []
    for r in rows:
        name = r["video"]
        journal = [a for i in jr[name]["instances"] for a in i["actions"]] if name in jr else None
        if journal is None and r["verdict"] != "no_violation" and name not in VISITOR_POSITIVE:
            continue
        path = cache_path(r["sha256"], detector, cache_tag)
        if not path.exists():
            raise ValueError(f"Нет L1-кэша для обучающего видео {name}: {path}")
        cache = json.loads(path.read_text())
        res = analyze(cache["boxes"], cache["fps"], calibrate=False, collect_evidence=True)
        ctx, acts = res["ecl_context"], res["raw_actions"]
        ys = label(acts, journal, r["verdict"], label_t.get(name, []), name)
        for a, y in zip(acts, ys):
            if y is not None and not a["action"].startswith("_"):
                data.append({"video": name, "cls": step_class(a), "t": a["t"], "y": y, "f": features(a, ctx)})
    return data


def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def thresholds_from(p: np.ndarray, y: np.ndarray, cls: np.ndarray) -> dict[str, float]:
    """Максимальный наблюдаемый порог с полнотой ≥ RECALL_TARGET; при <5 положительных сохраняем все.
    Неуспешные folds (NaN) и классы без положительных не калибруются. Не округляем порог вверх."""
    out = {}
    for c in sorted(set(cls)):
        if c not in CALIBRATED:
            continue
        m = cls == c
        pos = np.sort(p[m & (y == 1) & np.isfinite(p)])
        if not len(pos):
            continue
        need = 1.0 if len(pos) < 5 else RECALL_TARGET
        required = math.ceil(need * len(pos))
        out[str(c)] = float(pos[len(pos) - required])
    return out


def out_of_fold(x, y, videos) -> tuple[np.ndarray, dict]:
    """Inner LOO: scaler и регрессия fit только на остальных видео."""
    p = np.full(len(y), np.nan)
    skipped = {}
    for v in sorted(set(videos)):
        train = videos != v
        if len(set(y[train])) < 2:
            skipped[str(v)] = "training split has fewer than two classes"
            continue
        model = make_model().fit(x[train], y[train])
        p[~train] = model.predict_proba(x[~train])[:, 1]
    return p, skipped


def score_metrics(y, p) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    valid = np.isfinite(p)
    return {"candidates": len(y), "positive": int(y.sum()), "scored": int(valid.sum()),
            "omitted": int((~valid).sum()),
            "auc": float(roc_auc_score(y[valid], p[valid])) if len(set(y[valid])) == 2 else None,
            "brier": float(brier_score_loss(y[valid], p[valid])) if valid.any() else None}


def train(data: list[dict]) -> dict:
    """Nested LOO проверяет и модель, и выбор порога без меток outer test video."""
    if not data or len({d["y"] for d in data}) < 2:
        raise ValueError("ECL требует размеченные кандидаты обоих классов")

    names = sorted({k for d in data for k in d["f"]})
    x = np.array([[d["f"].get(n, 0.0) for n in names] for d in data], dtype=float)
    y = np.array([d["y"] for d in data])
    cls = np.array([d["cls"] for d in data])
    videos = np.array([d["video"] for d in data])
    p, cutoffs = np.full(len(y), np.nan), np.full(len(y), np.nan)
    folds, fold_report = {}, {}
    for v in sorted(set(videos)):
        tr = videos != v
        if len(set(y[tr])) < 2:
            part = {"model": None, "thresholds": {}, "reason": "training split has fewer than two classes"}
            folds[str(v)] = part
            fold_report[str(v)] = {"status": "skipped", "reason": part["reason"]}
            continue
        inner, skipped = out_of_fold(x[tr], y[tr], videos[tr])
        thr = thresholds_from(inner, y[tr], cls[tr])
        model = make_model().fit(x[tr], y[tr])
        p[~tr] = model.predict_proba(x[~tr])[:, 1]
        cutoffs[~tr] = [thr.get(c, np.nan) for c in cls[~tr]]
        folds[str(v)] = {"model": model, "thresholds": thr}
        fold_report[str(v)] = {"status": "scored", "training_videos": sorted(set(videos[tr])),
                               "thresholds": thr, "inner_skipped": skipped}
    full_oof, skipped = out_of_fold(x, y, videos)
    thr = thresholds_from(full_oof, y, cls)
    report = {"method": "nested leave-one-video-out; fixed upstream L1/L2", "overall": score_metrics(y, p),
              "folds": fold_report, "full_threshold_inner_skipped": skipped, "classes": {}}
    for c in sorted(set(cls)):
        m = cls == c
        assessed = m & np.isfinite(p) & np.isfinite(cutoffs)
        keep = p >= cutoffs
        report["classes"][str(c)] = {**score_metrics(y[m], p[m]), "threshold_assessed": int(assessed.sum()),
                                    "positive_kept": int((assessed & keep & (y == 1)).sum()),
                                    "positive_rejected": int((assessed & ~keep & (y == 1)).sum()),
                                    "negative_rejected": int((assessed & ~keep & (y == 0)).sum())}
    return {"format_version": 2, "model": make_model().fit(x, y), "features": names, "thresholds": thr,
            "folds": folds, "training_videos": sorted(set(videos)), "evaluation": report}


def fit(detector: str, cache_tag: str | None, out: Path) -> None:
    import joblib

    saved = train(candidates(detector, cache_tag))
    saved["training_config"] = {"detector": detector, "cache_tag": cache_tag,
                                "tracker": os.environ.get("TRACKER", "greedy"),
                                "step_classifier": os.environ.get("STEP_CLASSIFIER", "rules")}
    validate(saved)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved, out)
    report = out.with_suffix(".metrics.json")
    report.write_text(json.dumps({**saved["evaluation"], "training_config": saved["training_config"]},
                                 ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(saved["evaluation"]["overall"], ensure_ascii=False))
    print(f"→ {out}\n→ {report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--cache-tag")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.fit:
        fit(args.detector, args.cache_tag, args.out or DATA_DIR / "runs" / "sop" / f"ecl_{args.detector}.joblib")


if __name__ == "__main__":
    main()
