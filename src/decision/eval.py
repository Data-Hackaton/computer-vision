"""Оценка L2+L3 на кэше L1 (python -m cv.infer): вердикт, попадание в размеченную секунду, seq_accuracy.

  python -m decision.eval --detector yolo [--videos X-1 X-4] [--verbose]
  MODEL_DIR=/data/runs/releases/<тег> python -m decision.eval --summary eval.json   # релиз модели (infra/model.sh)
Группы: dev — видео, кадры или журнал которых использовались при обучении детектора и настройке правил;
honest — остальные (val детектора и видео с нарушениями из violations_unreviewed.yaml).
Отчёт — stdout и data/runs/eval/sop_<детектор>[@<тег>][_<трекер>].json (--out); итоги по группам — --summary.
TRACKER=bytetrack|botsort — трекер людей (decision.tracking), --cache-tag — опытный кэш L1 (cv.infer --cache-tag).
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml

from cv.detectors import DATA_DIR, REFERENCE_DIR, Detection, apply_model_dir
from cv.infer import cache_path, detector_arg
from cv.pipeline import build_video_events, load_spec
from decision.sop import seq_accuracy
from decision.steps import analyze, with_vlm
from decision.quality import typed_result, typed_summary

CATCH_WINDOW_SEC = 3  # как в data_prepare.reference_actions
# Кадры этих видео — в train детектора (r1 train, r1b), журнал X-1 — образец для правил
DEV = {"X-3", "X-5", "X-8", "X-10", "X-12", "X-14", "X-16", "X-21", "X-23", "X-25", "X-29", "X-31", "X-35", "X-39",
       "X-43", "X-4", "X-34", "X-38", "X-40", "X-1"}
# Что L2 умеет видеть — по этим шагам считается seq_accuracy
VISIBLE = {("person_enter", "client"), ("person_leave", "client"), ("person_leave", "cashier"), ("put", "cash_machine"),
           ("take", "cash_machine"), ("put", "desk"), ("take", "desk"), ("split_bundle", None), ("phone_call", None)}

REVIEW_SHORT = {"supported": "доказано", "probable": "вероятно", "insufficient": "мало", "conditional_issue": "условно",
                "no_supported_violation": "чисто"}


def key(a: dict) -> tuple:
    return (a["action"], a.get("actor"), a.get("object"), a.get("target"))


def visible(a: dict) -> bool:
    if a["action"] in ("person_enter", "person_leave"):
        return (a["action"], a.get("actor")) in VISIBLE
    if a["action"] in ("put", "take"):
        return a.get("object") == "banknotes" and (a["action"], a.get("target")) in VISIBLE
    return (a["action"], None) in VISIBLE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["yolo", "rfdetr"], help="без MODEL_DIR обязателен")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--verbose", action="store_true", help="печатать действия L2 по каждому видео")
    ap.add_argument("--vlm", action="store_true", help="спорные моменты — VLM (VLM_URL, ответы кэшируются в data/features)")
    ap.add_argument("--out", type=Path, help="отчёт по видео; по умолчанию data/runs/eval/sop_<детектор>[@<тег>].json")
    ap.add_argument("--summary", type=Path, help="итоги по группам в JSON (ворота выкатки модели в infra/model.sh)")
    ap.add_argument("--cache-tag", help="опытный кэш l1_<детектор>@<метка>.json (cv.infer --cache-tag)")
    args = ap.parse_args()
    tag = apply_model_dir()
    detector = detector_arg(ap, tag, args.detector)
    cache_tag = tag or args.cache_tag
    tracker = os.environ.get("TRACKER", "greedy")

    with (DATA_DIR / "labels.csv").open() as f:
        rows = list(csv.DictReader(f))
    labels_t: dict[str, list[float]] = defaultdict(list)  # копии (X-47 = X-4) добавляют свои секунды
    sha_of = {r["video"]: r["sha256"] for r in rows}
    for r in rows:
        if r["verdict"] == "violation":
            labels_t[r["duplicate_of"] or r["video"]].append(float(r["timecode_sec"]))
    rows = [r for r in rows if not r["duplicate_of"] and (not args.videos or r["video"] in args.videos)]
    spec = load_spec()
    journals = {**yaml.safe_load((REFERENCE_DIR / "actions.yaml").read_text()),
                **yaml.safe_load((REFERENCE_DIR / "violations.yaml").read_text())}
    unreviewed = yaml.safe_load((REFERENCE_DIR / "violations_unreviewed.yaml").read_text())
    review_path = REFERENCE_DIR / "video_review.yaml"  # «Полная проверка ForteBank»: статус доказанности по видео
    review = yaml.safe_load(review_path.read_text()) if review_path.exists() else {}

    report = []
    for r in rows:
        name = r["video"]
        path = cache_path(sha_of[name], detector, cache_tag)
        if not path.exists():
            print(f"{name:>5}: нет кэша L1 ({path})")
            continue
        cache = json.loads(path.read_text())
        os.environ["ECL_VIDEO"] = name  # ECL: на видео с журналом — модель, обученная без него (LOO)
        res = analyze(cache["boxes"], cache["fps"])
        if args.vlm:
            from decision.vlm import VLM_MODEL, Asker

            ask = Asker(DATA_DIR / "videos" / f"{name}.mov", res["zones"],
                        cache_file=path.parent / f"vlm_{detector}_{VLM_MODEL}.json", max_calls=50)
            res = with_vlm(res, ask)
        events = build_video_events(Detection(detector, instances=res["instances"]), spec, name)
        viol = [e for e in events if e["status"] == "violation"]
        warn = [e for e in events if e["status"] == "warning"]
        verdict = "violation" if viol else "no_violation"
        hits = [t for t in labels_t.get(name, []) if any(abs(e["t_start"] - t) <= CATCH_WINDOW_SEC for e in viol)]
        seq = None
        if name in journals:
            ref = [key(a) for i in journals[name]["instances"] for a in sorted(i["actions"], key=lambda a: a["t"])
                   if visible(a)]
            seq = round(seq_accuracy([key(a) for a in res["actions"] if visible(a)], ref), 3)
        item = {"video": name, "group": "dev" if name in DEV else "honest", "label": r["verdict"],
                "label_t": labels_t.get(name, []), "verdict": verdict, "hits": hits, "seq_accuracy": seq,
                "expected_type": (unreviewed.get(name) or {}).get("type") or (journals.get(name, {}).get("type")),
                "review": (review.get(name) or {}).get("status"),
                "tracks": {role: sum(r == role for r in res["roles"].values()) for role in ("cashier", "client")},
                "violations": [{"t": e["t_start"], "title": e["title"]} for e in viol],
                "warnings": [{"t": e["t_start"], "title": e.get("title") or e["event"]} for e in warn]}
        report.append(item)
        item["typed"] = typed_result(item["expected_type"], item["label_t"], item["violations"])
        mark = "OK " if verdict == r["verdict"] else "ERR"
        found = "; ".join(f"{v['t']:.1f} {v['title']}" for v in item["violations"][:3])
        print(f"{mark} {name:>5} [{item['group']:>6}] {r['verdict']:>12} → {verdict:<12} "
              f"{REVIEW_SHORT.get(item['review'], '?'):<9} "
              f"попал {len(hits)}/{len(item['label_t'])} seq={seq} | {found}")
        if args.verbose:
            prev = {}
            for seg in res["segments"]:  # только изменившиеся сигналы
                diff = {k: v for k, v in seg["state"].items() if prev.get(k) != v}
                prev = seg["state"]
                if diff:
                    print(f"      [{seg['t_start']:6.1f}] " + " ".join(f"{k}={v}" for k, v in diff.items()))
            for a in res["actions"]:
                print(f"        {a['t']:6.1f} {a['action']} {a.get('actor', '')} {a.get('object', '')} {a.get('target', '')}")

    groups = {}
    for group in ("dev", "honest", "all"):
        part = [x for x in report if group == "all" or x["group"] == group]
        if not part:
            continue
        tp = sum(x["label"] == "violation" and x["verdict"] == "violation" for x in part)
        fp = sum(x["label"] == "no_violation" and x["verdict"] == "violation" for x in part)
        fn = sum(x["label"] == "violation" and x["verdict"] == "no_violation" for x in part)
        tn = sum(x["label"] == "no_violation" and x["verdict"] == "no_violation" for x in part)
        caught = sum(bool(x["hits"]) for x in part if x["label"] == "violation")
        seqs = [x["seq_accuracy"] for x in part if x["seq_accuracy"] is not None]
        seq = round(sum(seqs) / len(seqs), 3) if seqs else None
        groups[group] = {"videos": len(part), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "accuracy": round((tp + tn) / len(part), 3), "hits": caught, "seq_accuracy": seq}
        print(f"{group:>6}: видео {len(part)}, точность вердикта {(tp + tn) / len(part):.2f} "
              f"(TP {tp}, FP {fp}, FN {fn}, TN {tn}), таймкод ±{CATCH_WINDOW_SEC} с: {caught}/{tp + fn}"
              + (f", seq_accuracy {seq:.2f}" if seqs else ""))
    for status, short in REVIEW_SHORT.items():  # насколько видимы нарушения по степени доказанности в проверке
        part = [x for x in report if x["review"] == status]
        if part:
            print(f"проверка «{short}»: видео {len(part)}, модель нашла нарушение в "
                  f"{sum(x['verdict'] == 'violation' for x in part)}, таймкод ±{CATCH_WINDOW_SEC} с: "
                  f"{sum(bool(x['hits']) for x in part)}/{sum(x['label'] == 'violation' for x in part)}")
    out = args.out or DATA_DIR / "runs" / "eval" / (f"sop_{detector}{f'@{cache_tag}' if cache_tag else ''}"
                                                     f"{f'_{tracker}' if tracker != 'greedy' else ''}"
                                                     f"{'_vlm' if args.vlm else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"→ {out}")
    if args.summary:
        missing = len(rows) - len(report)  # видео без кэша L1: ворота не должны сравнивать неполные прогоны
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({"detector": detector, "model": tag, "missing": missing, "groups": groups,
                                           "decision_policy": os.environ.get("DECISION_POLICY", "legacy"),
                                           "typed_groups": {g: typed_summary([r for r in report if g == "all" or r["group"] == g])
                                                            for g in ("dev", "honest", "all")}},
                                           ensure_ascii=False, indent=1))
        print(f"→ {args.summary}")


if __name__ == "__main__":
    main()
