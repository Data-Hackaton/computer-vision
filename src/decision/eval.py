"""Оценка L2+L3 на кэше L1 (python -m cv.infer): вердикт, попадание в размеченную секунду, seq_accuracy.

  python -m decision.eval --detector yolo [--videos X-1 X-4] [--verbose]
Группы: dev — видео, кадры или журнал которых использовались при обучении детектора и настройке правил;
honest — остальные (val детектора и видео с нарушениями из violations_unreviewed.yaml).
Отчёт — stdout и data/runs/eval/sop_<детектор>.json.
"""

import argparse
import csv
import json
from collections import defaultdict

import yaml

from cv.detectors import DATA_DIR, REFERENCE_DIR, Detection
from cv.infer import cache_path
from cv.pipeline import build_video_events, load_spec
from decision.sop import seq_accuracy
from decision.steps import analyze

CATCH_WINDOW_SEC = 3  # как в data_prepare.reference_actions
# Кадры этих видео — в train детектора (r1 train, r1b), журнал X-1 — образец для правил
DEV = {"X-3", "X-5", "X-8", "X-10", "X-12", "X-14", "X-16", "X-21", "X-23", "X-25", "X-29", "X-31", "X-35", "X-39",
       "X-43", "X-4", "X-34", "X-38", "X-40", "X-1"}
# Что L2 умеет видеть — по этим шагам считается seq_accuracy
VISIBLE = {("person_enter", "client"), ("person_leave", "client"), ("person_leave", "cashier"), ("put", "cash_machine"),
           ("take", "cash_machine"), ("put", "desk"), ("take", "desk"), ("split_bundle", None), ("phone_call", None)}


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
    ap.add_argument("--detector", required=True, choices=["yolo", "rfdetr"])
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--verbose", action="store_true", help="печатать действия L2 по каждому видео")
    args = ap.parse_args()

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

    report = []
    for r in rows:
        name = r["video"]
        path = cache_path(sha_of[name], args.detector)
        if not path.exists():
            print(f"{name:>5}: нет кэша L1 ({path})")
            continue
        cache = json.loads(path.read_text())
        res = analyze(cache["boxes"], cache["fps"])
        events = build_video_events(Detection(args.detector, instances=res["instances"]), spec, name)
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
                "violations": [{"t": e["t_start"], "title": e["title"]} for e in viol],
                "warnings": [{"t": e["t_start"], "title": e.get("title") or e["event"]} for e in warn]}
        report.append(item)
        mark = "OK " if verdict == r["verdict"] else "ERR"
        found = "; ".join(f"{v['t']:.1f} {v['title']}" for v in item["violations"][:3])
        print(f"{mark} {name:>5} [{item['group']:>6}] {r['verdict']:>12} → {verdict:<12} "
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
        print(f"{group:>6}: видео {len(part)}, точность вердикта {(tp + tn) / len(part):.2f} "
              f"(TP {tp}, FP {fp}, FN {fn}, TN {tn}), таймкод ±{CATCH_WINDOW_SEC} с: {caught}/{tp + fn}"
              + (f", seq_accuracy {sum(seqs) / len(seqs):.2f}" if seqs else ""))
    out = DATA_DIR / "runs" / "eval" / f"sop_{args.detector}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
