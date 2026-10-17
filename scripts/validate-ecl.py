"""Reproducible four-way comparison from one fresh pose L1 cache. Input data is never modified.

DATA_DIR=/data python scripts/validate-ecl.py --pose-cache-tag <tag> --out /tmp/ecl-check
The baseline uses the exact same boxes with only attributes.pose removed. Both ECL models are retrained
with nested video folds. L1/L2 remain fixed upstream models; this does not estimate new-camera generalization.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-cache-tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", default="yolo", choices=["yolo", "rfdetr"])
    ap.add_argument("--code-commit")
    ap.add_argument("--data-commit")
    ap.add_argument("--l1-log", type=Path)
    args = ap.parse_args()
    source, out = Path(os.environ.get("DATA_DIR", "/data")).resolve(), args.out.resolve()
    staged = out / "data"
    staged.mkdir(parents=True, exist_ok=True)
    if source == staged:
        ap.error("--out/data must differ from source DATA_DIR")
    (staged / "labels.csv").write_bytes((source / "labels.csv").read_bytes())
    for name in ("videos", "runs"):
        link = staged / name
        if not link.exists():
            link.symlink_to((source / name).resolve())
    rows = [r for r in csv.DictReader((source / "labels.csv").open()) if not r["duplicate_of"]]
    provenance = []
    for row in rows:
        path = source / "features" / row["sha256"] / f"l1_{args.detector}@{args.pose_cache_tag}.json"
        payload = path.read_bytes()
        cache = json.loads(payload)
        folder = staged / "features" / row["sha256"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"l1_{args.detector}@pose.json").write_bytes(payload)
        posed = sum(bool(b.get("attributes", {}).get("pose")) for b in cache["boxes"])
        for box in cache["boxes"]:
            box.get("attributes", {}).pop("pose", None)
        (folder / f"l1_{args.detector}@baseline.json").write_text(json.dumps(cache))
        provenance.append({"video": row["video"], "sha256": row["sha256"],
                           "l1_sha256": hashlib.sha256(payload).hexdigest(), "pose_boxes": posed})
    env = {**os.environ, "DATA_DIR": str(staged), "VLM_URL": "", "POSE": "off", "ECL": "off"}
    env.pop("MODEL_DIR", None)
    env.pop("ECL_VIDEO", None)
    report = {"code_commit": args.code_commit, "data_commit": args.data_commit,
              "method": "same L1 boxes; nested video folds for ECL; fixed upstream L1/L2; VLM disabled",
              "input_cache_tag": args.pose_cache_tag, "inputs": provenance, "variants": {}, "calibration": {}}
    step_path = Path(env.get("STEP_MODEL") or source / "runs" / "sop" / f"steps_{args.detector}.joblib")
    report["configuration"] = {"tracker": env.get("TRACKER", "greedy"),
        "step_classifier": env.get("STEP_CLASSIFIER", "rules"), "step_model": str(step_path),
        "step_model_sha256": hashlib.sha256(step_path.read_bytes()).hexdigest() if step_path.is_file() else None,
        "sample_fps": env.get("SAMPLE_FPS", "4"), "pose": "from input cache"}

    def run(name, module, extra, variables=None):
        start = time.monotonic()
        with (out / f"{name}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", module, "--detector", args.detector, *extra],
                           env={**env, **(variables or {})}, stdout=log, stderr=subprocess.STDOUT, check=True)
        return round(time.monotonic() - start, 3)

    for tag in ("baseline", "pose"):
        model = out / f"ecl-{tag}.joblib"
        run(f"fit-{tag}", "decision.evidence", ["--cache-tag", tag, "--fit", "--out", str(model)])
        metrics = json.loads(model.with_suffix(".metrics.json").read_text())
        report["calibration"][tag] = {k: metrics[k] for k in ("method", "overall", "classes")}
        for ecl in (False, True):
            name = tag + ("-ecl" if ecl else "")
            elapsed = run(name, "decision.eval", ["--cache-tag", tag, "--out", str(out / f"{name}.json"),
                          "--summary", str(out / f"{name}-summary.json")],
                          {"ECL": "on" if ecl else "off", "ECL_MODEL": str(model)})
            result = json.loads((out / f"{name}-summary.json").read_text())
            details = json.loads((out / f"{name}.json").read_text())
            report["variants"][name] = {"summary": result,
                "false_positives": [r["video"] for r in details if r["label"] == "no_violation" and r["verdict"] == "violation"],
                "evaluation_wall_sec": elapsed}
            print(name, result["groups"]["all"], flush=True)
    if args.l1_log:
        log = args.l1_log.read_text()
        times = [(float(m[2]), m[1]) for m in re.finditer(r"^\s*(\S+):\s+[\d.]+ с видео .* за\s+([\d.]+) с$", log, re.M)]
        if len(times) != len(rows):
            raise ValueError("L1 timing log must cover every unique video")
        seconds, video = max(times)
        report["l1_pose_timing"] = {"videos": len(times), "total_sec": round(sum(t for t, _ in times), 1),
                                    "max_sec": seconds, "slowest_video": video, "gate_max_sec": 100,
                                    "within_time_gate": seconds <= 100}
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(out / "comparison.json")


if __name__ == "__main__":
    main()
