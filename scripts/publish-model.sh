#!/usr/bin/env bash
# Публикация обученной модели — GitHub Release model-<детектор>-<прогон>-<ГГГГММДД> в Data-Hackaton/computer-vision.
# Публикация релиза запускает выкатку на k3s-lab: .github/workflows/model-deploy.yml → infra/model.sh deploy <тег>
# (оценка на 40 видео, сравнение с текущей моделью, переключение cv-worker, откат при сбое).
#   scripts/publish-model.sh yolo ../data_prepare/data/runs/yolo/det_l1-yolo26s \
#       --steps ../data_prepare/data/runs/sop/steps_yolo.joblib
#   --steps <joblib>  классификатор шагов, обученный на L1 этих весов (decision.learned --target steps --fit);
#                     без него cv-worker с этим релизом работает только на правилах
#   --conf <порог>    DETECTOR_CONF (по умолчанию 0.35); --weights <файл> — веса не по умолчанию для прогона
#   --tag <тег>       свой тег вместо model-<детектор>-<прогон>-<дата>
#   --draft           черновик релиза: на сервер не выкатывается, пока его не опубликуют
#   --out <каталог>   только собрать релиз (model.json и файлы), ничего не публиковать
# Обучение — README.md, раздел «Обучение». Нужны gh (вход в GitHub) и python3 на хосте.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 2 ] || usage
detector=$1 run_dir=${2%/}
shift 2
steps='' conf=0.35 weights='' tag='' draft='' out='' build_only=''
while [ $# -gt 0 ]; do
  case $1 in
    --steps) steps=$2; shift 2 ;;
    --conf) conf=$2; shift 2 ;;
    --weights) weights=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --draft) draft=--draft; shift ;;
    --out) out=$2; build_only=1; shift 2 ;;
    *) usage ;;
  esac
done

case $detector in
  yolo) weights=${weights:-$run_dir/weights/best.pt} ;;
  rfdetr) weights=${weights:-$run_dir/checkpoint_best_total.pth} ;;
  *) echo "детектор: yolo или rfdetr"; exit 1 ;;
esac
[ -f "$weights" ] || { echo "нет весов $weights"; exit 1; }
[ -z "$steps" ] || [ -f "$steps" ] || { echo "нет классификатора шагов $steps"; exit 1; }
run=$(basename "$run_dir")
tag=${tag:-model-$detector-$run-$(date +%Y%m%d)}
[[ $tag =~ ^model-[A-Za-z0-9._-]+$ ]] || { echo "тег должен быть вида model-…: $tag"; exit 1; }

if [ -z "$out" ]; then
  out=$(mktemp -d)
  trap 'rm -rf "$out"' EXIT
fi
mkdir -p "$out"
cp "$weights" "$out/"
[ -z "$steps" ] || cp "$steps" "$out/steps.joblib"

# Манифест: что читает cv.detectors.apply_model_dir, плюс происхождение — коммиты кода и данных, метрики обучения
python3 - "$out" "$tag" "$detector" "$(basename "$weights")" "${steps:+steps.joblib}" "$conf" "$run_dir" \
  "$(git -C "$repo" rev-parse HEAD)" "$(git -C "$repo/../data_prepare" rev-parse HEAD 2>/dev/null || true)" <<'EOF'
import csv, hashlib, json, sys, time
from pathlib import Path

out, tag, detector, weights, steps, conf, run_dir, code, data = sys.argv[1:]
out, run_dir = Path(out), Path(run_dir)


def sha256(name):
    h = hashlib.sha256()
    with (out / name).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_metrics():
    """Лучшая эпоха на val прогона: results.csv (Ultralytics) или metrics.csv (RF-DETR). Справочно — ворота
    выкатки считают свои метрики на 40 видео."""
    for name, m50, m5095 in (("results.csv", "metrics/mAP50(B)", "metrics/mAP50-95(B)"),
                             ("metrics.csv", "val/mAP_50", "val/mAP_50_95")):
        path = run_dir / name
        if not path.exists():
            continue
        rows = [r for r in csv.DictReader(path.open()) if (r.get(m5095) or "").strip()]
        if rows:
            best = max(rows, key=lambda r: float(r[m5095]))
            return {"source": name, "epoch": int(float(best["epoch"])), "map50": round(float(best[m50]), 4),
                    "map50_95": round(float(best[m5095]), 4)}
    return None


manifest = {"tag": tag, "detector": detector, "run": run_dir.name,
            "weights": {"file": weights, "sha256": sha256(weights)},
            "steps": {"file": steps, "sha256": sha256(steps)} if steps else None,
            "params": {"conf": float(conf)}, "code_commit": code, "data_commit": data or None,
            "train_metrics": train_metrics(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
(out / "model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
print(json.dumps(manifest, ensure_ascii=False, indent=1))
EOF

[ -z "$build_only" ] || { echo "собрано в $out, не публикую"; exit 0; }
notes="Модель $detector, прогон $run, порог $conf$([ -n "$steps" ] && echo ", с классификатором шагов" || echo ", L2 только правила").
Выкатка на k3s-lab — workflow model-deploy (оценка, сравнение с текущей, откат при сбое)."
gh release create "$tag" -R Data-Hackaton/computer-vision --target main --title "$tag" --notes "$notes" $draft \
  "$out"/*
echo "опубликован $tag${draft:+ (черновик)}"
