"""Обучение RF-DETR (Apache-2.0) на data/datasets/<name>/coco → data/runs/rfdetr/<run>.

Вход модели квадратный; разрешение должно делиться на 32 (patch 16 × 2 окна). Кадр 464×848 сжимается
до квадрата, поэтому для мелких рук и удостоверений берём разрешение выше штатного (small: 512 → 640).
Память GTX 1650 маленькая: batch 2 и накопление градиента до эффективных 16.
"""

import argparse
import os
from pathlib import Path

from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall

MODELS = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    ap.add_argument("--dataset", default="det_r1")
    ap.add_argument("--size", choices=list(MODELS), default="small")
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--run", default=None, help="имя прогона, по умолчанию <dataset>-rfdetr-<size>")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true", help="если в fp16 пошли NaN")
    args = ap.parse_args()
    if args.resolution % 32:
        raise SystemExit("resolution должно делиться на 32")

    weights_dir = Path(os.environ.get("TORCH_HOME", "/cache/torch")) / "rfdetr"
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_dir)  # туда RF-DETR скачивает предобученные веса

    run = args.run or f"{args.dataset}-rfdetr-{args.size}"
    output_dir = args.data / "runs" / "rfdetr" / run
    model = MODELS[args.size](resolution=args.resolution, **({"amp": False} if args.no_amp else {}))
    model.train(
        dataset_dir=str(args.data / "datasets" / args.dataset / "coco"),
        output_dir=str(output_dir),
        epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.accum, lr=1e-4,
        early_stopping=True, early_stopping_patience=10, num_workers=4, tensorboard=False,
        # в fp16 на GTX 1650 оценки детекций на валидации уходят в NaN (17.10): --no-amp выключает и fp16-оценку
        **({"fp16_eval": False} if args.no_amp else {}),
    )
    print(f"→ {output_dir}")


if __name__ == "__main__":
    main()
