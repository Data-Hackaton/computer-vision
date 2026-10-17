"""Обучение Ultralytics YOLO на data/datasets/<name>/yolo → data/runs/yolo/<run>.

Лицензия Ultralytics — AGPL-3.0: учим для сравнения с RF-DETR (Apache-2.0), выбор — по eval.
На GTX 1650 Ultralytics сам выключает AMP (известные NaN), поэтому fp32 и batch подбирается под память.
Аугментации поворотом, перспективой и отражением — под другой ракурс финального видео.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    ap.add_argument("--dataset", default="det_r1")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--run", default=None, help="имя прогона, по умолчанию <dataset>-<model>")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=864, help="родная высота кадра 848 — мелочь не ужимается")
    ap.add_argument("--batch", type=float, default=-1, help="-1 — подобрать по памяти, 0.x — доля памяти, N — штук")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    weights_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/cache/ultralytics")) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_dir)  # туда Ultralytics скачивает предобученные веса

    run = args.run or f"{args.dataset}-{Path(args.model).stem}"
    model = YOLO(args.model)
    model.train(
        data=str(args.data / "datasets" / args.dataset / "yolo" / "data.yaml"),
        project=str(args.data / "runs" / "yolo"), name=run, exist_ok=True,
        epochs=args.epochs, imgsz=args.imgsz, batch=int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch,
        patience=args.patience, workers=args.workers, device=0, seed=0,
        degrees=10.0, perspective=0.0005, fliplr=0.5, scale=0.5, mosaic=1.0, close_mosaic=10,
        hsv_v=0.5,  # в X-12 гасят свет — кадры почти тёмные
        plots=True,
    )
    print(f"→ {args.data / 'runs' / 'yolo' / run}")


if __name__ == "__main__":
    main()
