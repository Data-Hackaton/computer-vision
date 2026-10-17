"""Сравнение детекторов на val-видео одной метрикой.

Эталон — data/datasets/<dataset>/coco/valid/_annotations.coco.json. Для каждой модели: AP50 и AP50-95
по классам (pycocotools), путаница person ↔ reflection, скорость кадров/с на этой GPU.
Модели: --yolo <weights.pt> и/или --rfdetr <checkpoint.pth> (--rfdetr-size, --rfdetr-resolution).
Итог — таблица в stdout и data/runs/eval/<dataset>.json.
"""

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def yolo_predictor(weights: str, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(weights)

    def predict(path: Path):
        r = model.predict(str(path), imgsz=imgsz, conf=0.001, verbose=False, device=0)[0].boxes
        return r.xyxy.cpu().numpy(), r.cls.cpu().numpy().astype(int), r.conf.cpu().numpy()

    return predict


def rfdetr_predictor(checkpoint: str, size: str, resolution: int):
    from cv.train_rfdetr import MODELS

    model = MODELS[size](pretrain_weights=checkpoint, resolution=resolution)

    def predict(path: Path):
        d = model.predict(Image.open(path).convert("RGB"), threshold=0.001)
        return d.xyxy, d.class_id.astype(int), d.confidence

    return predict


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def evaluate(name: str, predict, gt: COCO, image_dir: Path, class_names: list[str]) -> dict:
    images = gt.loadImgs(gt.getImgIds())
    predict(image_dir / images[0]["file_name"])  # прогрев
    dets, confusion = [], {"person→reflection": 0, "reflection→person": 0}
    started = time.time()
    for im in images:
        boxes, classes, scores = predict(image_dir / im["file_name"])
        for b, c, s in zip(boxes, classes, scores):
            dets.append({"image_id": im["id"], "category_id": int(c),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s)})
        # путаница: уверенный бокс одного класса поверх эталонного бокса другого
        anns = gt.loadAnns(gt.getAnnIds(imgIds=im["id"]))
        for src, dst in (("person", "reflection"), ("reflection", "person")):
            gt_boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                                 for a in anns if a["category_id"] == class_names.index(src)]).reshape(-1, 4)
            pred = boxes[(classes == class_names.index(dst)) & (scores >= 0.3)]
            if len(gt_boxes) and len(pred):
                confusion[f"{src}→{dst}"] += int((iou(gt_boxes, pred).max(axis=1) >= 0.5).sum())
    fps = len(images) / (time.time() - started)

    per_class = {}
    if dets:
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, gt.loadRes(dets), "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        precision = ev.eval["precision"]  # [IoU, recall, class, area, maxDets]
        for k, cat_id in enumerate(ev.params.catIds):
            p_all, p50 = precision[:, :, k, 0, 2], precision[0, :, k, 0, 2]
            per_class[class_names[cat_id]] = {
                "ap50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None,
                "ap50_95": float(p_all[p_all > -1].mean()) if (p_all > -1).any() else None,
                "gt": len(gt.getAnnIds(catIds=[cat_id])),
            }
        summary = {"ap50": float(ev.stats[1]), "ap50_95": float(ev.stats[0])}
    else:
        summary = {"ap50": 0.0, "ap50_95": 0.0}
    return {"model": name, **summary, "fps": round(fps, 1), "per_class": per_class, "confusion": confusion}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    ap.add_argument("--dataset", default="det_r1")
    ap.add_argument("--yolo", help="веса YOLO .pt")
    ap.add_argument("--yolo-imgsz", type=int, default=864)
    ap.add_argument("--rfdetr", help="чекпойнт RF-DETR .pth")
    ap.add_argument("--rfdetr-size", default="small")
    ap.add_argument("--rfdetr-resolution", type=int, default=640)
    args = ap.parse_args()

    valid = args.data / "datasets" / args.dataset / "coco" / "valid"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(valid / "_annotations.coco.json"))
    class_names = [c["name"] for c in sorted(gt.loadCats(gt.getCatIds()), key=lambda c: c["id"])]

    results = []
    if args.yolo:
        results.append(evaluate(f"yolo:{Path(args.yolo).parent.parent.name}", yolo_predictor(args.yolo, args.yolo_imgsz),
                                gt, valid, class_names))
    if args.rfdetr:
        results.append(evaluate(f"rfdetr-{args.rfdetr_size}:{Path(args.rfdetr).parent.name}",
                                rfdetr_predictor(args.rfdetr, args.rfdetr_size, args.rfdetr_resolution),
                                gt, valid, class_names))
    if not results:
        raise SystemExit("нужна хотя бы одна модель: --yolo и/или --rfdetr")

    print(f"{'класс':<14}{'gt':>5}" + "".join(f"{r['model'][:24]:>26}" for r in results))
    for name in class_names:
        n_gt = results[0]["per_class"].get(name, {}).get("gt", 0)
        cells = []
        for r in results:
            c = r["per_class"].get(name, {})
            cells.append(f"{c['ap50']:.2f} / {c['ap50_95']:.2f}" if c.get("ap50") is not None else "—")
        print(f"{name:<14}{n_gt:>5}" + "".join(f"{cell:>26}" for cell in cells))
    for r in results:
        print(f"{r['model']}: AP50 {r['ap50']:.3f}, AP50-95 {r['ap50_95']:.3f}, {r['fps']} кадров/с, путаница {r['confusion']}")

    out = args.data / "runs" / "eval" / f"{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
