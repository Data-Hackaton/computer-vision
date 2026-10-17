"""Предразметка кадров Grounding DINO → COCO для импорта в CVAT.

data/frames/<round>/images/*.jpg → data/frames/<round>/prelabels.coco.json (+ превью с боксами).
Модель open-vocabulary, поэтому классы задаются фразами. Людей в отражении она не отличает:
все люди уходят как `person`, валидатор в CVAT меняет метку на `reflection`.
Порядок категорий совпадает с data_prepare/cvat/labels.json.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # Apache-2.0

# класс → (фразы для модели, порог, сколько боксов максимум на кадр)
CLASSES: dict[str, tuple[list[str], float, int]] = {
    "person": (["person"], 0.35, 6),
    "reflection": (["reflection of a person in the glass"], 0.35, 6),
    "hand": (["hand"], 0.25, 8),
    "banknotes": (["banknotes", "stack of money"], 0.30, 8),
    "id_card": (["id card"], 0.35, 2),
    "cash_machine": (["banknote counting machine"], 0.30, 1),
    "safe": (["grey metal safe"], 0.30, 1),
    "door": (["glass door"], 0.30, 1),
    "phone": (["mobile phone"], 0.45, 2),  # стационарный телефон на столе даёт 0.31–0.40
    "bag": (["green cloth bag"], 0.45, 3),  # ниже 0.45 — посторонние предметы в каждом кадре
    "table": (["wooden office desk"], 0.30, 1),
    "foreign_object": (["jacket", "coat", "handbag", "backpack", "cup"], 0.35, 4),
    "computer": (["laptop", "computer monitor"], 0.30, 2),
    "ignore_zone": ([], 1.0, 0),  # из отдельного прохода по стеклу, см. video_zone
}
# Зона игнорирования — стекло, в котором отражаются люди и предметы. Камера на видео неподвижна, поэтому зона
# ищется на нескольких кадрах видео и берётся медиана. Внутри зоны мелкие предметы не на кассовом столе
# выбрасываются, а люди целиком внутри и не дотягивающие до её низа (не в дверном проёме) — это reflection.
REGION_PHRASES = ["glass wall", "glass window"]
REGION_FRAMES = 6
SMALL_IN_ZONE = {"hand", "banknotes", "id_card", "phone", "bag", "foreign_object", "computer"}
# Атрибуты CVAT, которые следуют из фразы
PHRASE_ATTRS = {
    "jacket": {"kind": "clothing"}, "coat": {"kind": "clothing"},
    "handbag": {"kind": "personal_bag"}, "backpack": {"kind": "personal_bag"}, "cup": {"kind": "cup"},
    "laptop": {"kind": "laptop"}, "computer monitor": {"kind": "monitor"},
}
# Приманки: модель присваивает похожие объекты этим фразам, а не классам; такие боксы выбрасываются.
# «white table» — белый стол на переднем плане, это не кассовый стол
DECOYS = ["white drawer cabinet", "desk telephone", "white table"]
# Максимальная площадь бокса в долях кадра: «grey metal safe» иначе цепляется за белый стол на переднем плане
MAX_AREA = {"safe": 0.15, "cash_machine": 0.15, "bag": 0.10, "banknotes": 0.03, "id_card": 0.02, "phone": 0.02,
            "hand": 0.02, "table": 0.5, "foreign_object": 0.10, "computer": 0.08}
PHRASE_TO_CLASS: dict[str, str | None] = {p: c for c, (phrases, _, _) in CLASSES.items() for p in phrases}
PHRASE_TO_CLASS.update({p: None for p in DECOYS})
# person и reflection подавляются друг другом: один и тот же человек не бывает и тем и другим
NMS_GROUP = {"person": "person", "reflection": "person"}
COLORS = ["#e6194b", "#808080", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
          "#9a6324", "#fabebe", "#008080", "#000075"]


def match_phrase(text: str) -> str | None:
    """Фраза из словаря, к которой относится ответ модели; None — нет или неоднозначно."""
    text = text.strip().lower()
    if text in PHRASE_TO_CLASS:
        return text
    # модель может вернуть часть фразы («money», «counting machine»); неоднозначный кусок («glass») — выбросить
    candidates = [phrase for phrase in PHRASE_TO_CLASS if text and (text in phrase or phrase in text)]
    if len({PHRASE_TO_CLASS[p] for p in candidates}) != 1:
        return None
    # все кандидаты одного класса; если это приманки (класс None) — вернётся приманка
    return next((p for p in candidates if PHRASE_TO_CLASS[p]), candidates[0])


def phrase_class(text: str) -> str | None:
    phrase = match_phrase(text)
    return PHRASE_TO_CLASS[phrase] if phrase else None


def refilter(path: Path) -> None:
    """Заново применить пороги и лимиты CLASSES к готовой предразметке, без модели."""
    coco = json.loads(path.read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    kept, per_image = [], {}
    for ann in sorted(coco["annotations"], key=lambda a: -a["score"]):
        name = names[ann["category_id"]]
        _, threshold, limit = CLASSES[name]
        key = (ann["image_id"], name)
        if ann["score"] < threshold or per_image.get(key, 0) >= limit:
            continue
        per_image[key] = per_image.get(key, 0) + 1
        kept.append({**ann, "id": len(kept) + 1})
    print(f"{path}: {len(coco['annotations'])} → {len(kept)} боксов")
    coco["annotations"] = kept
    path.write_text(json.dumps(coco))


def largest_region(model, processor, image: Image.Image, device: str, dtype) -> list[float] | None:
    """Самый большой бокс стекла на кадре (не меньше 5% площади) или None."""
    w, h = image.size
    inputs = processor(images=image, text=[REGION_PHRASES], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[(h, w)])[0]
    boxes = [b.tolist() for b in result["boxes"] if (b[2] - b[0]) * (b[3] - b[1]) / (w * h) >= 0.05]
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)


def video_zone(model, processor, paths: list[Path], device: str, dtype) -> list[float] | None:
    """Медианный бокс стекла по REGION_FRAMES кадрам видео."""
    sample = paths[:: max(1, len(paths) // REGION_FRAMES)][:REGION_FRAMES]
    found = [b for p in sample if (b := largest_region(model, processor, Image.open(p).convert("RGB"), device, dtype))]
    if not found:
        return None
    return [float(torch.tensor([b[k] for b in found]).median()) for k in range(4)]


def center_in(b: list[float], zone: list[float]) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def inside(b: list[float], zone: list[float], tol: float = 4.0) -> bool:
    return b[0] >= zone[0] - tol and b[1] >= zone[1] - tol and b[2] <= zone[2] + tol and b[3] <= zone[3] + tol


def intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def overlap_share(a: list[float], b: list[float]) -> float:
    """Доля площади a, лежащая внутри b."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    ap.add_argument("--round", default="r1")
    ap.add_argument("--limit", type=int, help="только первые N кадров (проверка)")
    ap.add_argument("--preview", type=int, default=20, help="сколько кадров отрисовать с боксами")
    ap.add_argument("--refilter", action="store_true", help="только переприменить пороги к prelabels.coco.json")
    ap.add_argument("--short-edge", type=int, default=640, help="короткая сторона входа модели; 800 по умолчанию у модели не влезает в 4 ГБ")
    ap.add_argument("--glob", nargs="*", default=["*.jpg"], help="какие кадры брать, например 'X-5_*' 'X-35_*'")
    ap.add_argument("--out", default="prelabels.coco.json", help="имя файла в data/frames/<round>; превью — в preview_<имя>")
    args = ap.parse_args()

    root = args.data / "frames" / args.round
    if args.refilter:
        refilter(root / args.out)
        return
    images = sorted({p for g in args.glob for p in (root / "images").glob(g)})[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.size = {"shortest_edge": args.short_edge, "longest_edge": round(args.short_edge * 1.83)}
    dtype = torch.float32  # в fp16 Grounding DINO выдаёт NaN
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    phrases = [p for p in PHRASE_TO_CLASS]
    class_names = list(CLASSES)

    coco = {
        "images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(class_names)],
    }
    preview_dir = root / ("preview" if args.out == "prelabels.coco.json" else f"preview_{Path(args.out).stem}")
    preview_dir.mkdir(exist_ok=True)
    started = time.time()

    by_video: dict[str, list[Path]] = {}
    for p in images:
        by_video.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    zones = {v: video_zone(model, processor, paths, device, dtype) for v, paths in by_video.items()}
    print(f"зоны стекла: {sum(z is not None for z in zones.values())}/{len(zones)} видео, {time.time() - started:.0f} с")

    for img_id, path in enumerate(images, start=1):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        zone = zones.get(path.stem.rsplit("_", 1)[0])
        coco["images"].append({"id": img_id, "file_name": path.name, "width": w, "height": h})

        inputs = processor(images=image, text=[phrases], return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=min(t for _, t, n in CLASSES.values() if n),
            text_threshold=0.2, target_sizes=[(h, w)],
        )[0]
        texts = result.get("text_labels", result.get("labels"))

        keep = []
        for box, score, text in zip(result["boxes"], result["scores"], texts):
            phrase = match_phrase(text)
            cls = PHRASE_TO_CLASS[phrase] if phrase else None
            if cls is None or score < CLASSES[cls][1]:
                continue
            x1, y1, x2, y2 = box.tolist()
            if (x2 - x1) * (y2 - y1) / (w * h) > MAX_AREA.get(cls, 1.0):
                continue
            keep.append((box.float().cpu(), float(score), class_names.index(cls), PHRASE_ATTRS.get(phrase, {})))
        if keep:
            boxes = torch.stack([k[0] for k in keep])
            scores = torch.tensor([k[1] for k in keep])
            cats = torch.tensor([k[2] for k in keep])
            groups = torch.tensor([class_names.index(NMS_GROUP.get(class_names[k[2]], class_names[k[2]])) for k in keep])
            order = batched_nms(boxes, scores, groups, iou_threshold=0.5)
            # одежда и сумка на человеке — не посторонний предмет на столе
            people = boxes[cats == class_names.index("person")]
            worn = set()
            for i in (cats == class_names.index("foreign_object")).nonzero().flatten().tolist():
                b = boxes[i]
                area = (b[2] - b[0]) * (b[3] - b[1])
                for p in people:
                    inter = (torch.min(b[2], p[2]) - torch.max(b[0], p[0])).clamp(min=0) * \
                            (torch.min(b[3], p[3]) - torch.max(b[1], p[1])).clamp(min=0)
                    if inter / area > 0.6:
                        worn.add(i)
            per_class: dict[int, int] = {}
            chosen = []
            for i in order.tolist():
                cat = int(cats[i])
                if i in worn or per_class.get(cat, 0) >= CLASSES[class_names[cat]][2]:
                    continue
                per_class[cat] = per_class.get(cat, 0) + 1
                chosen.append((cat, boxes[i].tolist(), float(scores[i]), keep[i][3]))
            if zone:
                tables = [b for cat, b, _, _ in chosen if class_names[cat] == "table"]
                doors = [b for cat, b, _, _ in chosen if class_names[cat] == "door"]
                filtered = []
                for cat, b, score, attrs in chosen:
                    name = class_names[cat]
                    if name in SMALL_IN_ZONE and center_in(b, zone) and not any(intersects(b, t) for t in tables):
                        continue  # предмет в стекле, а не на кассовом столе
                    in_doorway = any(overlap_share(b, d) > 0.5 for d in doors)
                    if (name == "person" and inside(b, zone) and not in_doorway
                            and b[3] < zone[3] - 0.05 * (zone[3] - zone[1])):
                        cat = class_names.index("reflection")  # человек в стекле, не в двери и не у низа стекла
                    filtered.append((cat, b, score, attrs))
                chosen = filtered
            for cat, (x1, y1, x2, y2), score, attrs in chosen:
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": cat + 1,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0, "score": round(score, 3),
                    "attributes": attrs,
                })
        if zone:
            x1, y1, x2, y2 = zone
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": img_id,
                "category_id": class_names.index("ignore_zone") + 1,
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0, "score": 1.0, "attributes": {"kind": "glass"},
            })

        if img_id <= args.preview:
            draw = ImageDraw.Draw(image)
            for ann in (a for a in coco["annotations"] if a["image_id"] == img_id):
                x, y, bw, bh = ann["bbox"]
                color = COLORS[ann["category_id"] - 1]
                draw.rectangle([x, y, x + bw, y + bh], outline=color, width=2)
                draw.text((x + 2, y + 1), f"{class_names[ann['category_id'] - 1]} {ann['score']:.2f}", fill=color)
            image.save(preview_dir / path.name, quality=85)
        if img_id % 100 == 0:
            print(f"{img_id}/{len(images)} кадров, {time.time() - started:.0f} с")

    out = root / args.out
    out.write_text(json.dumps(coco))
    counts = {c: 0 for c in class_names}
    for a in coco["annotations"]:
        counts[class_names[a["category_id"] - 1]] += 1
    print(f"→ {out}: {len(images)} кадров, {len(coco['annotations'])} боксов, {time.time() - started:.0f} с")
    print("по классам:", counts)


if __name__ == "__main__":
    main()
