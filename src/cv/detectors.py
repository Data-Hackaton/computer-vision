"""Детекторы для пайплайна: видео → боксы по кадрам и (если умеет) журнал действий.

Интерфейс один, выбор — переменной DETECTOR. Пока обученного детектора нет, работает StubDetector:
для видео из датасета (узнаёт по sha256 через data/labels.csv) отдаёт предразметку Grounding DINO
(data/frames/*/prelabels.coco.json) и эталонный журнал действий (reference/actions.yaml, violations.yaml);
для любого другого видео — пусто, вердикт no_violation. YOLO/RF-DETR + распознаватель действий
встанут сюда же, без изменений API и бэкенда.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
REFERENCE_DIR = Path(os.environ.get("REFERENCE_DIR", "/opt/data_prepare/reference"))


@dataclass
class Detection:
    """Результат детектора по видео."""

    detector: str
    boxes: list[dict] = field(default_factory=list)  # {t, cls, bbox: [x1,y1,x2,y2], conf, attributes}
    instances: list[dict] = field(default_factory=list)  # журнал действий: [{scenario, partial?, actions}]
    note: str | None = None


class Detector:
    name = "base"

    def run(self, video: Path, sha256: str) -> Detection:
        raise NotImplementedError


class StubDetector(Detector):
    name = "stub"

    def __init__(self) -> None:
        with (DATA_DIR / "labels.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.by_sha = {r["sha256"]: r["video"] for r in rows if not r["duplicate_of"]}
        journal = yaml.safe_load((REFERENCE_DIR / "actions.yaml").read_text())
        catalog = yaml.safe_load((REFERENCE_DIR / "violations.yaml").read_text())
        self.instances = {**{v: e["instances"] for v, e in journal.items()},
                          **{v: e["instances"] for v, e in catalog.items()}}
        self.prelabels = self._load_prelabels()

    @staticmethod
    def _load_prelabels() -> dict[str, list[dict]]:
        boxes: dict[str, list[dict]] = {}
        for path in sorted(DATA_DIR.glob("frames/*/prelabels.coco.json")):
            coco = json.loads(path.read_text())
            names = {c["id"]: c["name"] for c in coco["categories"]}
            images = {im["id"]: im["file_name"] for im in coco["images"]}
            for a in coco["annotations"]:
                video, _, ms = Path(images[a["image_id"]]).stem.rpartition("_")
                x, y, w, h = a["bbox"]
                boxes.setdefault(video, []).append({
                    "t": int(ms) / 1000, "cls": names[a["category_id"]], "bbox": [x, y, x + w, y + h],
                    "conf": a.get("score"), "attributes": a.get("attributes") or {},
                })
        return boxes

    def run(self, video: Path, sha256: str) -> Detection:
        name = self.by_sha.get(sha256)
        if name is None:
            return Detection(self.name, note="видео не из датасета: заглушка не знает его содержимого")
        boxes = sorted(self.prelabels.get(name, []), key=lambda b: b["t"])
        return Detection(self.name, boxes=boxes, instances=self.instances.get(name, []), note=f"датасет: {name}")


def get_detector() -> Detector:
    kind = os.environ.get("DETECTOR", "stub")
    if kind == "stub":
        return StubDetector()
    raise ValueError(f"неизвестный детектор: {kind}")
