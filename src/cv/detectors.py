"""Детекторы для пайплайна: видео → боксы по кадрам (L1) → журнал действий (L2, пакет decision).

Выбор — переменной DETECTOR:
  stub    — заглушка: видео из датасета (узнаёт по sha256 через data/labels.csv) получает предразметку Grounding DINO
            и эталонный журнал (reference/actions.yaml, violations.yaml); любое другое видео — пусто;
  yolo    — YOLO26 (Ultralytics, AGPL-3.0 — для сравнения, в прод не идёт);
  rfdetr  — RF-DETR (Apache-2.0).
DETECTOR_WEIGHTS — веса (по умолчанию лучший прогон в data/runs), SAMPLE_FPS — кадров в секунду на анализ,
DETECTOR_CONF — порог уверенности, DETECTOR_DEVICE — cpu или номер GPU (по умолчанию GPU, если есть). Интерфейс Detection общий: worker и бэкенд не зависят от выбора.
MODEL_DIR — каталог релиза модели (model.json, веса, классификатор шагов — scripts/publish-model.sh): задаёт всё это
поверх окружения, см. apply_model_dir.
"""

import csv
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cv.geometry import glass_filter, iou, median_box, overlap_share

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
REFERENCE_DIR = Path(os.environ.get("REFERENCE_DIR", "/opt/data_prepare/reference"))
# Порядок классов как в data_prepare/cvat/labels.json — так собран датасет det_l1 (id с нуля)
CLASS_NAMES = ["person", "reflection", "hand", "banknotes", "id_card", "cash_machine", "safe", "door", "phone", "bag",
               "table", "foreign_object", "computer", "ignore_zone"]
# Сколько боксов класса на кадре максимум (как в предразметке)
MAX_PER_FRAME = {"person": 6, "reflection": 6, "hand": 8, "banknotes": 8, "id_card": 2, "cash_machine": 1, "safe": 1,
                 "door": 1, "phone": 2, "bag": 3, "table": 1, "foreign_object": 4, "computer": 2, "ignore_zone": 1}
DEFAULT_WEIGHTS = {
    "yolo": DATA_DIR / "runs" / "yolo" / "det_l1-yolo26s" / "weights" / "best.pt",
    "rfdetr": DATA_DIR / "runs" / "rfdetr" / "det_l1-rfdetr-small" / "checkpoint_best_total.pth",
}


@dataclass
class Detection:
    """Результат детектора по видео."""

    detector: str
    boxes: list[dict] = field(default_factory=list)  # {t, cls, bbox: [x1,y1,x2,y2], conf, attributes, track_id?}
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


def iter_frames(video: Path, fps: float) -> Iterator[tuple[float, "object"]]:
    """Кадры видео с частотой fps: (секунда, кадр BGR). OpenCV сам учитывает поворот из метаданных .mov."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, i, next_i = native / fps, 0, 0.0
    try:
        while cap.grab():
            if i + 1e-6 >= next_i:
                ok, frame = cap.retrieve()
                if ok:
                    yield i / native, frame
                next_i += step
            i += 1
    finally:
        cap.release()


def pixel_attributes(frame, cls: str, bbox: list[float]) -> dict:
    """Признаки по пикселям бокса, которых нет в классах детектора:
    computer.bright — доля ярких пикселей (экран светится; после блокировки или закрытия крышки падает);
    safe.dark — доля тёмных пикселей (у открытого сейфа видно тёмное нутро, закрытый — светло-серый)."""
    if cls not in ("computer", "safe"):
        return {}
    import cv2

    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {}
    v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]
    if cls == "computer":
        return {"bright": round(float((v > 150).mean()), 3)}
    return {"dark": round(float((v < 60).mean()), 3)}


def postprocess(raw: list[tuple[float, list[tuple[str, list[float], float]]]]) -> list[dict]:
    """Сырые предсказания по кадрам → лимиты по классам, человек/отражение без дублей, фильтр стекла.

    Зона стекла — медиана предсказанных ignore_zone по видео (камера неподвижна); на выходе она одна на кадр.
    """
    zone = median_box([max((b for c, b, _ in dets if c == "ignore_zone"),
                           key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                       for _, dets in raw if any(c == "ignore_zone" for c, _, _ in dets)])
    out = []
    for t, dets in raw:
        kept, per_class = [], {}
        for cls, bbox, conf in sorted(dets, key=lambda d: -d[2]):
            if cls == "ignore_zone" or per_class.get(cls, 0) >= MAX_PER_FRAME.get(cls, 4):
                continue
            # один человек не бывает и человеком, и отражением — оставляем более уверенный вариант;
            # бокс человека почти целиком внутри другого — его же часть (по пояс), а не второй человек
            if cls in ("person", "reflection") and any(
                    c in ("person", "reflection") and (iou(b, bbox) > 0.5 or overlap_share(bbox, b) > 0.7
                                                       or overlap_share(b, bbox) > 0.7) for c, b, _ in kept):
                continue
            per_class[cls] = per_class.get(cls, 0) + 1
            kept.append((cls, bbox, conf))
        confs = {id(b): conf for _, b, conf in kept}
        for cls, bbox in glass_filter([(c, b) for c, b, _ in kept], zone):
            out.append({"t": round(t, 3), "cls": cls, "bbox": [round(v, 1) for v in bbox],
                        "conf": round(confs.get(id(bbox), 0.0), 3), "attributes": {}})
        if zone:
            out.append({"t": round(t, 3), "cls": "ignore_zone", "bbox": [round(v, 1) for v in zone], "conf": 1.0,
                        "attributes": {"kind": "glass"}})
    return out


class FrameDetector(Detector):
    """Общая часть обученных детекторов: кадры → предсказания пачками → постобработка → L2."""

    batch = 8

    def __init__(self) -> None:
        self.weights = Path(os.environ.get("DETECTOR_WEIGHTS") or DEFAULT_WEIGHTS[self.name])
        self.conf = float(os.environ.get("DETECTOR_CONF", "0.35"))
        self.fps = float(os.environ.get("SAMPLE_FPS", "4"))
        self.device = os.environ.get("DETECTOR_DEVICE") or None
        self.load()

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, frames: list) -> list[list[tuple[str, list[float], float]]]:
        """Кадры BGR → по каждому [(класс, [x1, y1, x2, y2], уверенность)]."""
        raise NotImplementedError

    def detect(self, video: Path) -> list[dict]:
        """Только L1: боксы по кадрам после постобработки и признаки по пикселям экрана и сейфа."""
        raw, chunk, out = [], [], []
        # признаки по пикселям считаются сразу по пачке, чтобы не держать в памяти все кадры видео
        pixels: dict[float, dict] = {}

        def flush():
            preds = self.predict([f for _, f in chunk])
            for (t, frame), dets in zip(chunk, preds):
                raw.append((t, dets))
                pixels[round(t, 3)] = {tuple(round(v, 1) for v in b): pixel_attributes(frame, c, b)
                                       for c, b, _ in dets if c in ("computer", "safe")}

        for t, frame in iter_frames(video, self.fps):
            chunk.append((t, frame))
            if len(chunk) == self.batch:
                flush()
                chunk = []
        if chunk:
            flush()
        for b in postprocess(raw):
            if b["cls"] in ("computer", "safe"):
                b["attributes"].update(pixels.get(b["t"], {}).get(tuple(b["bbox"]), {}))
            out.append(b)
        return out

    def run(self, video: Path, sha256: str) -> Detection:
        from decision.steps import analyze, with_vlm
        from decision.vlm import VLM_URL, Asker

        started = time.time()
        boxes = self.detect(video)
        res = analyze(boxes, self.fps)
        note = f"{self.weights.name}, {self.fps:g} кадр/с"
        if VLM_URL:  # спорные моменты — VLM (сервис vlm); недоступна — остаются правила
            ask = Asker(video, res["zones"])
            res = with_vlm(res, ask)
            note += f", VLM: {ask.calls} вопросов" + (f" ({ask.log[-1]})" if ask.log and "ошибка" in ask.log[-1] else "")
        return Detection(self.name, boxes=boxes, instances=res["instances"], note=f"{note}, {time.time() - started:.0f} с")


class YoloDetector(FrameDetector):
    name = "yolo"

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))
        self.names = self.model.names

    def predict(self, frames: list) -> list[list[tuple[str, list[float], float]]]:
        results = self.model.predict(frames, imgsz=640, conf=self.conf, verbose=False, device=self.device)
        return [[(self.names[int(c)], b.tolist(), float(s))
                 for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                for r in results]


class RfDetrDetector(FrameDetector):
    name = "rfdetr"

    def load(self) -> None:
        from rfdetr import RFDETRNano, RFDETRSmall

        size = os.environ.get("RFDETR_SIZE", "small")
        resolution = int(os.environ.get("RFDETR_RESOLUTION", "512"))
        self.model = {"small": RFDETRSmall, "nano": RFDETRNano}[size](
            pretrain_weights=str(self.weights), resolution=resolution, **({"device": self.device} if self.device else {}))

    def predict(self, frames: list) -> list[list[tuple[str, list[float], float]]]:
        rgb = [f[:, :, ::-1].copy() for f in frames]
        dets = self.model.predict(rgb, threshold=self.conf)
        dets = dets if isinstance(dets, list) else [dets]
        return [[(CLASS_NAMES[int(c)], b.tolist(), float(s)) for b, c, s in zip(d.xyxy, d.class_id, d.confidence)
                 if 0 <= int(c) < len(CLASS_NAMES)] for d in dets]


def apply_model_dir() -> str | None:
    """MODEL_DIR → DETECTOR, DETECTOR_WEIGHTS, STEP_CLASSIFIER/STEP_MODEL, DETECTOR_CONF, RFDETR_*, MODEL_TAG из model.json.

    Классификатор шагов обучен на L1 этих весов, поэтому релиз без него — только правила, а не чужой steps_<детектор>.
    Каталог — по realpath: процесс закреплён на теге, даже если симлинк current переключат на ходу.
    Возвращает тег; без MODEL_DIR — None и окружение не трогается."""
    if not os.environ.get("MODEL_DIR"):
        return None
    root = Path(os.environ["MODEL_DIR"]).resolve()
    manifest = json.loads((root / "model.json").read_text())
    env = {"DETECTOR": manifest["detector"], "DETECTOR_WEIGHTS": str(root / manifest["weights"]["file"]),
           "MODEL_TAG": manifest["tag"], "STEP_CLASSIFIER": "hybrid" if manifest.get("steps") else "rules"}
    if manifest.get("steps"):
        env["STEP_MODEL"] = str(root / manifest["steps"]["file"])
    params = manifest.get("params") or {}
    for name, var in (("conf", "DETECTOR_CONF"), ("rfdetr_size", "RFDETR_SIZE"), ("rfdetr_resolution", "RFDETR_RESOLUTION")):
        if params.get(name) is not None:
            env[var] = str(params[name])
    os.environ.update(env)
    return manifest["tag"]


def get_detector(kind: str | None = None) -> Detector:
    kind = kind or os.environ.get("DETECTOR", "stub")
    detectors = {"stub": StubDetector, "yolo": YoloDetector, "rfdetr": RfDetrDetector}
    if kind not in detectors:
        raise ValueError(f"неизвестный детектор: {kind}")
    return detectors[kind]()
