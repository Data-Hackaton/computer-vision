"""Частота кадров на анализ: SAMPLE_FPS=native — каждый кадр, FRAME_BUDGET, params.sample_fps из model.json.

Запуск — как test_decision.py:
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_sampling.py
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

import cv.detectors as detectors
from cv.detectors import FrameDetector, apply_model_dir


class Counter(FrameDetector):
    """Без модели: считает кадры, которые дошли до детектора."""

    name = "yolo"

    def load(self) -> None:
        self.frames = 0

    def predict(self, frames, conf=None):
        self.frames += len(frames)
        return [[] for _ in frames]


def env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return old


def clip(root: str, fps: int = 10, n: int = 10) -> Path:
    path = Path(root) / "clip.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        out.write(np.full((48, 64, 3), i * 20, np.uint8))
    out.release()
    return path


def frames_seen(rate: str, video: Path) -> Counter:
    old = env(SAMPLE_FPS=rate, DETECTOR_WEIGHTS="/nonexistent.pt", DETECTOR_VIEWS=None, DETECTOR_LOW_CONF=None)
    try:
        det = Counter()
        det.detect(video)
        return det
    finally:
        env(**old)


def test_native_takes_every_frame():
    with tempfile.TemporaryDirectory() as root:
        video = clip(root)
        det = frames_seen("native", video)
        assert det.frames == 10 and det.last_fps == 10 and not det.over_budget
        det = frames_seen("4", video)
        assert det.frames == 4 and det.last_fps == 4
        det = frames_seen("60", video)  # выше родной — родная, а не 60 в L2
        assert det.frames == 10 and det.last_fps == 10


def test_frame_budget_thins_long_video():
    budget = detectors.FRAME_BUDGET
    detectors.FRAME_BUDGET = 5
    try:
        with tempfile.TemporaryDirectory() as root:
            det = frames_seen("native", clip(root))
        assert det.over_budget and det.frames == 5 and det.last_fps == 5
    finally:
        detectors.FRAME_BUDGET = budget


def test_model_dir_sets_sample_fps():
    keys = ("MODEL_DIR", "DETECTOR", "DETECTOR_WEIGHTS", "MODEL_TAG", "STEP_CLASSIFIER", "STEP_MODEL", "DETECTOR_CONF",
            "DETECTOR_VIEWS", "TRACKER", "SAMPLE_FPS")
    old = env(**{k: None for k in keys})
    try:
        with tempfile.TemporaryDirectory() as root:
            manifest = {"tag": "model-yolo-test", "detector": "yolo", "weights": {"file": "best.pt"}, "steps": None,
                        "params": {"conf": 0.35, "sample_fps": "native"}}
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            os.environ["MODEL_DIR"] = root
            apply_model_dir()
            assert os.environ["SAMPLE_FPS"] == "native"
            manifest["params"]["sample_fps"] = 0
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            apply_model_dir()
            assert os.environ["SAMPLE_FPS"] == "native"
            del manifest["params"]["sample_fps"]  # релиз без параметра — прежние 4 кадр/с
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            apply_model_dir()
            assert os.environ["SAMPLE_FPS"] == "4"
    finally:
        env(**old)


def test_short_cashier_tail_keeps_legacy_sampling_behavior():
    from decision.steps import SIGNALS, segments

    states = [{**dict.fromkeys(SIGNALS, 0), "t": i / 4, "cashier_present": int(i < 40)} for i in range(42)]
    old = env(SAMPLE_FPS="4")
    try:
        assert segments(states, 4)[-1]["state"]["cashier_present"] == 1
        env(SAMPLE_FPS="native")
        assert segments(states, 4)[-1]["state"]["cashier_present"] == 0
    finally:
        env(**old)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
