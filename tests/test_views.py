"""Ансамбль видов кадра (cv.views): разбор DETECTOR_VIEWS, слияние WBF, порог после слияния, params.views из model.json.

Запуск — как test_decision.py:
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_views.py
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from cv.detectors import FrameDetector, apply_model_dir
from cv.views import parse_views, wbf

VIEWS = '[{"name": "orig", "weight": 0.76}, {"name": "unsharp", "weight": 0.12}, {"name": "clahe", "weight": 0.12}]'


def close(a, b):
    return abs(a - b) < 1e-6


def test_parse_views_normalizes_weights():
    assert parse_views(None) == [] and parse_views("  ") == []
    got = parse_views('[{"name": "orig", "weight": 2}, {"name": "clahe", "weight": 1}, {"name": "unsharp", "weight": 1}]')
    assert [n for n, _ in got] == ["orig", "clahe", "unsharp"]
    assert [round(w, 3) for _, w in got] == [0.5, 0.25, 0.25]
    for bad in ('[{"name": "sepia", "weight": 1}]', '[{"name": "orig", "weight": 0}]', "[]"):
        try:
            parse_views(bad)
        except ValueError:
            continue
        raise AssertionError(f"принят {bad}")


def test_wbf_fuses_agreeing_views():
    w = [0.76, 0.12, 0.12]
    got = wbf([[("hand", [0, 0, 10, 10], 0.8)], [("hand", [2, 0, 12, 10], 0.6)], []], w)
    assert len(got) == 1
    cls, box, score = got[0]
    # уверенность — сумма вклада видов, рамка — среднее с весами уверенность × вес вида
    assert cls == "hand" and close(score, 0.76 * 0.8 + 0.12 * 0.6)
    assert close(box[0], 2 * 0.12 * 0.6 / (0.76 * 0.8 + 0.12 * 0.6))


def test_wbf_keeps_classes_and_far_boxes_apart():
    w = [0.5, 0.25, 0.25]
    got = wbf([[("hand", [0, 0, 10, 10], 0.9), ("hand", [50, 50, 60, 60], 0.9)], [("phone", [0, 0, 10, 10], 0.8)], []], w)
    assert sorted((c, round(s, 3)) for c, _, s in got) == [("hand", 0.45), ("hand", 0.45), ("phone", 0.2)]
    # два попадания одного вида в один кластер считаются один раз — лучшее
    got = wbf([[("hand", [0, 0, 10, 10], 0.9), ("hand", [1, 0, 11, 10], 0.5)], [], []], w)
    assert len(got) == 1 and close(got[0][2], 0.45)


class Fake(FrameDetector):
    """Без модели: predict отдаёт заготовленные ответы по очереди — по одному на вид."""

    name = "yolo"

    def load(self) -> None:
        self.calls = []

    def predict(self, frames, conf=None):
        self.calls.append(conf)
        return self.answers.pop(0)


def env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return old


def test_predict_frames_threshold_after_fusion():
    old = env(DETECTOR_VIEWS=VIEWS, DETECTOR_CONF="0.35", DETECTOR_WEIGHTS="/nonexistent.pt")
    try:
        det = Fake()
        frame = np.zeros((8, 8, 3), np.uint8)
        # рамку видят все три вида — проходит; только исходный при 0,4 → 0,304 < 0,35 — отсеивается
        det.answers = [[[("hand", [0, 0, 10, 10], 0.5), ("phone", [20, 20, 30, 30], 0.4)]],
                       [[("hand", [0, 0, 10, 10], 0.5)]], [[("hand", [0, 0, 10, 10], 0.5)]]]
        got = det.predict_frames([frame])
        assert det.calls == [0.05, 0.05, 0.05]
        assert [(c, round(s, 3)) for c, _, s in got[0]] == [("hand", 0.5)]
        env(DETECTOR_VIEWS="")
        det = Fake()
        det.answers = [[[("phone", [20, 20, 30, 30], 0.4)]]]
        assert det.predict_frames([frame]) == [[("phone", [20, 20, 30, 30], 0.4)]] and det.calls == [None]
    finally:
        env(**old)


def test_model_dir_sets_views():
    keys = ("MODEL_DIR", "DETECTOR", "DETECTOR_WEIGHTS", "MODEL_TAG", "STEP_CLASSIFIER", "STEP_MODEL", "DETECTOR_CONF",
            "DETECTOR_VIEWS")
    old = env(**{k: None for k in keys})
    try:
        with tempfile.TemporaryDirectory() as root:
            manifest = {"tag": "model-yolo-test", "detector": "yolo", "weights": {"file": "best.pt"}, "steps": None,
                        "params": {"conf": 0.35, "views": json.loads(VIEWS)}}
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            os.environ["MODEL_DIR"] = root
            assert apply_model_dir() == "model-yolo-test"
            assert parse_views(os.environ["DETECTOR_VIEWS"]) == parse_views(VIEWS)
            # релиз без views сбрасывает ансамбль прошлого релиза
            del manifest["params"]["views"]
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            apply_model_dir()
            assert os.environ["DETECTOR_VIEWS"] == "" and os.environ["STEP_CLASSIFIER"] == "rules"
    finally:
        env(**old)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
