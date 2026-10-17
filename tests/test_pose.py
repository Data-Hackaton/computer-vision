"""Позы людей в L1 (cv.pose → attributes.pose): только у person, каждый POSE_EVERY-й кадр, params.pose из релиза.

Запуск — как test_decision.py:
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_pose.py
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

import cv.pose as pose
from cv.detectors import FrameDetector, apply_model_dir

PERSON = ("person", [10.0, 5.0, 40.0, 45.0], 0.9)
WEAK_PERSON = ("person", [45.0, 5.0, 60.0, 45.0], 0.4)
TABLE = ("table", [0.0, 30.0, 64.0, 48.0], 0.9)


class FakePose:
    kind = "fake"

    def __init__(self):
        self.calls = 0

    def landmarks(self, frame, bbox):
        self.calls += 1
        return [[bbox[0], bbox[1], 0.9]] * 33


class Det(FrameDetector):
    name = "yolo"

    def load(self) -> None:
        pass

    def predict(self, frames, conf=None):
        return [[PERSON, WEAK_PERSON, TABLE] for _ in frames]


def env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return old


def clip(root: str, n: int = 8) -> Path:
    path = Path(root) / "clip.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 48))
    for i in range(n):
        out.write(np.full((48, 64, 3), i * 20, np.uint8))
    out.release()
    return path


def run(root: str, every: str, fake) -> list[dict]:
    old = env(SAMPLE_FPS="4", POSE_EVERY=every, DETECTOR_WEIGHTS="/nonexistent.pt", DETECTOR_VIEWS=None,
              DETECTOR_LOW_CONF=None)
    orig = pose.PoseEstimator.from_env
    pose.PoseEstimator.from_env = classmethod(lambda cls: fake)
    try:
        return Det().detect(clip(root))
    finally:
        pose.PoseEstimator.from_env = orig
        env(**old)


def test_pose_only_on_confident_people():
    with tempfile.TemporaryDirectory() as root:
        fake = FakePose()
        boxes = run(root, "1", fake)
    people = [b for b in boxes if b["cls"] == "person"]
    with_pose = [b for b in people if "pose" in b["attributes"]]
    assert len(with_pose) == 8 and all(b["bbox"] == PERSON[1] for b in with_pose)  # слабый person — без позы
    assert len(with_pose[0]["attributes"]["pose"]) == 33 and fake.calls == 8
    assert not any("pose" in b["attributes"] for b in boxes if b["cls"] != "person")


def test_pose_every_second_frame_and_off():
    with tempfile.TemporaryDirectory() as root:
        fake = FakePose()
        boxes = run(root, "2", fake)
        assert sum("pose" in b["attributes"] for b in boxes) == 4 and fake.calls == 4
        boxes = run(root, "1", None)  # POSE=off или нет mediapipe → from_env даёт None
        assert not any("pose" in b["attributes"] for b in boxes)


def test_from_env_without_mediapipe_or_model():
    old = env(POSE="lite", POSE_MODEL_DIR="/nonexistent")
    try:
        assert pose.PoseEstimator.from_env() is None  # модели нет — позы выключены, а не падение
        env(POSE="off")
        assert pose.PoseEstimator.from_env() is None
    finally:
        env(**old)


def test_model_dir_sets_pose():
    keys = ("MODEL_DIR", "DETECTOR", "DETECTOR_WEIGHTS", "MODEL_TAG", "STEP_CLASSIFIER", "STEP_MODEL", "DETECTOR_CONF",
            "DETECTOR_VIEWS", "TRACKER", "SAMPLE_FPS", "POSE", "POSE_EVERY")
    old = env(**{k: None for k in keys})
    try:
        with tempfile.TemporaryDirectory() as root:
            manifest = {"tag": "model-yolo-test", "detector": "yolo", "weights": {"file": "best.pt"}, "steps": None,
                        "params": {"conf": 0.35, "pose": "lite", "pose_every": 3}}
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            os.environ["MODEL_DIR"] = root
            apply_model_dir()
            assert (os.environ["POSE"], os.environ["POSE_EVERY"]) == ("lite", "3")
            manifest["params"] = {"conf": 0.35}
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            apply_model_dir()
            assert (os.environ["POSE"], os.environ["POSE_EVERY"]) == ("off", "1")
    finally:
        env(**old)


def test_close_is_idempotent():
    from unittest.mock import Mock

    model = object.__new__(pose.PoseEstimator)
    native = model.landmarker = Mock()
    model.close()
    model.close()
    native.close.assert_called_once()
    assert model.landmarker is None


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
