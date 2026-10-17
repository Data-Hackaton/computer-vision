"""Трекеры людей Ultralytics (decision.tracking): замена lap, буфер потерянного трека, отражения, вспышки, слабые боксы.

Запуск — как test_decision.py:
  docker compose --profile tools run --rm -v ../computer-vision/tests:/app/tests:ro cv python /app/tests/test_tracking.py
"""

import os

import numpy as np

from decision.scene import frames_of, track_people
from decision.steps import analyze

FPS = 4
CASHIER = [150, 450, 260, 800]
CLIENT = [600, 150, 700, 420]


def box(t, bbox, cls="person", conf=0.9, low=False):
    return {"t": round(t, 3), "cls": cls, "bbox": list(bbox), "conf": conf, "attributes": {"low": True} if low else {}}


def run(boxes, kind="bytetrack", low=None):
    old = os.environ.get("TRACKER")
    os.environ["TRACKER"] = kind
    try:
        track_people(frames_of(boxes), FPS, low=low)
    finally:
        os.environ.pop("TRACKER") if old is None else os.environ.__setitem__("TRACKER", old)
    return boxes


def ids(boxes, bbox):
    return [b.get("track_id") for b in boxes if b["bbox"] == list(bbox)]


def test_lap_replacement_respects_cost_limit():
    from decision.tracking import _install_lap

    _install_lap()
    import lap

    cost = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.95]])
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=0.5)
    # пары дороже 0,5 не берутся; третья строка без пары, потому что её дешёвый столбец занят выгоднее
    assert list(x) == [0, 1, -1] and list(y) == [0, 1]
    _, x, _ = lap.lapjv(np.array([[0.7]]), extend_cost=True, cost_limit=0.5)
    assert list(x) == [-1]


def test_gap_shorter_than_buffer_keeps_id():
    # кассир пропал на 2 с (буфер 3 с) — тот же трек; клиент — свой
    boxes = [box(i / FPS, CASHIER) for i in range(40) if not 12 <= i < 20] + [box(i / FPS, CLIENT) for i in range(40)]
    run(boxes)
    cashier, client = set(ids(boxes, CASHIER)), set(ids(boxes, CLIENT))
    assert len(cashier) == 1 and len(client) == 1 and cashier != client
    assert min(b["track_id"] for b in boxes) == 1  # нумерация с 1 в каждом видео


def test_gap_longer_than_buffer_gives_new_id():
    boxes = [box(i / FPS, CASHIER) for i in range(10)] + [box(i / FPS, CASHIER) for i in range(30, 40)]
    run(boxes)
    assert len(set(ids(boxes, CASHIER))) == 2


def test_single_frame_flash_and_lone_reflection_get_no_track():
    boxes = [box(i / FPS, CASHIER) for i in range(20)]
    boxes.append(box(2.0, CLIENT))  # вспышка на один кадр
    boxes += [box(i / FPS, [400, 100, 480, 300], "reflection") for i in range(20)]  # отражение без человека
    boxes.append(box(1.0, [160, 460, 250, 600]))  # вложенный в кассира
    run(boxes)
    assert ids(boxes, CLIENT) == [None]
    assert set(ids(boxes, [400, 100, 480, 300])) == {None}
    assert ids(boxes, [160, 460, 250, 600]) == [None]


def test_reflection_continues_person_track():
    # кассир встал перед стеклом — в L1 это reflection той же рамки: трек продолжается
    boxes = [box(i / FPS, CASHIER) for i in range(10)] + [box(i / FPS, CASHIER, "reflection") for i in range(10, 16)]
    run(boxes)
    assert set(b.get("track_id") for b in boxes) == {1}


def test_weak_boxes_extend_track_and_unmatched_are_dropped():
    # кассира заслонили: 2,5–4,5 с только слабые боксы; слабый бокс в стороне трека не продлевает
    strong = [box(i / FPS, CASHIER) for i in range(10)] + [box(i / FPS, CASHIER) for i in range(18, 30)]
    weak = [box(i / FPS, CASHIER, conf=0.2, low=True) for i in range(10, 18)] + [box(3.0, CLIENT, conf=0.2, low=True)]
    old = os.environ.get("TRACKER")
    os.environ["TRACKER"] = "bytetrack"
    try:
        boxes = strong + weak
        analyze(boxes, FPS)
    finally:
        os.environ.pop("TRACKER") if old is None else os.environ.__setitem__("TRACKER", old)
    kept_weak = [b for b in boxes if b["attributes"].get("low")]
    assert len(kept_weak) == 8 and {b["track_id"] for b in kept_weak} == {1}
    assert not any(b["bbox"] == CLIENT for b in boxes)
    # жадный трекер слабые боксы не берёт
    boxes = [dict(b, attributes=dict(b["attributes"])) for b in strong + weak]
    for b in boxes:
        b.pop("track_id", None)
    analyze(boxes, FPS)
    assert not any(b["attributes"].get("low") for b in boxes)


def test_model_dir_sets_tracker():
    import json
    import tempfile
    from pathlib import Path

    from cv.detectors import apply_model_dir

    keys = ("MODEL_DIR", "DETECTOR", "DETECTOR_WEIGHTS", "MODEL_TAG", "STEP_CLASSIFIER", "STEP_MODEL", "DETECTOR_CONF",
            "DETECTOR_VIEWS", "TRACKER")
    old = {k: os.environ.pop(k, None) for k in keys}
    try:
        with tempfile.TemporaryDirectory() as root:
            manifest = {"tag": "model-yolo-test", "detector": "yolo", "weights": {"file": "best.pt"}, "steps": None,
                        "params": {"conf": 0.35, "tracker": "bytetrack"}}
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            os.environ["MODEL_DIR"] = root
            apply_model_dir()
            assert os.environ["TRACKER"] == "bytetrack"
            # релиз без трекера возвращает жадный
            del manifest["params"]["tracker"]
            (Path(root) / "model.json").write_text(json.dumps(manifest))
            apply_model_dir()
            assert os.environ["TRACKER"] == "greedy"
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_botsort_runs_without_frames():
    boxes = [box(i / FPS, CASHIER) for i in range(20)] + [box(i / FPS, CLIENT) for i in range(20)]
    run(boxes, "botsort")
    assert len(set(ids(boxes, CASHIER))) == 1 and len(set(ids(boxes, CLIENT))) == 1


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
    print(f"{len(tests)} passed")
