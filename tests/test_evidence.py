"""ECL: held-out video isolation, recall thresholds, startup validation and event propagation."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from cv.detectors import Detection, apply_model_dir, get_detector
from cv.pipeline import build_video_events, describe
from decision import evidence


def examples():
    return [{"video": f"v{i}", "cls": "phone_call", "y": y, "t": float(y),
             "f": {"signal": float(y) + i / 10, "noise": float(i % 2)}}
            for i in range(4) for y in (0, 1)]


class ThresholdTests(unittest.TestCase):
    def test_recall_at_boundaries_and_small_samples(self):
        for n in (1, 4, 5, 10, 11):
            for pos in (np.linspace(0, .03, n), np.ones(n), np.full(n, .0001)):
                with self.subTest(n=n, probabilities=pos):
                    pred = np.r_[pos, .8, np.nan]
                    labels = np.r_[np.ones(n), 0, 1]
                    classes = np.array(["phone_call"] * len(pred))
                    threshold = evidence.thresholds_from(pred, labels, classes)["phone_call"]
                    self.assertGreaterEqual(np.mean(pos >= threshold), 1 if n < 5 else .9)
                    self.assertIn(threshold, pos)

    def test_unlabelled_or_negative_only_classes_have_no_threshold(self):
        self.assertEqual(evidence.thresholds_from(np.array([.8, .1]), np.array([1, 0]),
                         np.array(["put:banknotes>desk", "phone_call"])), {})


class FoldTests(unittest.TestCase):
    def test_heldout_labels_never_affect_its_model_or_thresholds(self):
        original = examples()
        changed = copy.deepcopy(original)
        for row in changed:
            if row["video"] == "v0":
                row["y"] = 1 - row["y"]
        one, two = evidence.train(original), evidence.train(changed)
        evidence.validate(one)
        self.assertEqual(one["folds"]["v0"]["thresholds"], two["folds"]["v0"]["thresholds"])
        probe = np.array([[0., 0.], [1., 1.]])
        np.testing.assert_array_equal(one["folds"]["v0"]["model"].predict_proba(probe),
                                      two["folds"]["v0"]["model"].predict_proba(probe))
        self.assertNotIn("v0", one["evaluation"]["folds"]["v0"]["training_videos"])
        self.assertEqual(one["evaluation"]["overall"]["scored"], 8)
        self.assertIsNotNone(one["evaluation"]["overall"]["brier"])
        json.dumps(one["evaluation"], allow_nan=False)

    def test_unusable_folds_are_omitted_without_full_model_fallback(self):
        data = [{"video": f"v{y}", "cls": "phone_call", "y": y, "f": {"x": float(y)}} for y in (0, 1)]
        saved = evidence.train(data)
        evidence.validate(saved)
        self.assertEqual(saved["evaluation"]["overall"]["scored"], 0)
        self.assertEqual(saved["evaluation"]["overall"]["omitted"], 2)
        self.assertIsNone(saved["evaluation"]["overall"]["auc"])
        self.assertIsNone(saved["evaluation"]["overall"]["brier"])
        self.assertIsNone(evidence.select_fold(saved, "v0")["model"])
        self.assertIs(evidence.select_fold(saved, "unseen"), saved)
        actions = [{"t": 1, "action": "phone_call"}]
        with patch.object(evidence, "load", return_value=saved), patch.dict(os.environ, {"ECL_VIDEO": "v0"}), \
                patch.object(evidence, "vector", side_effect=AssertionError("must not score failed fold")):
            evidence.apply(actions, {})
        self.assertNotIn("p", actions[0])
        self.assertEqual(saved["thresholds"], {})
        del saved["folds"]["v0"]["thresholds"]
        with self.assertRaisesRegex(ValueError, "пропущенный fold"):
            evidence.validate(saved)

    def test_empty_and_single_class_training_fail_explicitly(self):
        for data in ([], [r for r in examples() if r["y"] == 1]):
            with self.assertRaisesRegex(ValueError, "обоих классов"):
                evidence.train(data)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = evidence.train(examples())

    def tearDown(self):
        evidence._cache.clear()

    def test_load_validates_and_refreshes_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecl.joblib"
            joblib.dump(self.saved, path)
            with patch.dict(os.environ, {"ECL_MODEL": str(path)}):
                self.assertEqual(evidence.load()["format_version"], 2)
                path.write_bytes(b"bad model")
                with self.assertRaisesRegex(RuntimeError, "Некорректная модель ECL"):
                    evidence.load()
                path.unlink()
                with self.assertRaisesRegex(RuntimeError, "модель недоступна"):
                    evidence.apply([], {})

    def test_bad_feature_shape_and_classes_rejected(self):
        saved = copy.deepcopy(self.saved)
        saved["features"].append("extra")
        with self.assertRaisesRegex(ValueError, "число признаков"):
            evidence.validate(saved)
        saved = copy.deepcopy(self.saved)
        saved["model"][-1].classes_ = np.array([1, 0])
        with self.assertRaisesRegex(ValueError, "классы"):
            evidence.validate(saved)

    def test_manifest_enables_ecl_and_legacy_release_clears_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            root = Path(directory)
            path = root / "ecl.joblib"
            joblib.dump(self.saved, path)
            manifest = {"tag": "model-test", "detector": "yolo", "weights": {"file": "best.pt"},
                        "ecl": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
            (root / "model.json").write_text(json.dumps(manifest))
            os.environ["MODEL_DIR"] = str(root)
            apply_model_dir()
            self.assertEqual(os.environ["ECL"], "on")
            self.assertEqual(os.environ["ECL_MODEL"], str(path))
            self.assertEqual(evidence.load()["format_version"], 2)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "ECL-артефакт"):
                apply_model_dir()
            path.unlink()
            with self.assertRaises(RuntimeError):
                apply_model_dir()
            del manifest["ecl"]
            (root / "model.json").write_text(json.dumps(manifest))
            apply_model_dir()
            self.assertEqual((os.environ["ECL"], os.environ["ECL_MODEL"]), ("off", ""))

    def test_startup_fails_before_first_task_but_off_does_not_load(self):
        with patch.dict(os.environ, {"ECL": "on", "ECL_MODEL": "/missing/model.joblib"}):
            with self.assertRaisesRegex(RuntimeError, "модель недоступна"):
                get_detector("stub")
        with patch.dict(os.environ, {"ECL": "off"}), patch.object(evidence, "load", side_effect=AssertionError):
            self.assertEqual(get_detector("stub").name, "stub")


class LabelAndEventTests(unittest.TestCase):
    def test_partial_annotations_and_one_to_one_matching(self):
        desk = {"t": 1., "action": "put", "object": "banknotes", "target": "desk"}
        phone = {"t": 2., "action": "phone_call"}
        self.assertEqual(evidence.label([desk, phone, dict(phone)], [phone], "violation", [], "X-1"),
                         [None, 1, 0])
        self.assertEqual(evidence.label([phone], None, "no_violation", [], "X-3"), [None])

    def test_duplicate_actions_keep_own_probability_and_warning_survives_description(self):
        step = {"action": "phone_call", "actor": "cashier", "required": True}
        spec = {"scenarios": {"client_service": {"steps": [step], "invariants": [
            {"kind": "forbidden", "action": step, "norm": "test", "title": "Phone", "why": "test"}]}}}
        def action(p, **extra):
            return {"t": 1., **step, "p": p, "certain": False, **extra}
        instances = [{"scenario": "client_service", "actions": [action(.11), action(.22), action(.93, certain=True)]},
                     {"scenario": "client_service", "actions": [action(.33)]}]
        events = build_video_events(Detection("test", instances=instances), spec)
        self.assertEqual([e.get("ecl_p") for e in events], [.11, .22, None, .33])
        self.assertEqual([e["status"] for e in events], ["warning", "warning", "violation", "warning"])
        self.assertTrue(all(e["step"] == 1 for e in events))
        for event in (events[0], events[1], events[3]):
            rendered = describe(event, {"Phone": "Reason"})
            self.assertIn(f"{event['ecl_p']:.2f}", rendered)
            self.assertEqual(rendered.count("Модель не уверена"), 1)
        # A pre-existing human 'certain: false' is not an ECL output; off keeps prior behavior.
        instances[0]["actions"] = [{"t": 1., **step, "certain": False}]
        self.assertEqual(build_video_events(Detection("test", instances=instances[:1]), spec)[0]["status"], "violation")


if __name__ == "__main__":
    unittest.main()
