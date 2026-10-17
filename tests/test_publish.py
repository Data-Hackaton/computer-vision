"""Host-only integration of local release packaging; never calls GitHub."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git") and (REPO / "scripts/publish-model.sh").exists(), "run on host with git")
class PublishTests(unittest.TestCase):
    def test_artifacts_and_worktree_provenance(self):
        common = Path(subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
                                               "--git-common-dir"], text=True).strip())
        data = common.parent.parent / "data_prepare"
        expected = subprocess.check_output(["git", "-C", str(data), "rev-parse", "HEAD"], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "weights").mkdir(parents=True)
            (run / "weights/best.pt").write_bytes(b"weights fixture")
            ecl = root / "calibrator.joblib"
            ecl.write_bytes(b"artifact fixture")
            env = {k: v for k, v in os.environ.items() if k != "DATA_REPO"}
            for included in (False, True):
                out = root / str(included)
                command = ["bash", str(REPO / "scripts/publish-model.sh"), "yolo", str(run),
                           "--tag", "model-package-test", "--out", str(out)]
                if included:
                    command += ["--ecl", str(ecl)]
                subprocess.run(command, check=True, capture_output=True, text=True, env=env)
                manifest = json.loads((out / "model.json").read_text())
                self.assertEqual(manifest["data_commit"], expected)
                self.assertEqual("ecl" in manifest, included)
                if included:
                    self.assertEqual(manifest["ecl"]["sha256"], hashlib.sha256(ecl.read_bytes()).hexdigest())
                    self.assertEqual((out / manifest["ecl"]["file"]).read_bytes(), ecl.read_bytes())


if __name__ == "__main__":
    unittest.main()
