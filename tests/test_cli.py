from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from config import ROOT


class CommandTests(unittest.TestCase):
    def test_help_runs_outside_the_project_without_opening_a_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ["collection", "preprocess", "train_model", "predict", "evaluate"]:
                with self.subTest(command=name):
                    result = subprocess.run([sys.executable, str(ROOT / "src" / f"{name}.py"), "--help"],
                                            cwd=directory, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_missing_run_fails_before_opening_a_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / "src/predict.py"), "--run",
                                     str(Path(directory) / "missing")], cwd=directory,
                                    capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Run train_model.py first", result.stderr)

    def test_invalid_word_fails_before_opening_a_camera(self):
        result = subprocess.run([sys.executable, str(ROOT / "src/collection.py"), "../escape"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Word must contain", result.stderr)
