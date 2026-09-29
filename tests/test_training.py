import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import INPUT_SHAPE
from train_model import train


class TrainingTests(unittest.TestCase):
    def test_too_few_takes_fail_before_training(self):
        data = (np.zeros((2, *INPUT_SHAPE), np.float32), np.array([0, 1]), ["yes", "no"], ["a", "b"])
        with tempfile.TemporaryDirectory() as temporary, patch("train_model.load_dataset", return_value=data):
            with self.assertRaisesRegex(ValueError, "five takes"):
                train("unused", Path(temporary) / "run")

    def test_invalid_run_settings_fail_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "positive"):
                train("unused", Path(temporary) / "run", epochs=0)
            with self.assertRaisesRegex(ValueError, "already exists"):
                train("unused", temporary)
