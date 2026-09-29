import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from artifacts import load_run, save_run
from config import INPUT_SHAPE
from network import build_3d_cnn


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.model = build_3d_cnn(INPUT_SHAPE, 2)
        save_run(self.model, self.run, {"labels": ["yes", "no"]})

    def test_round_trip_keeps_predictions_and_labels_without_training_data(self):
        model, metadata = load_run(self.run)
        self.assertEqual(metadata["labels"], ["yes", "no"])
        clip = np.random.default_rng(7).random((1, *INPUT_SHAPE), dtype=np.float32)
        np.testing.assert_allclose(self.model(clip).numpy(), model(clip).numpy(), atol=1e-6)

    def test_incompatible_preprocessing_is_rejected(self):
        path = self.run / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["preprocessing"] = -1
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_run(self.run)

    def test_corrupt_model_is_rejected_before_loading(self):
        with (self.run / "model.keras").open("ab") as file:
            file.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_run(self.run)

    def test_existing_run_is_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_run(self.model, self.run, {"labels": ["yes", "no"]})

    def test_label_count_must_match_the_model(self):
        path = self.run / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["labels"].append("maybe")
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "shape does not match"):
            load_run(self.run)

    def test_missing_run_has_a_useful_error(self):
        with self.assertRaisesRegex(ValueError, "train_model.py first"):
            load_run(self.run / "missing")
