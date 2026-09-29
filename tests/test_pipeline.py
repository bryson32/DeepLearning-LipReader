import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from artifacts import load_run
from collection import save_take
from config import FRAME_COUNT
from evaluate import evaluate
from images import load_take
from predict import predict_sequence
from preprocess import process_dataset
from train_model import train


class PipelineTests(unittest.TestCase):
    def test_record_preprocess_train_reload_predict_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(11)
            for folder, count in [("training", 5), ("testing", 1)]:
                for word in ["yes", "no"]:
                    for _ in range(count):
                        crops = rng.integers(0, 256, (FRAME_COUNT, 80, 112, 3), dtype=np.uint8)
                        save_take(crops, root / folder, word)
                process_dataset(root / folder, root / f"processed_{folder}")
            model, metadata = train(root / "processed_training", root / "run", epochs=1, batch_size=2)
            self.assertEqual(len(metadata["history"]["loss"]), 1)
            self.assertEqual(set(metadata["training_hashes"]) & set(metadata["validation_hashes"]), set())
            self.assertEqual(len(metadata["training_hashes"]), 8)
            self.assertEqual(len(metadata["validation_hashes"]), 2)
            report = evaluate(root / "run", root / "processed_testing")
            self.assertEqual(report["clips"], 2)
            self.assertEqual(len(report["confusion_matrix"]), 2)
            with self.assertRaisesRegex(ValueError, "separate session"):
                evaluate(root / "run", root / "processed_training")
            shutil.rmtree(root / "processed_training")
            loaded, saved = load_run(root / "run")
            clip = load_take(root / "testing/yes/take_1")
            original = predict_sequence(model, metadata["labels"], clip)
            restored = predict_sequence(loaded, saved["labels"], clip)
            self.assertEqual(original["word"], restored["word"])
            self.assertAlmostEqual(original["score"], restored["score"], places=6)
            self.assertEqual(saved, json.loads((root / "run/metadata.json").read_text()))
