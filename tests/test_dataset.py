import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import INPUT_SHAPE, PREPROCESSING_VERSION
from dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "preprocessing.json").write_text(json.dumps({"version": PREPROCESSING_VERSION}))
        self.rng = np.random.default_rng(4)
        for word in ["zebra", "apple"]:
            (self.root / word).mkdir()
            np.save(self.root / word / "take_1.npy", self.rng.random(INPUT_SHAPE[:-1], dtype=np.float32))

    def test_labels_and_hashes_have_stable_order(self):
        x, y, labels, hashes = load_dataset(self.root)
        self.assertEqual(labels, ["apple", "zebra"])
        self.assertEqual(y.tolist(), [0, 1])
        self.assertEqual(x.shape, (2, *INPUT_SHAPE))
        self.assertEqual(hashes, load_dataset(self.root)[3])

    def test_model_label_order_is_preserved_for_subset(self):
        _, y, labels, _ = load_dataset(self.root, ["zebra", "unused", "apple"])
        self.assertEqual(labels, ["zebra", "unused", "apple"])
        self.assertEqual(y.tolist(), [0, 2])

    def test_nan_and_wrong_shape_fail(self):
        file = self.root / "apple/take_1.npy"
        for clip in [np.full(INPUT_SHAPE[:-1], np.nan, np.float32), np.zeros((2, 2), np.float32)]:
            np.save(file, clip)
            with self.assertRaises(ValueError):
                load_dataset(self.root)

    def test_duplicate_clips_fail(self):
        clip = np.load(self.root / "apple/take_1.npy")
        np.save(self.root / "zebra/take_1.npy", clip)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_dataset(self.root)

    def test_legacy_preprocessing_is_rejected(self):
        (self.root / "preprocessing.json").unlink()
        with self.assertRaisesRegex(ValueError, "preprocess.py"):
            load_dataset(self.root)

    def test_unknown_word_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent from the model"):
            load_dataset(self.root, ["apple"])

    def test_values_outside_normalized_range_are_rejected(self):
        for value in [-0.1, 1.1]:
            np.save(self.root / "apple/take_1.npy", np.full(INPUT_SHAPE[:-1], value, np.float32))
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_dataset(self.root)
