import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from config import FRAME_COUNT
from images import load_take, prepare_sequence
from preprocess import process_dataset


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.take = self.root / "raw" / "hello" / "take_1"
        self.take.mkdir(parents=True)
        self.crops = np.random.default_rng(42).integers(0, 256, (FRAME_COUNT, 80, 112, 3), dtype=np.uint8)
        for i, crop in enumerate(self.crops):
            cv2.imwrite(str(self.take / f"frame_{i}.png"), crop)

    def test_disk_and_live_sequences_match_in_time_order(self):
        np.testing.assert_array_equal(load_take(self.take), prepare_sequence(self.crops))

    def test_missing_frame_is_rejected(self):
        (self.take / "frame_2.png").unlink()
        with self.assertRaisesRegex(ValueError, "expected frames"):
            load_take(self.take)

    def test_failed_preprocessing_does_not_publish_partial_dataset(self):
        output = self.root / "processed"
        with patch("preprocess.load_take", side_effect=ValueError("bad frame")):
            with self.assertRaises(ValueError):
                process_dataset(self.root / "raw", output)
        self.assertFalse(output.exists())

    def test_dataset_is_written_once(self):
        output = self.root / "processed"
        process_dataset(self.root / "raw", output)
        np.testing.assert_array_equal(np.load(output / "hello/take_1.npy"), prepare_sequence(self.crops))
        self.assertTrue((output / "preprocessing.json").is_file())
        with self.assertRaisesRegex(ValueError, "already exists"):
            process_dataset(self.root / "raw", output)
