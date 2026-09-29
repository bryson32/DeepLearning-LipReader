import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from collection import save_take
from config import FRAME_COUNT
from images import load_take, prepare_sequence


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frames = np.random.default_rng(3).integers(0, 256, (FRAME_COUNT, 80, 112, 3), dtype=np.uint8)

    def test_recordings_round_trip_without_overwriting(self):
        save_take(self.frames, self.root, "hello")
        save_take(self.frames, self.root, "hello")
        self.assertTrue((self.root / "hello/take_2").is_dir())
        np.testing.assert_array_equal(load_take(self.root / "hello/take_1"), prepare_sequence(self.frames))

    def test_failed_write_leaves_no_partial_take(self):
        with patch("collection.cv2.imwrite", return_value=False):
            with self.assertRaises(OSError):
                save_take(self.frames, self.root, "hello")
        self.assertEqual(list((self.root / "hello").iterdir()), [])

    def test_invalid_word_and_incomplete_clip_are_rejected(self):
        for frames, word in [(self.frames, "../escape"), (self.frames[:2], "hello")]:
            with self.assertRaises(ValueError):
                save_take(frames, self.root, word)
