import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from images import crop_mouth, preprocess_frame


class ImageTests(unittest.TestCase):
    def test_crop_uses_clean_pixels_without_mutating_frame(self):
        frame = np.full((200, 300, 3), 80, dtype=np.uint8)
        original = frame.copy()
        landmarks = SimpleNamespace(part=lambda i: SimpleNamespace(x=100 + i % 2 * 60, y=90 + i % 2 * 20))
        crop, bounds = crop_mouth(frame, landmarks)
        np.testing.assert_array_equal(frame, original)
        x0, y0, x1, y1 = bounds
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        self.assertEqual(crop.shape, (80, 112, 3))
        self.assertTrue(np.all(crop == 80))

    def test_preprocessing_is_finite_float32(self):
        for image in [np.zeros((80, 112, 3), np.uint8), np.full((80, 112, 3), 255, np.uint8)]:
            result = preprocess_frame(image)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, (80, 112))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_invalid_image_fails(self):
        for image in [None, np.zeros((5, 5, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                preprocess_frame(image)
