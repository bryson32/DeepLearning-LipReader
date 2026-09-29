import unittest

import numpy as np

from config import INPUT_SHAPE
from network import build_3d_cnn


class NetworkTests(unittest.TestCase):
    def test_model_accepts_a_clip_and_has_a_bounded_parameter_count(self):
        model = build_3d_cnn(INPUT_SHAPE, 3)
        result = model(np.zeros((1, *INPUT_SHAPE), np.float32), training=False).numpy()
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        self.assertLess(model.count_params(), 100_000)
