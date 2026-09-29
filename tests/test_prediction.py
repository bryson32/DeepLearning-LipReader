import unittest
from unittest.mock import Mock

import numpy as np

from config import INPUT_SHAPE
from predict import predict_sequence


class PredictionTests(unittest.TestCase):
    def test_decoding_uses_saved_label_order(self):
        model = Mock()
        model.return_value.numpy.return_value = np.array([[0.1, 0.9]], np.float32)
        result = predict_sequence(model, ["zebra", "apple"], np.zeros(INPUT_SHAPE[:-1], np.float32))
        self.assertEqual(result["word"], "apple")
        self.assertAlmostEqual(result["score"], 0.9)
        tensor = model.call_args.args[0]
        self.assertEqual(tensor.shape, (1, *INPUT_SHAPE))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertFalse(model.call_args.kwargs["training"])

    def test_bad_clip_never_reaches_the_model(self):
        model = Mock()
        with self.assertRaises(ValueError):
            predict_sequence(model, ["yes", "no"], np.zeros((2, 3), np.float32))
        model.assert_not_called()
