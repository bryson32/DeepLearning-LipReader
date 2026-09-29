import unittest
from unittest.mock import Mock, patch

import numpy as np

from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.metadata = {"labels": ["yes", "no"], "training_hashes": ["train"],
                         "validation_hashes": ["val"], "model_sha256": "model"}

    def test_overlap_is_rejected_before_inference(self):
        for digest in ["train", "val"]:
            with patch("evaluate.load_run", return_value=(self.model, self.metadata)), \
                 patch("evaluate.load_dataset", return_value=(None, None, ["yes", "no"], [digest])):
                with self.assertRaisesRegex(ValueError, "separate session"):
                    evaluate("run", "data")
        self.model.predict.assert_not_called()

    def test_metrics_follow_argmax_even_below_point_five(self):
        self.metadata["labels"] = ["yes", "no", "maybe"]
        self.model.predict.return_value = np.array([[0.4, 0.3, 0.3], [0.1, 0.8, 0.1]])
        data = (np.empty((2, 1)), np.array([0, 1]), self.metadata["labels"], ["new1", "new2"])
        with patch("evaluate.load_run", return_value=(self.model, self.metadata)), \
             patch("evaluate.load_dataset", return_value=data):
            result = evaluate("run", "data")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(result["classification_report"]["yes"]["recall"], 1.0)
        self.assertAlmostEqual(result["classification_report"]["macro avg"]["f1-score"], 2 / 3)

    def test_missing_split_provenance_is_rejected(self):
        self.metadata.pop("training_hashes")
        data = (None, None, ["yes", "no"], ["new"])
        with patch("evaluate.load_run", return_value=(self.model, self.metadata)), \
             patch("evaluate.load_dataset", return_value=data):
            with self.assertRaisesRegex(ValueError, "missing its training"):
                evaluate("run", "data")
        self.model.predict.assert_not_called()
