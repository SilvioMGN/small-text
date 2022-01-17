import unittest

import numpy as np

from numpy.testing import assert_array_equal
from small_text.utils.classification import empty_result, prediction_result


class ClassificationUtilsTest(unittest.TestCase):

    def test_prediction_result(self):
        proba = np.array([
            [0.1, 0.2, 0.6, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.2, 0.2],
            [0.3, 0.2, 0.5, 0.1],
        ])
        result = prediction_result(proba, False, proba.shape[1])
        expected = np.array([2, 0, 0, 2])
        assert_array_equal(expected, result)

    def test_empty_result_single_label_prediction(self):
        num_labels = 3
        prediction = empty_result(False, num_labels, return_proba=False)

        self.assertTrue(isinstance(prediction, np.ndarray))
        self.assertEqual(np.int64, prediction.dtype)
        self.assertEqual((0, 3), prediction.shape)

    def test_empty_result_single_label_proba(self):
        num_labels = 3
        proba = empty_result(False, num_labels, return_prediction=False)

        self.assertTrue(isinstance(proba, np.ndarray))
        self.assertEqual(float, proba.dtype)
        self.assertEqual((0, 3), proba.shape)

    def test_empty_result_single_label_both(self):
        num_labels = 3
        prediction, proba = empty_result(False, num_labels)

        self.assertTrue(isinstance(prediction, np.ndarray))
        self.assertEqual(np.int64, prediction.dtype)
        self.assertEqual((0, 3), prediction.shape)

        self.assertTrue(isinstance(proba, np.ndarray))
        self.assertEqual(float, proba.dtype)
        self.assertEqual((0, 3), proba.shape)

    def test_empty_result_invalid_call(self):
        num_labels = 3
        multi_label_args = [True, False]
        for multi_label in multi_label_args:
            with self.assertRaisesRegex(ValueError, 'Invalid usage: At least one of'):
                empty_result(multi_label, num_labels, return_prediction=False, return_proba=False)
