import unittest
import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from small_text.base import LABEL_IGNORED
from small_text.utils.labels import (
    concatenate,
    get_ignored_labels_mask,
    list_to_csr,
    remove_by_index
)

from tests.utils.testing import assert_csr_matrix_equal


class LabelUtilsTest(unittest.TestCase):

    def test_concatenate_dense(self):
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])

        result = concatenate(x, y)
        expected = np.array([1, 2, 3, 3, 2, 1])

        assert_array_equal(expected, result)

    def test_concatenate_sparse(self):
        x = csr_matrix(np.array([[0, 1], [1, 0], [1, 1]]))
        y = csr_matrix(np.array([[1, 1], [1, 0], [0, 1]]))

        result = concatenate(x, y)
        expected = csr_matrix(
            np.array([
                [0, 1], [1, 0], [1, 1], [1, 1], [1, 0], [0, 1]
            ])
        )

        assert_csr_matrix_equal(expected, result)

    def test_get_ignored_labels_mask_dense(self):

        y = np.array([1, LABEL_IGNORED, 3, 2])
        mask = get_ignored_labels_mask(y, LABEL_IGNORED)

        assert_array_equal(np.array([False, True, False, False]), mask)

    def test_get_ignored_labels_mask_sparse(self):

        y = csr_matrix(np.array([[1, 1], [LABEL_IGNORED, 0], [LABEL_IGNORED, LABEL_IGNORED], [1, 0]]))
        mask = get_ignored_labels_mask(y, LABEL_IGNORED)

        assert_array_equal(np.array([False, True, True, False]), mask)

    def test_remove_by_index_dense(self):
        y = np.array([3, 2, 1, 2, 1])
        y_new = remove_by_index(y, 3)
        expected = np.array([3, 2, 1, 1])

        assert_array_equal(expected, y_new)

    def test_remove_by_index_list_dense(self):
        y = np.array([3, 2, 1, 2, 1])
        y_new = remove_by_index(y, [3, 4])
        expected = np.array([3, 2, 1])

        assert_array_equal(expected, y_new)

    def test_remove_by_index_sparse(self):
        y = csr_matrix(np.array([[1, 1], [1, 0], [0, 1], [1, 1]]))

        y_new = remove_by_index(y, 2)
        expected = csr_matrix(
            np.array([
                [1, 1], [1, 0], [1, 1]
            ])
        )

        assert_csr_matrix_equal(expected, y_new)

    def test_remove_by_index_list_sparse(self):
        y = csr_matrix(np.array([[1, 1], [1, 0], [0, 1], [1, 1]]))

        y_new = remove_by_index(y, [2, 3])
        expected = csr_matrix(
            np.array([
                [1, 1], [1, 0]
            ])
        )

        assert_csr_matrix_equal(expected, y_new)

    def test_list_to_csr(self):
        label_list = [[], [0, 1], [1, 2, 3], [1], [], [0]]
        result = list_to_csr(label_list, (6, 4))

        self.assertTrue(isinstance(result, csr_matrix))
        self.assertEqual(np.int64, result.dtype)
        self.assertEqual(np.int64, result.data.dtype)
        self.assertEqual(np.int32, result.indices.dtype)
        self.assertEqual(np.int32, result.indices.dtype)

    def test_list_to_csr_float(self):
        label_list = [[], [0, 1], [1, 2, 3], [1], [], [0]]
        result = list_to_csr(label_list, (6, 4), dtype=np.float64)

        self.assertTrue(isinstance(result, csr_matrix))
        self.assertEqual(np.float64, result.dtype)
        self.assertEqual(np.float64, result.data.dtype)
        self.assertEqual(np.int32, result.indices.dtype)
        self.assertEqual(np.int32, result.indices.dtype)
