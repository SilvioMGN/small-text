import unittest
import pytest

import numpy as np

from parameterized import parameterized_class
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from torch.nn.modules import BCEWithLogitsLoss

    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNEmbeddingMixin
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
@parameterized_class([{'multi_label': True},
                      {'multi_label': False}])
class KimCNNTest(unittest.TestCase):

    def _get_dataset(self, num_samples=100, num_classes=4):
        return random_text_classification_dataset(num_samples, max_length=60, num_classes=num_classes,
                                                  multi_label=self.multi_label)

    def test_fit_with_non_default_criterion_and_class_weighting(self):
        num_classes = 2
        embedding_matrix = torch.rand(10, 20)

        classifier = KimCNNClassifier(num_classes, embedding_matrix=embedding_matrix, device='cpu',
                                      class_weight='balanced')
        train_set = random_text_classification_dataset(8)
        criterion = BCEWithLogitsLoss()

        with self.assertWarnsRegex(RuntimeWarning, 'Class weighting will have no effect'):
            classifier.fit(train_set, criterion=criterion)

    def test_fit_predict(self):
        ds = self._get_dataset()

        num_classes = 4

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        clf = KimCNNClassifier(num_classes, multi_label=self.multi_label,
                               embedding_matrix=embedding_matrix)
        clf.fit(ds)

        predictions = clf.predict(ds)
        self.assertEqual(len(ds), predictions.shape[0])
