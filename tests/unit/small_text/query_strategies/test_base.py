import unittest
import numpy as np

from small_text.classifiers.classification import SklearnClassifier, ConfidenceEnhancedLinearSVC
from small_text.query_strategies.base import ClassificationType, constraints
from small_text.query_strategies.strategies import RandomSampling

from tests.utils.datasets import random_sklearn_dataset


class FakeSingleLabelQueryStrategy(RandomSampling):

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        return super().query(clf, x, x_indices_unlabeled, x_indices_labeled, y, n=n)


class FakeMultiLabelQueryStrategy(RandomSampling):

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        return super().query(clf, x, x_indices_unlabeled, x_indices_labeled, y, n=n)


class FakeSingleLabelQueryStrategy(RandomSampling):

    @constraints(classification_type='single-label')
    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        return super().query(clf, x, x_indices_unlabeled, x_indices_labeled, y, n=n)


class ConstraintTest(unittest.TestCase):

    def test_without_constraint(self):
        sls = FakeSingleLabelQueryStrategy()
        self._test_query_strategy(sls)

    def _test_query_strategy(self, query_strategy):

        clf = SklearnClassifier(ConfidenceEnhancedLinearSVC(), 2)
        ds = random_sklearn_dataset(num_samples=100)

        x_indices_all = np.arange(len(ds))
        x_indices_labeled = np.random.choice(x_indices_all, 10, replace=False)
        x_indices_unlabeled = np.delete(x_indices_all, x_indices_labeled)
        y = np.random.randint(2)

        query_strategy.query(clf, ds, x_indices_unlabeled, x_indices_labeled, y)
