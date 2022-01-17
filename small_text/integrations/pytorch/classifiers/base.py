import logging
import warnings

from abc import abstractmethod

from small_text.classifiers.classification import Classifier
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn.functional as F

    from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
    from small_text.integrations.pytorch.utils.data import get_class_weights
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


logger = logging.getLogger(__name__)


class PytorchClassifier(Classifier):

    def __init__(self, multi_label=False, device=None):

        self.multi_label = multi_label

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.device.startswith('cuda'):
            logging.info('torch.version.cuda: %s', torch.version.cuda)
            logging.info('torch.cuda.is_available(): %s', torch.cuda.is_available())
            if torch.cuda.is_available():
                logging.info('torch.cuda.current_device(): %s', torch.cuda.current_device())

    @abstractmethod
    def fit(self, train_set, validation_set=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_set, return_proba=False):
        """
        Parameters
        ----------
        test_set : small_text.integrations.pytorch.PytorchTextClassificationDataset
            Test set.
        return_proba : bool
            If True, additionally returns the confidence distribution over all classes.

        Returns
        -------
        predictions : np.ndarray
            List of predictions.
        scores : np.ndarray (optional)
            Distribution of confidence scores over all classes if `return_proba` is True.
        """
        pass

    @abstractmethod
    def predict_proba(self, test_set):
        """
        Parameters
        ----------
        test_set : small_text.integrations.pytorch.PytorchTextClassificationDataset
            Test set.

        Returns
        -------
        scores : np.ndarray
            Distribution of confidence scores over all classes.
        """
        pass

    def get_default_criterion(self):

        if self.multi_label or self.num_classes == 2:
            return BCEWithLogitsLoss(pos_weight=self.class_weights_)
        else:
            return CrossEntropyLoss(weight=self.class_weights_)

    def initialize_class_weights(self, sub_train):
        if self.class_weight == 'balanced':
            if self.multi_label:
                warnings.warn('Setting class_weight to \'balanced\' is intended for the '
                              'single-label use case and might not have a beneficial '
                              'effect for multi-label classification')
            class_weights_ = get_class_weights(sub_train.y, self.num_classes)
            class_weights_ = class_weights_.to(self.device)
        elif self.class_weight is None:
            class_weights_ = None
        else:
            raise ValueError(f'Invalid value for class_weight kwarg: {self.class_weight}')

        return class_weights_

    def sum_up_accuracy_(self, logits, cls):
        if self.multi_label:
            proba = torch.sigmoid(logits)
            thresholded = F.threshold(proba, 0.5, 0)
            thresholded[thresholded > 0] = 1
            num_labels = logits.shape[1]
            acc = (thresholded == cls).sum(axis=1) / num_labels
            acc = acc.sum().item()
        else:
            acc = (logits.argmax(1) == cls).sum().item()

        return acc
