import numpy as np

from scipy.sparse import csr_matrix

from small_text.data.datasets import split_data
from small_text.utils.labels import list_to_csr


def get_splits(train_set, validation_set, multi_label=False, validation_set_size=0.1):
    if validation_set is None:
        if multi_label:
            # TODO: adapt sampling for multi-label
            sub_train, sub_valid = split_data(train_set, y=train_set.y.indices, strategy='random',
                                              validation_set_size=validation_set_size)
        else:
            sub_train, sub_valid = split_data(train_set, y=train_set.y, strategy='stratified',
                                              validation_set_size=validation_set_size)
    else:
        sub_train, sub_valid = train_set, validation_set

    return sub_train, sub_valid


# TODO: unittest
def prediction_result(proba, multi_label, num_classes, enc=None, return_proba=False):
    if multi_label:
        predictions_binarized = np.where(proba > 0.5, 1, 0)
        if enc is not None:
            predictions = enc.inverse_transform(predictions_binarized)
        else:
            def multihot_to_list(x):
                return [i for i, item in enumerate(x) if item > 0]
            predictions = [multihot_to_list(row) for row in predictions_binarized]
        predictions = list_to_csr(predictions, shape=(len(predictions), num_classes))

        if return_proba:
            data = proba[predictions_binarized.astype(bool)]
            proba = csr_matrix((data, predictions.indices, predictions.indptr),
                               shape=predictions.shape,
                               dtype=np.float64)
    else:
        predictions = np.argmax(proba, axis=1)

    if return_proba:
        return predictions, proba

    return predictions


def empty_result(multi_label, num_classes, return_prediction=True, return_proba=True):
    """
    Helper method which returns an empty classification result. This ensures that all results have
    the correct dtype.

    At least one of `prediction` and `proba` must be `True`.

    Parameters
    ----------
    multi_label : bool
        Indicates a multi-label setting if `True`, otherwise a single-label setting
        when set to `False`.
    num_classes : int
        The number of classes.
    return_prediction : bool
        If `True`, returns an empty result of prediction.
    return_proba : bool
        If `True`, returns an empty result of probabilities.

    Returns
    -------
    predictions : np.ndarray[np.int64]
        An empty ndarray of predictions if `return_prediction` is True.
    proba : np.ndarray[float]
        An empty ndarray of probabilities if `return_proba` is True.
    """
    if not return_prediction and not return_proba:
        raise ValueError('Invalid usage: At least one of \'prediction\' or \'proba\' must be True')

    elif multi_label:
        if return_prediction and return_proba:
            return csr_matrix((0, num_classes), dtype=np.int64), csr_matrix((0, num_classes), dtype=float)
        elif return_prediction:
            return csr_matrix((0, num_classes), dtype=np.int64)
        else:
            return csr_matrix((0, num_classes), dtype=float)
    else:
        if return_prediction and return_proba:
            return np.empty((0, num_classes), dtype=np.int64), np.empty(shape=(0, num_classes), dtype=float)
        elif return_prediction:
            return np.empty((0, num_classes), dtype=np.int64)
        else:
            return np.empty((0, num_classes), dtype=float)
