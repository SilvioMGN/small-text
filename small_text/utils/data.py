from small_text.base import LABEL_UNLABELED


def list_length(x):
    if hasattr(x, 'shape'):
        return x.shape[0]
    else:
        return len(x)


# TODO: unittest
def check_training_data(train_set, validation_set):

    if not train_set.is_multi_label:
        # this is not possible for csr_matrix so we only check the single-label case
        if (train_set.y == LABEL_UNLABELED).any():
            raise ValueError('Training labels must be labeled (greater or equal zero)')
        if validation_set is not None and (validation_set.y == LABEL_UNLABELED).any():
            raise ValueError('Validation set labels must be labeled (greater or equal zero)')
