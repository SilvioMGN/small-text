import numpy as np

from scipy.sparse import csr_matrix

from small_text.utils.data import list_length
from small_text.data.sampling import balanced_sampling
from small_text.data.sampling import multilabel_stratified_subsets_sampling
from small_text.data.sampling import stratified_sampling


def random_initialization(x, n_samples=10):
    """Randomly draws from the given dataset x.

    Parameters
    ----------
    x :
        A supported dataset.
    n_samples :  int
        Number of samples to draw.

    Returns
    -------
    indices : np.array[int]
        Numpy array containing indices relative to x.
    """
    return np.random.choice(list_length(x), size=n_samples, replace=False)


def random_initialization_stratified(y, n_samples=10, multilabel_strategy='labelsets'):
    if isinstance(y, csr_matrix):
        if multilabel_strategy == 'labelsets':
            return multilabel_stratified_subsets_sampling(y, n_samples=n_samples)
        else:
            raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
    else:
        return stratified_sampling(y, n_samples=n_samples)


# TODO: multi-label?
def random_initialization_balanced(y, n_samples=10):
    return balanced_sampling(y, n_samples=n_samples)
