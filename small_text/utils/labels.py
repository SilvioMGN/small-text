import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack


# TODO: unittest
def get_num_labels(y):
    if isinstance(y, csr_matrix):
        return np.max(y.indices) + 1
    else:
        return np.max(y) + 1


def remove_by_index(y, indices):
    if isinstance(y, csr_matrix):
        mask = np.ones(y.shape[0], dtype=bool)
        mask[indices] = False
        return y[mask, :]
    else:
        return np.delete(y, indices)


def get_ignored_labels_mask(y, ignored_label_value):
    if isinstance(y, csr_matrix):
        return np.array([(row.toarray() == ignored_label_value).any() for row in y])
    else:
        return y == np.array([ignored_label_value])


def concatenate(a, b):
    if isinstance(a, csr_matrix) and isinstance(b, csr_matrix):
       return vstack([a, b])
    else:
        return np.concatenate([a, b])


# TODO: unittest
def csr_to_list(y):
    return [y.indices[tup[0]:tup[1]].tolist()
            for tup in list(zip(y.indptr, y.indptr[1:]))]


# TODO: unittest
def list_to_csr(label_list, shape, dtype=np.int64):

    if np.all(np.array([len(item) for item in label_list]) == 0):
        return csr_matrix(shape, dtype=dtype)

    # TODO: TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind'
    #   whenever list consists of only empty tuples
    col_ind = [item if len(item) > 0 else np.empty(0, dtype=np.int64)
               for item in label_list]

    col_ind = np.concatenate(col_ind, dtype=np.int64)
    row_ind = np.concatenate([[i] * len(item)
                              for i, item in enumerate(label_list) if len(item)], dtype=np.int64)
    data = np.ones_like(col_ind, dtype=np.int64)

    return csr_matrix((data, (row_ind, col_ind)), shape=shape, dtype=dtype)
