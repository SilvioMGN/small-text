from enum import Enum
from functools import partial, wraps

from scipy.sparse import csr_matrix

# args: (clf, x, x_indices_unlabeled, x_indices_labeled, y)
NUM_QUERY_ARGS = 5


class ClassificationType(Enum):
    SINGLE_LABEL = 'single-label'
    MULTI_LABEL = 'multi-label'


def constraints(query_func=None, classification_type=None):
    if not callable(query_func):
        return partial(constraints, classification_type=classification_type)

    @wraps(query_func)
    def wrap_query_func(*args, **kwargs):
        # args: (clf, x, x_indices_unlabeled, x_indices_labeled, y)
        if len(args) != NUM_QUERY_ARGS:
            raise TypeError(f'{query_func.__name__} is expected to be a function which takes '
                            f'{len()} positional arguments but {len(args)} were given')

        if classification_type is not None:
            if isinstance(classification_type, str):
                pass

            if classification_type == ClassificationType.SINGLE_LABEL:
                pass

        return query_func(*args, **kwargs)

    return wrap_query_func

"""def constraints(query_func=None, *, classification_type=None):
    if query_func is None:
        return partial(constraints, classification_type=classification_type)

    @wraps(query_func)
    def wrap_query_func(*args, **kwargs):
        # args: (clf, x, x_indices_unlabeled, x_indices_labeled, y)
        if len(args) != 5:
            print()

        if classification_type is not None:
            if isinstance(classification_type, str):
                classification_type = ClassificationType[classification_type]

            if classification_type == ClassificationType.SINGLE_LABEL:
                pass

        return query_func(*args, **kwargs)

    return wrap_query_func"""
