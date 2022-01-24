from enum import Enum
from functools import partial, wraps

from scipy.sparse import csr_matrix

# args: (self, clf, x, x_indices_unlabeled, x_indices_labeled, y)
NUM_QUERY_ARGS = 6


class ClassificationType(Enum):
    SINGLE_LABEL = 'single-label'
    MULTI_LABEL = 'multi-label'

    @staticmethod
    def from_str(classification_type_str):
        if classification_type_str == 'single-label':
            return ClassificationType.SINGLE_LABEL
        elif classification_type_str == 'multi-label':
            return ClassificationType.MULTI_LABEL
        else:
            raise ValueError('Cannot convert string to classification type enum: '
                             f'{classification_type_str}')


def constraints(query_func=None, classification_type=None):
    if not callable(query_func):
        return partial(constraints, classification_type=classification_type)

    @wraps(query_func)
    def wrap_query_func(*args, **kwargs):
        if len(args) != NUM_QUERY_ARGS:
            raise TypeError(f'{query_func.__name__} is expected to be a function which takes '
                            f'{NUM_QUERY_ARGS} positional arguments but {len(args)} were given')

        y = args[5]

        if classification_type is not None:
            if isinstance(classification_type, str):
                classification_type_ = ClassificationType.from_str(classification_type)

            if classification_type_ == ClassificationType.SINGLE_LABEL and isinstance(y, csr_matrix):
                raise RuntimeError(f'Invalid configuration: This query strategy requires '
                                   f'classification_type={str(classification_type_.value)} '
                                   f'but multi-label data was encountered')
            elif classification_type_ == ClassificationType.MULTI_LABEL \
                    and not isinstance(y, csr_matrix):
                raise RuntimeError(f'Invalid configuration: This query strategy requires '
                                   f'classification_type={str(classification_type_.value)} '
                                   f'but single-label data was encountered')

        return query_func(*args, **kwargs)

    return wrap_query_func
