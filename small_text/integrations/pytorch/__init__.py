from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.pytorch.datasets import (
        PytorchTextClassificationDataset  # noqa:F401
    )
except PytorchNotFoundError:
    pass
