import random

import numpy as np

from pycrograd.tensor import Tensor


def create_normal_weights(rows: int, cols: int) -> "Tensor":
    data = np.array(
        [[random.uniform(-1, 1) for _col in range(cols)] for _row in range(rows)]
    )
    return Tensor(data, requires_grad=True)


def create_kaiming_normal_weighta(rows: int, cols: int) -> "Tensor":
    random_array = np.random.randn(rows, cols).astype(np.float32)  # noqa: NPY002
    data = random_array * np.sqrt(2.0 / rows)
    return Tensor(data, requires_grad=True)
