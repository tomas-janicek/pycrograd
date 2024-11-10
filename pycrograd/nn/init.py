import random

import numpy as np

from pycrograd.tensor import Matrix


def create_normal_weights(rows: int, cols: int) -> "Matrix":
    data = np.array(
        [[random.uniform(-1, 1) for _col in range(cols)] for _row in range(rows)]
    )
    return Matrix(data, requires_grad=True)


def create_kaiming_normal_weighta(rows: int, cols: int) -> "Matrix":
    random_array = np.random.randn(rows, cols).astype(np.float32)  # noqa: NPY002
    data = random_array * np.sqrt(2.0 / rows)
    return Matrix(data, requires_grad=True)
