import numpy as np

from pycrograd import matrix, tensor


def create_normal_weights(rows: int, cols: int) -> "tensor.Tensor":
    random_matrix = matrix.Matrix.rand(rows, cols)
    return tensor.Tensor(random_matrix, requires_grad=True)


def create_kaiming_normal_weighta(rows: int, cols: int) -> "tensor.Tensor":
    random_matrix = matrix.Matrix.randn(rows, cols)
    data = random_matrix * np.sqrt(2.0 / rows)
    return tensor.Tensor(data, requires_grad=True)
