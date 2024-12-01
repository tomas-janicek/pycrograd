import math
from array import array

import torch
from torch.nn import functional as F

from pycrograd import matrix, tensor


def test_grad_initialization() -> None:
    m1 = matrix.Matrix(1, 3, array("f", [1.0, 2.0, 3.0]))
    t1 = tensor.Tensor(m1, requires_grad=False)
    m2 = matrix.Matrix(1, 3, array("f", [1.0, 2.0, 3.0]))
    t2 = tensor.Tensor(m2, requires_grad=True)

    assert t1.grad is None

    assert t2.grad is not None
    assert t2.grad.rows == 1 and t2.grad.cols == 3
    assert t2.grad[0, 0] == 0.0
    assert t2.grad[0, 2] == 0.0


def test_sum() -> None:
    tt1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    tt2 = tt1.sum()

    m1 = matrix.Matrix(3, 2, array("f", [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]))
    t1 = tensor.Tensor(
        m1,
        requires_grad=True,
    )
    t2 = t1.sum()

    assert tt2.item() == t2.item()

    tt2.backward()
    t2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_mean() -> None:
    tt1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    tt2 = tt1.mean()

    t1 = tensor.Tensor(
        matrix.Matrix(3, 2, array("f", [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])),
        requires_grad=True,
    )
    t2 = t1.mean()

    assert tt2.item() == t2.item()

    tt2.backward()
    t2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_matmul() -> None:
    tt1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    tt2 = torch.tensor([[7.0], [8.0], [9.0]], requires_grad=True)
    result1 = tt1 @ tt2

    t1 = tensor.Tensor(
        matrix.Matrix(2, 3, array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        requires_grad=True,
    )
    t2 = tensor.Tensor(
        matrix.Matrix(3, 1, array("f", [7.0, 8.0, 9.0])), requires_grad=True
    )
    result2 = t1 @ t2

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]

    sut1 = result1.sum()
    sut2 = result2.sum()

    assert sut1.item() == sut2.item()

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]

    assert tt2.grad is not None
    assert t2.grad is not None
    assert tt2.grad[0, 0] == t2.grad[0, 0]
    assert tt2.grad[1, 0] == t2.grad[1, 0]
    assert tt2.grad[2, 0] == t2.grad[2, 0]


def test_addition() -> None:
    tt1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    tt2 = torch.tensor([[4.0, 0.0], [5.0, 1.0], [6.0, 2.0]], requires_grad=True)
    result1 = tt1 + tt2

    t1 = tensor.Tensor(
        matrix.Matrix(3, 2, array("f", [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])),
        requires_grad=True,
    )
    t2 = tensor.Tensor(
        matrix.Matrix(3, 2, array("f", [4.0, 0.0, 5.0, 1.0, 6.0, 2.0])),
        requires_grad=True,
    )
    result2 = t1 + t2

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]
    assert result1[1, 1] == result2[1, 1]

    sut1 = result1.sum()
    sut2 = result2.sum()

    assert sut1.item() == sut2.item()

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]

    assert tt2.grad is not None
    assert t2.grad is not None
    assert tt2.grad[0, 0] == t2.grad[0, 0]
    assert tt2.grad[0, 1] == t2.grad[0, 1]
    assert tt2.grad[1, 1] == t2.grad[1, 1]


def test_multiplication() -> None:
    tt1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    result1 = tt1 * 2

    t1 = tensor.Tensor(
        matrix.Matrix(3, 2, array("f", [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])),
        requires_grad=True,
    )
    result2 = t1 * 2

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]
    assert result1[1, 1] == result2[1, 1]

    sut1 = result1.sum()
    sut2 = result2.sum()

    assert sut1.item() == sut2.item()

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_relu() -> None:
    tt1 = torch.tensor([[-1.0, 4.0], [2.0, -5.0], [-3.0, -6.0]], requires_grad=True)
    result1 = tt1.relu()

    t1 = tensor.Tensor(
        matrix.Matrix(3, 2, array("f", [-1.0, 4.0, 2.0, -5.0, -3.0, -6.0])),
        requires_grad=True,
    )
    result2 = t1.relu()

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]
    assert result1[1, 1] == result2[1, 1]

    sut1 = result1.sum()
    sut2 = result2.sum()

    assert sut1.item() == sut2.item()

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_combination() -> None:
    tt1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    tt2 = torch.tensor([[-1.0, 4.0, 5.0], [0.0, -3.0, -6.0]], requires_grad=True)
    t3 = torch.tensor([[7.0], [8.0], [9.0]], requires_grad=True)
    result1 = 3 * ((tt1 + tt2) - tt1**2) @ (-t3 / 2)

    t1 = tensor.Tensor(
        matrix.Matrix(2, 3, array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        requires_grad=True,
    )
    t2 = tensor.Tensor(
        matrix.Matrix(2, 3, array("f", [-1.0, 4.0, 5.0, 0.0, -3.0, -6.0])),
        requires_grad=True,
    )
    m3 = tensor.Tensor(
        matrix.Matrix(3, 1, array("f", [7.0, 8.0, 9.0])), requires_grad=True
    )
    result2 = 3 * ((t1 + t2) - t1**2) @ (-m3 / 2)

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]

    sut1 = result1.sum()
    sut2 = result2.sum()

    assert sut1.item() == sut2.item()

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_use_one_multiple_times() -> None:
    tt1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], requires_grad=True
    )
    result1 = tt1 @ tt1 @ tt1 @ tt1 @ tt1 @ tt1

    t1 = tensor.Tensor(
        matrix.Matrix(3, 3, array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0])),
        requires_grad=True,
    )
    result2 = t1 @ t1 @ t1 @ t1 @ t1 @ t1

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]

    sut1 = result1.sum()
    sut2 = result2.sum()

    assert sut1.item() == sut2.item()

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_log() -> None:
    tt1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], requires_grad=True
    )
    log1 = tt1.log()

    t1 = tensor.Tensor(
        matrix.Matrix(3, 3, array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0])),
        requires_grad=True,
    )
    log2 = t1.log()

    assert log1[0, 0] == log2[0, 0]
    assert log1[1, 0] == log2[1, 0]
    assert log1[2, 2] == log2[2, 2]

    sut1 = log1.sum()
    sut2 = log2.sum()

    assert math.isclose(sut1.item(), sut2.item(), rel_tol=1e-4)

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_exp() -> None:
    tt1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], requires_grad=True
    )
    exp1 = tt1.exp()

    t1 = tensor.Tensor(
        matrix.Matrix(3, 3, array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0])),
        requires_grad=True,
    )
    exp2 = t1.exp()

    assert math.isclose(exp1[0, 0], exp2[0, 0], rel_tol=1e-3)
    assert math.isclose(exp1[1, 0], exp2[1, 0], rel_tol=1e-3)
    assert math.isclose(exp1[2, 2], exp2[2, 2], rel_tol=1e-3)

    sut1 = exp1.sum()
    sut2 = exp2.sum()

    assert math.isclose(sut1.item(), sut2.item(), rel_tol=1e-3)

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert tt1.grad[0, 0] == t1.grad[0, 0]
    assert tt1.grad[0, 1] == t1.grad[0, 1]
    assert tt1.grad[1, 1] == t1.grad[1, 1]


def test_log_softmax() -> None:
    tt1 = torch.tensor(
        [[0.5], [0.75], [0.01]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(tt1, dim=0)

    t1 = tensor.Tensor(
        matrix.Matrix(3, 1, array("f", [0.5, 0.75, 0.01])),
        requires_grad=True,
    )
    ls2 = t1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sut1 = ls1.sum()
    sut2 = ls2.sum()

    assert math.isclose(sut1.item(), sut2.item(), rel_tol=1e-3)

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert math.isclose(tt1.grad[0, 0], t1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(tt1.grad[1, 0], t1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(tt1.grad[2, 0], t1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax_one_big() -> None:
    tt1 = torch.tensor(
        [[1.5], [26.0], [10000.0]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(tt1, dim=0)

    t1 = tensor.Tensor(
        matrix.Matrix(3, 1, array("f", [1.5, 26.0, 10000.0])),
        requires_grad=True,
    )
    ls2 = t1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sut1 = ls1.sum()
    sut2 = ls2.sum()

    assert math.isclose(sut1.item(), sut2.item(), rel_tol=1e-3)

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert math.isclose(tt1.grad[0, 0], t1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(tt1.grad[1, 0], t1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(tt1.grad[2, 0], t1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax_two_even() -> None:
    tt1 = torch.tensor(
        [[1.5], [3000000.0], [3000000.0]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(tt1, dim=0)

    t1 = tensor.Tensor(
        matrix.Matrix(3, 1, array("f", [1.5, 3000000.0, 3000000.0])),
        requires_grad=True,
    )
    ls2 = t1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sut1 = ls1.sum()
    sut2 = ls2.sum()

    assert math.isclose(sut1.item(), sut2.item(), rel_tol=1e-3)

    sut1.backward()
    sut2.backward()

    assert tt1.grad is not None
    assert t1.grad is not None
    assert math.isclose(tt1.grad[0, 0], t1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(tt1.grad[1, 0], t1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(tt1.grad[2, 0], t1.grad[2, 0], abs_tol=1e-5)
