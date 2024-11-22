import math

import numpy as np
import torch
from torch.nn import functional as F

from pycrograd import tensor


def test_grad_initialization() -> None:
    m1 = tensor.Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=False)
    m2 = tensor.Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)

    assert m1.grad is None

    assert m2.grad is not None
    assert m2.grad.shape == (1, 3)
    assert m2.grad[0, 0] == 0.0
    assert m2.grad[0, 2] == 0.0


def test_sum() -> None:
    t1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    t2 = t1.sum()

    m1 = tensor.Tensor(
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]), requires_grad=True
    )
    m2 = m1.sum()

    assert t2.item() == m2.item()

    t2.backward()
    m2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_mean() -> None:
    t1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    t2 = t1.mean()

    m1 = tensor.Tensor(
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]), requires_grad=True
    )
    m2 = m1.mean()

    assert t2.item() == m2.item()

    t2.backward()
    m2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_matmul() -> None:
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    t2 = torch.tensor([[7.0], [8.0], [9.0]], requires_grad=True)
    result1 = t1 @ t2

    m1 = tensor.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
    m2 = tensor.Tensor(np.array([[7.0], [8.0], [9.0]]), requires_grad=True)
    result2 = m1 @ m2

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]

    sum1 = result1.sum()
    sum2 = result2.sum()

    assert sum1.item() == sum2.item()

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]

    assert t2.grad is not None
    assert m2.grad is not None
    assert t2.grad[0, 0] == m2.grad[0, 0]
    assert t2.grad[1, 0] == m2.grad[1, 0]
    assert t2.grad[2, 0] == m2.grad[2, 0]


def test_addition() -> None:
    t1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    t2 = torch.tensor([[4.0, 0.0], [5.0, 1.0], [6.0, 2.0]], requires_grad=True)
    result1 = t1 + t2

    m1 = tensor.Tensor(
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]), requires_grad=True
    )
    m2 = tensor.Tensor(
        np.array([[4.0, 0.0], [5.0, 1.0], [6.0, 2.0]]), requires_grad=True
    )
    result2 = m1 + m2

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]
    assert result1[1, 1] == result2[1, 1]

    sum1 = result1.sum()
    sum2 = result2.sum()

    assert sum1.item() == sum2.item()

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]

    assert t2.grad is not None
    assert m2.grad is not None
    assert t2.grad[0, 0] == m2.grad[0, 0]
    assert t2.grad[0, 1] == m2.grad[0, 1]
    assert t2.grad[1, 1] == m2.grad[1, 1]


def test_multiplication() -> None:
    t1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], requires_grad=True)
    result1 = t1 * 2

    m1 = tensor.Tensor(
        np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]), requires_grad=True
    )
    result2 = m1 * 2

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]
    assert result1[1, 1] == result2[1, 1]

    sum1 = result1.sum()
    sum2 = result2.sum()

    assert sum1.item() == sum2.item()

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_relu() -> None:
    t1 = torch.tensor([[-1.0, 4.0], [2.0, -5.0], [-3.0, -6.0]], requires_grad=True)
    result1 = t1.relu()

    m1 = tensor.Tensor(
        np.array([[-1.0, 4.0], [2.0, -5.0], [-3.0, -6.0]]), requires_grad=True
    )
    result2 = m1.relu()

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]
    assert result1[1, 1] == result2[1, 1]

    sum1 = result1.sum()
    sum2 = result2.sum()

    assert sum1.item() == sum2.item()

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_combination() -> None:
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    t2 = torch.tensor([[-1.0, 4.0, 5.0], [0.0, -3.0, -6.0]], requires_grad=True)
    t3 = torch.tensor([[7.0], [8.0], [9.0]], requires_grad=True)
    result1 = 3 * ((t1 + t2) - t1**2) @ (-t3 / 2)

    m1 = tensor.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
    m2 = tensor.Tensor(
        np.array([[-1.0, 4.0, 5.0], [0.0, -3.0, -6.0]]), requires_grad=True
    )
    m3 = tensor.Tensor(np.array([[7.0], [8.0], [9.0]]), requires_grad=True)
    result2 = 3 * ((m1 + m2) - m1**2) @ (-m3 / 2)

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]

    sum1 = result1.sum()
    sum2 = result2.sum()

    assert sum1.item() == sum2.item()

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_use_one_multiple_times() -> None:
    t1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], requires_grad=True
    )
    result1 = t1 @ t1 @ t1 @ t1 @ t1 @ t1

    m1 = tensor.Tensor(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]),
        requires_grad=True,
    )
    result2 = m1 @ m1 @ m1 @ m1 @ m1 @ m1

    assert result1[0, 0] == result2[0, 0]
    assert result1[1, 0] == result2[1, 0]

    sum1 = result1.sum()
    sum2 = result2.sum()

    assert sum1.item() == sum2.item()

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_log() -> None:
    t1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], requires_grad=True
    )
    log1 = t1.log()

    m1 = tensor.Tensor(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]),
        requires_grad=True,
    )
    log2 = m1.log()

    assert log1[0, 0] == log2[0, 0]
    assert log1[1, 0] == log2[1, 0]
    assert log1[2, 2] == log2[2, 2]

    sum1 = log1.sum()
    sum2 = log2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-4)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_exp() -> None:
    t1 = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], requires_grad=True
    )
    exp1 = t1.exp()

    m1 = tensor.Tensor(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]),
        requires_grad=True,
    )
    exp2 = m1.exp()

    assert math.isclose(exp1[0, 0], exp2[0, 0], rel_tol=1e-3)
    assert math.isclose(exp1[1, 0], exp2[1, 0], rel_tol=1e-3)
    assert math.isclose(exp1[2, 2], exp2[2, 2], rel_tol=1e-3)

    sum1 = exp1.sum()
    sum2 = exp2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert t1.grad[0, 0] == m1.grad[0, 0]
    assert t1.grad[0, 1] == m1.grad[0, 1]
    assert t1.grad[1, 1] == m1.grad[1, 1]


def test_softmax() -> None:
    t1 = torch.tensor(
        [[0.0], [2.0], [3.0]],
        requires_grad=True,
    )
    s1 = F.softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[0.0], [2.0], [3.0]]),
        requires_grad=True,
    )
    s2 = m1.softmax()

    assert math.isclose(s1[0, 0], s2[0, 0], rel_tol=1e-3)
    assert math.isclose(s1[1, 0], s2[1, 0], rel_tol=1e-3)
    assert math.isclose(s1[2, 0], s2[2, 0], rel_tol=1e-3)

    sum1 = s1.sum()
    sum2 = s2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_softmax_2() -> None:
    t1 = torch.tensor(
        [[2.0], [1.0], [0.1]],
        requires_grad=True,
    )
    s1 = F.softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[2.0], [1.0], [0.1]]),
        requires_grad=True,
    )
    s2 = m1.softmax()

    assert math.isclose(s1[0, 0], s2[0, 0], rel_tol=1e-3)
    assert math.isclose(s1[1, 0], s2[1, 0], rel_tol=1e-3)
    assert math.isclose(s1[2, 0], s2[2, 0], rel_tol=1e-3)

    sum1 = s1.sum()
    sum2 = s2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_softmax_equal() -> None:
    t1 = torch.tensor(
        [[1.0], [1.5], [1.75]],
        requires_grad=True,
    )
    s1 = F.softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[1.0], [1.5], [1.75]]),
        requires_grad=True,
    )
    s2 = m1.softmax()

    assert math.isclose(s1[0, 0], s2[0, 0], rel_tol=1e-3)
    assert math.isclose(s1[1, 0], s2[1, 0], rel_tol=1e-3)
    assert math.isclose(s1[2, 0], s2[2, 0], rel_tol=1e-3)

    sum1 = s1.sum()
    sum2 = s2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_softmax_one_big() -> None:
    t1 = torch.tensor(
        [[1.5], [26.0], [3000000.0]],
        requires_grad=True,
    )
    s1 = F.softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[1.5], [26.0], [3000000.0]]),
        requires_grad=True,
    )
    s2 = m1.softmax()

    assert math.isclose(s1[0, 0], s2[0, 0], rel_tol=1e-3)
    assert math.isclose(s1[1, 0], s2[1, 0], rel_tol=1e-3)
    assert math.isclose(s1[2, 0], s2[2, 0], rel_tol=1e-3)

    sum1 = s1.sum()
    sum2 = s2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax() -> None:
    t1 = torch.tensor(
        [[1.5], [26.0], [3000000.0]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[1.5], [26.0], [3000000.0]]),
        requires_grad=True,
    )
    ls2 = m1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sum1 = ls1.sum()
    sum2 = ls2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    # We do not care whether it is +/- if it is reasonably close to zero.
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax_2() -> None:
    t1 = torch.tensor(
        [[1.5], [26.0], [10000.0]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[1.5], [26.0], [10000.0]]),
        requires_grad=True,
    )
    ls2 = m1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sum1 = ls1.sum()
    sum2 = ls2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax_3() -> None:
    t1 = torch.tensor(
        [[0.5], [0.75], [0.01]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[0.5], [0.75], [0.01]]),
        requires_grad=True,
    )
    ls2 = m1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sum1 = ls1.sum()
    sum2 = ls2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax_4() -> None:
    t1 = torch.tensor(
        [[1.5], [3000000.0], [3000000.0]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[1.5], [3000000.0], [3000000.0]]),
        requires_grad=True,
    )
    ls2 = m1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sum1 = ls1.sum()
    sum2 = ls2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)


def test_log_softmax_5() -> None:
    t1 = torch.tensor(
        [[0.5], [200.0], [2000.0]],
        requires_grad=True,
    )
    ls1 = F.log_softmax(t1, dim=0)

    m1 = tensor.Tensor(
        np.array([[0.5], [200.0], [2000.0]]),
        requires_grad=True,
    )
    ls2 = m1.log_softmax()

    assert math.isclose(ls1[0, 0], ls2[0, 0], rel_tol=1e-3)
    assert math.isclose(ls1[1, 0], ls2[1, 0], rel_tol=1e-3)
    assert math.isclose(ls1[2, 0], ls2[2, 0], rel_tol=1e-3)

    sum1 = ls1.sum()
    sum2 = ls2.sum()

    assert math.isclose(sum1.item(), sum2.item(), rel_tol=1e-3)

    sum1.backward()
    sum2.backward()

    assert t1.grad is not None
    assert m1.grad is not None
    assert math.isclose(t1.grad[0, 0], m1.grad[0, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[1, 0], m1.grad[1, 0], abs_tol=1e-5)
    assert math.isclose(t1.grad[2, 0], m1.grad[2, 0], abs_tol=1e-5)
