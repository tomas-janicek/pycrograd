import math
import typing

if typing.TYPE_CHECKING:
    from pycrograd import tensor


def sum_backward(out: "tensor.Tensor") -> None:
    a, *_ = out.prev

    assert out.grad is not None and a.grad is not None

    out_grad = out.grad[0, 0]
    for row in range(a.rows):
        for col in range(a.cols):
            a.grad[row, col] += out_grad


def concat_backward(out: "tensor.Tensor") -> None:
    a, b = out.prev

    assert out.grad is not None and (a.grad is not None or b.grad is not None)

    if a.grad is not None:
        for row in range(a.rows):
            for col in range(a.cols):
                a.grad[row, col] += out.grad[row, col]

    if b.grad is not None:
        for row in range(b.rows):
            for col in range(b.cols):
                b.grad[row, col] += out.grad[row, col + a.cols]


def addition_backward(out: "tensor.Tensor") -> None:
    a, b = out.prev

    assert out.grad is not None and (a.grad is not None or b.grad is not None)
    assert a.rows == b.rows == out.rows
    assert a.cols == b.cols == out.cols

    for row in range(out.rows):
        for col in range(out.cols):
            grad_value = out.grad[row, col]
            if a.grad is not None:
                a.grad[row, col] += grad_value
            if b.grad is not None:
                b.grad[row, col] += grad_value


def matmul_backward(out: "tensor.Tensor") -> None:
    a, b = out.prev

    assert out.grad is not None and (a.grad is not None or b.grad is not None)

    for m in range(a.rows):
        for k in range(a.cols):
            for n in range(b.cols):
                if a.grad is not None:
                    a.grad[m, k] += b[k, n] * out.grad[m, n]
                if b.grad is not None:
                    b.grad[k, n] += a[m, k] * out.grad[m, n]


def power_backward(out: "tensor.Tensor") -> None:
    m, *_ = out.prev
    power, *_ = out.grad_args

    assert m.grad is not None and out.grad is not None
    assert m.rows == out.rows
    assert m.cols == out.cols

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += (power * m[row, col] ** (power - 1)) * out.grad[
                row, col
            ]


def mul_backward(out: "tensor.Tensor") -> None:
    m, *_ = out.prev
    multiplier, *_ = out.grad_args

    assert m.grad is not None and out.grad is not None
    assert m.rows == out.rows
    assert m.cols == out.cols

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += multiplier * out.grad[row, col]


def relu_backward(out: "tensor.Tensor") -> None:
    m, *_ = out.prev

    assert m.grad is not None and out.grad is not None

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += (out[row, col] > 0) * out.grad[row, col]


def log_backward(out: "tensor.Tensor") -> None:
    m, *_ = out.prev

    assert m.grad is not None and out.grad is not None

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += 1 / m[row, col] * out.grad[row, col]


def exp_backward(out: "tensor.Tensor") -> None:
    m, *_ = out.prev

    assert m.grad is not None and out.grad is not None

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += math.exp(m[row, col]) * out.grad[row, col]


def log_softmax_backward(out: "tensor.Tensor") -> None:
    m, *_ = out.prev

    assert m.grad is not None and out.grad is not None

    grad = out.grad - (out.data.exp() * out.grad.sum().item())

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] = grad[row, col]
