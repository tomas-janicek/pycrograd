import math
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from pycrograd import tensor


def sum_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    a, *_ = children

    assert out.grad and a.grad

    out_grad = out.grad[0, 0]
    for row in range(a.rows):
        for col in range(a.cols):
            a.grad[row, col] += out_grad


def concat_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    a, b = children

    assert out.grad and (a.grad or b.grad)

    if a.grad:
        for row in range(a.rows):
            for col in range(a.cols):
                a.grad[row, col] += out.grad[row, col]

    if b.grad:
        for row in range(b.rows):
            for col in range(b.cols):
                b.grad[row, col] += out.grad[row, col + a.cols]


def addition_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    a, b = children

    assert out.grad and (a.grad or b.grad)
    assert a.rows == b.rows == out.rows
    assert a.cols == b.cols == out.cols

    for row in range(out.rows):
        for col in range(out.cols):
            grad_value = out.grad[row, col]
            if a.grad:
                a.grad[row, col] += grad_value
            if b.grad:
                b.grad[row, col] += grad_value


def matmul_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    a, b = children

    assert out.grad and (a.grad or b.grad)

    for m in range(a.rows):
        for k in range(a.cols):
            for n in range(b.cols):
                if a.grad:
                    a.grad[m, k] += b[k, n] * out.grad[m, n]
                if b.grad:
                    b.grad[k, n] += a[m, k] * out.grad[m, n]


def power_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children
    power, *_ = grad_args

    assert m.grad and out.grad
    assert m.rows == out.rows
    assert m.cols == out.cols

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += (power * m[row, col] ** (power - 1)) * out.grad[
                row, col
            ]


def mul_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children
    multiplier, *_ = grad_args

    assert m.grad and out.grad
    assert m.rows == out.rows
    assert m.cols == out.cols

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += multiplier * out.grad[row, col]


def relu_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children

    assert m.grad and out.grad

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += (out[row, col] > 0) * out.grad[row, col]


def log_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children

    assert m.grad and out.grad

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += 1 / m[row, col] * out.grad[row, col]


def exp_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children

    assert m.grad and out.grad

    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] += math.exp(m[row, col]) * out.grad[row, col]


def softmax_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children

    assert m.grad and out.grad

    row_vector = out.data.T
    row_vector_grad = out.grad.data.T

    identity = np.eye(len(out), dtype=np.float32)
    jacobian = row_vector[:, :, None] * (identity[None, :, :] - row_vector[:, None, :])

    grad = (jacobian @ row_vector_grad[:, :, None]).squeeze(2)
    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] = grad[col, row]


def log_softmax_backward(
    out: "tensor.Matrix",
    children: typing.Sequence["tensor.Matrix"],
    grad_args: typing.Sequence[typing.Any],
) -> None:
    m, *_ = children

    assert m.grad and out.grad

    row_vector = out.data.T
    row_vector_grad = out.grad.data.T

    softmax_values = np.exp(row_vector)

    grad = row_vector_grad - (
        softmax_values * row_vector_grad.sum(axis=1, keepdims=True)
    )
    for row in range(m.rows):
        for col in range(m.cols):
            m.grad[row, col] = grad[col, row]
