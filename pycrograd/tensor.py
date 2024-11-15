import collections
import typing

import numpy as np
from numpy import typing as np_typing

from pycrograd import grad

GradFunction = typing.Callable[["Tensor"], None]


def _default_backward(out: "Tensor") -> None:
    return


class Tensor:
    def __init__(
        self,
        data: np_typing.NDArray[np.float32],
        requires_grad: bool = False,
        _op: str = "",
        _children: typing.Sequence["Tensor"] = (),
        _grad_args: typing.Sequence[typing.Any] = (),
    ) -> None:
        assert len(data.shape) == 2

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

        self.op = _op
        self.prev = _children
        self.grad_args: typing.Sequence[typing.Any] = _grad_args
        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros((self.rows, self.cols))
        else:
            self.grad = None
        self._backward: GradFunction = _default_backward

    @staticmethod
    def create_zeroed(rows: int, cols: int, requires_grad: bool = False) -> "Tensor":
        return Tensor.create_with_value(
            0.0, rows=rows, cols=cols, requires_grad=requires_grad
        )

    @staticmethod
    def create_with_value(
        value: float, rows: int, cols: int, requires_grad: bool = False
    ) -> "Tensor":
        data = np.array(
            [[value for _col in range(cols)] for _row in range(rows)]
        ).astype(np.float32)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def create_vector(
        it: typing.Iterable[float], requires_grad: bool = False
    ) -> "Tensor":
        data = np.array([[item] for item in it]).astype(np.float32)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def create_scalar(value: float, requires_grad: bool = False) -> "Tensor":
        return Tensor(
            np.array([[value]]).astype(np.float32), requires_grad=requires_grad
        )

    def backward(self) -> None:
        backward_topology: list[Tensor] = []
        visited: set[Tensor] = set()
        q: collections.deque[Tensor] = collections.deque()
        q.append(self)
        while q:
            v = q.pop()
            visited.add(v)
            if all(child in visited for child in v.prev):
                backward_topology.append(v)
            else:
                q.append(v)
                for child in v.prev:
                    if child not in visited:
                        q.append(child)

        # go one variable at a time and apply the chain rule to get its gradient
        self._set_grad_to_ones()
        for v in reversed(backward_topology):
            v._backward(v)

    def item(self) -> float:
        assert self._is_scalar()

        return self.data[0, 0]

    def mean(self) -> "Tensor":
        sum = self.sum()
        mean = sum / (self.rows * self.cols)
        return mean

    def sum(self) -> "Tensor":
        sum = 0
        for row in range(self.rows):
            for col in range(self.cols):
                sum += self[row, col]

        out = Tensor.create_scalar(sum, requires_grad=self.requires_grad)
        out._set_grad_metadata(grad.sum_backward, children=(self,), op="sum")
        return out

    def log(self) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = np.log(self.data[row, col])

        out._set_grad_metadata(backward=grad.log_backward, children=(self,), op="log")
        return out

    def concat(self, other: "Tensor") -> "Tensor":
        assert self.rows == other.rows

        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols + other.cols,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col]

        for row in range(other.rows):
            for col in range(other.cols):
                out[row, col + self.cols] = other[row, col]

        out._set_grad_metadata(
            backward=grad.concat_backward, children=(self, other), op="concat"
        )
        return out

    def relu(self) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = max(self.data[row][col], 0)

        out._set_grad_metadata(backward=grad.relu_backward, children=(self,), op="relu")
        return out

    # Sources
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    # https://insidelearningmachines.com/cross_entropy_loss/
    # https://binpord.github.io/2021/09/26/softmax_backprop.html
    def softmax(self) -> "Tensor":
        assert self._is_vector()

        max_value = np.max(self.data)
        stabilized_original = self.data - max_value
        exponentials: np_typing.NDArray[np.float32] = np.exp(stabilized_original)
        sum_exponentials: np.float32 = np.sum(exponentials.data)

        probabilities = exponentials / sum_exponentials

        out = Tensor(
            probabilities,
            requires_grad=self.requires_grad,
        )

        out._set_grad_metadata(
            backward=grad.softmax_backward, children=(self,), op="softmax"
        )
        return out

    def log_softmax(self) -> "Tensor":
        assert self._is_vector()

        max_value = np.max(self.data)
        stabilized_original = self.data - max_value
        exponentials = np.exp(stabilized_original)
        sum_exponentials = np.sum(exponentials.data)

        log_sum_exponentials = np.log(sum_exponentials)
        log_probabilities_data = stabilized_original - log_sum_exponentials

        log_probabilities = Tensor(
            log_probabilities_data,
            requires_grad=self.requires_grad,
        )
        log_probabilities._set_grad_metadata(
            backward=grad.log_softmax_backward, children=(self,), op="softmax"
        )

        return log_probabilities

    def exp(self) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = np.exp(self.data[row][col])

        out._set_grad_metadata(backward=grad.exp_backward, children=(self,), op="log")
        return out

    def __getitem__(self, indexes: tuple[int, int]) -> float:
        row, col = indexes
        return self.data[row][col]

    def __setitem__(self, indexes: tuple[int, int], value: float) -> None:
        row, col = indexes
        self.data[row][col] = value

    def __matmul__(self, other: "Tensor") -> "Tensor":
        assert self.cols == other.rows

        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=other.cols,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        for m in range(out.rows):
            for k in range(self.cols):
                for n in range(out.cols):
                    out[m, n] += self[m, k] * other[k, n]

        out._set_grad_metadata(
            backward=grad.matmul_backward, children=(self, other), op="@"
        )
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        assert self.rows == other.rows
        assert self.cols == other.cols

        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = self[row, col] + other[row, col]

        out._set_grad_metadata(
            backward=grad.addition_backward, children=(self, other), op="+"
        )
        return out

    def __mul__(self, other: int | float) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            requires_grad=self.requires_grad,
        )
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] * other

        out._set_grad_metadata(
            backward=grad.mul_backward,
            children=(self,),
            op="*",
            grad_args=(other,),
        )
        return out

    def __pow__(self, other: int | float) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = self[row, col] ** other

        out._set_grad_metadata(
            backward=grad.power_backward,
            children=(self,),
            op=f"**{other}",
            grad_args=(other,),
        )
        return out

    def __neg__(self) -> "Tensor":  # -self
        return self * -1.0

    def __sub__(self, other: "Tensor") -> "Tensor":  # self - other
        return self + (-other)

    def __truediv__(self, other: int | float) -> "Tensor":  # self / other
        return self * other**-1

    def __radd__(self, other: "Tensor") -> "Tensor":  # other + self
        return self + other

    def __rmatmul__(self, other: "Tensor") -> "Tensor":  # other @ self
        return other @ self

    def __rsub__(self, other: "Tensor") -> "Tensor":  # other - self
        return other + (-self)

    def __rmul__(self, other: int | float) -> "Tensor":  # other * self
        return self * other

    def __rtruediv__(self, other: int | float) -> "Tensor":  # other / self
        return other * self**-1

    def __len__(self) -> int:
        return self.rows * self.cols

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("  #
            f"rows={self.rows}, "  #
            f"cols={self.cols}, "  #
            f"requires_grad={self.requires_grad}, "  #
            f"_op={self.op})"
        )
        return _repr

    def _set_grad_metadata(
        self,
        backward: GradFunction,
        children: typing.Sequence["Tensor"],
        grad_args: typing.Sequence[typing.Any] = (),
        op: str = "",
    ) -> None:
        if self.requires_grad:
            self._backward = backward
            self.prev = children
            self.grad_args = grad_args
            self.op = op

    def _set_grad_to_ones(self) -> None:
        assert self.grad is not None

        for row in range(self.rows):
            for col in range(self.cols):
                self.grad[row, col] = 1

    def _is_scalar(self) -> bool:
        return self.rows == 1 and self.cols == 1

    def _is_vector(self) -> bool:
        return self.cols == 1
