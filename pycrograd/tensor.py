import collections
import typing

import numpy as np
from numpy import typing as np_typing

from pycrograd import grads

GradFunction = typing.Callable[["Tensor"], None]


def _default_backward(out: "Tensor") -> None:
    return


class Tensor:
    def __init__(
        self,
        data: np_typing.NDArray[np.float32],
        op: str = "",
        backward: GradFunction = _default_backward,
        previous: typing.Sequence["Tensor"] = (),
        grad_args: typing.Sequence[float] = (),
        requires_grad: bool = False,
    ) -> None:
        assert len(data.shape) == 2

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = np.zeros((self.rows, self.cols))
            self.op = op
            self.prev = previous
            self.grad_args = grad_args
            self._backward = backward
        else:
            self.grad = None
            self.op = ""
            self.prev: typing.Sequence[Tensor] = ()
            self.grad_args: typing.Sequence[float] = ()
            self._backward = _default_backward

    @staticmethod
    def create_zeroed(
        rows: int,
        cols: int,
        op: str = "",
        backward: GradFunction = _default_backward,
        previous: typing.Sequence["Tensor"] = (),
        grad_args: typing.Sequence[float] = (),
        requires_grad: bool = False,
    ) -> "Tensor":
        return Tensor.create_with_value(
            0.0,
            rows=rows,
            cols=cols,
            op=op,
            backward=backward,
            previous=previous,
            grad_args=grad_args,
            requires_grad=requires_grad,
        )

    @staticmethod
    def create_with_value(
        value: float,
        rows: int,
        cols: int,
        op: str = "",
        backward: GradFunction = _default_backward,
        previous: typing.Sequence["Tensor"] = (),
        grad_args: typing.Sequence[float] = (),
        requires_grad: bool = False,
    ) -> "Tensor":
        data = np.array(
            [[value for _col in range(cols)] for _row in range(rows)]
        ).astype(np.float32)
        return Tensor(
            data,
            requires_grad=requires_grad,
            op=op,
            backward=backward,
            previous=previous,
            grad_args=grad_args,
        )

    @staticmethod
    def create_vector(
        it: typing.Iterable[float],
        op: str = "",
        backward: GradFunction = _default_backward,
        previous: typing.Sequence["Tensor"] = (),
        grad_args: typing.Sequence[float] = (),
        requires_grad: bool = False,
    ) -> "Tensor":
        data = np.array([[item] for item in it]).astype(np.float32)
        return Tensor(
            data,
            requires_grad=requires_grad,
            op=op,
            backward=backward,
            previous=previous,
            grad_args=grad_args,
        )

    @staticmethod
    def create_scalar(
        value: float,
        op: str = "",
        backward: GradFunction = _default_backward,
        previous: typing.Sequence["Tensor"] = (),
        grad_args: typing.Sequence[float] = (),
        requires_grad: bool = False,
    ) -> "Tensor":
        return Tensor(
            np.array([[value]]).astype(np.float32),
            op=op,
            backward=backward,
            previous=previous,
            grad_args=grad_args,
            requires_grad=requires_grad,
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

        out = Tensor.create_scalar(
            sum,
            op="sum",
            backward=grads.sum_backward,
            previous=(self,),
            requires_grad=self.requires_grad,
        )
        return out

    def log(self) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            op="log",
            backward=grads.log_backward,
            previous=(self,),
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = np.log(self.data[row, col])

        return out

    def concat(self, other: "Tensor") -> "Tensor":
        assert self.rows == other.rows

        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols + other.cols,
            op="concat",
            backward=grads.concat_backward,
            previous=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col]

        for row in range(other.rows):
            for col in range(other.cols):
                out[row, col + self.cols] = other[row, col]

        return out

    def relu(self) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            op="relu",
            backward=grads.relu_backward,
            previous=(self,),
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = max(self.data[row][col], 0)

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
            op="softmax",
            backward=grads.softmax_backward,
            previous=(self,),
            requires_grad=self.requires_grad,
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
            op="log_softmax",
            backward=grads.log_softmax_backward,
            previous=(self,),
            requires_grad=self.requires_grad,
        )

        return log_probabilities

    def exp(self) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            op="exp",
            backward=grads.exp_backward,
            previous=(self,),
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = np.exp(self.data[row][col])

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
            op="@",
            backward=grads.matmul_backward,
            previous=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        for m in range(out.rows):
            for k in range(self.cols):
                for n in range(out.cols):
                    out[m, n] += self[m, k] * other[k, n]

        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        assert self.rows == other.rows
        assert self.cols == other.cols

        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            op="+",
            backward=grads.addition_backward,
            previous=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = self[row, col] + other[row, col]

        return out

    def __mul__(self, other: int | float) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            op="*",
            backward=grads.mul_backward,
            previous=(self,),
            grad_args=(other,),
            requires_grad=self.requires_grad,
        )
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] * other

        return out

    def __pow__(self, other: int | float) -> "Tensor":
        out = Tensor.create_zeroed(
            rows=self.rows,
            cols=self.cols,
            op=f"**{other}",
            backward=grads.power_backward,
            previous=(self,),
            grad_args=(other,),
            requires_grad=self.requires_grad,
        )
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = self[row, col] ** other

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

    def _set_grad_to_ones(self) -> None:
        assert self.grad is not None

        for row in range(self.rows):
            for col in range(self.cols):
                self.grad[row, col] = 1

    def _is_scalar(self) -> bool:
        return self.rows == 1 and self.cols == 1

    def _is_vector(self) -> bool:
        return self.cols == 1
