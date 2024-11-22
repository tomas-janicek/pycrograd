import math
import random
import typing
from array import array


class Matrix:
    # Initialize taking a pointer, don't set any elements
    def __init__(
        self,
        rows: int,
        cols: int,
        data: array[float],
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.data = data

    @classmethod
    def create_zeroed(cls, rows: int, cols: int) -> "Matrix":
        data = array("f", (rows * cols) * [0.0])
        return cls(rows, cols, data)

    @classmethod
    def create_from_value(cls, value: float) -> "Matrix":
        return cls(1, 1, array("f", [value]))

    @classmethod
    def rand(cls, rows: int, cols: int) -> "Matrix":
        data = array("f")
        for _ in range(rows * cols):
            data.append(random.uniform(0, 1))
        return cls(rows, cols, data)

    @classmethod
    def randn(cls, rows: int, cols: int) -> "Matrix":
        data = array("f")
        for _ in range(rows * cols):
            data.append(random.gauss())
        return cls(rows, cols, data)

    def item(self) -> float:
        self._is_scalar()

        return self[0, 0]

    def __getitem__(self, indexes: tuple[int, int]) -> float:
        row, col = indexes
        return self.data[row * self.cols + col]

    def __setitem__(self, indexes: tuple[int, int], val: float) -> None:
        row, col = indexes
        self.data[row * self.cols + col] = val

    def __matmul__(self, other: typing.Self) -> "Matrix":
        assert self.cols == other.rows

        out = Matrix.create_zeroed(
            rows=self.rows,
            cols=other.cols,
        )

        for m in range(out.rows):
            for k in range(self.cols):
                for n in range(out.cols):
                    out[m, n] += self[m, k] * other[k, n]
        return out

    def __rmatmul__(self, other: typing.Self) -> "Matrix":  # other @ self
        return other @ self

    def __add__(self, other: typing.Self | float) -> "Matrix":
        match other:
            case Matrix():
                return self._add_matrix(other)
            case float() | int():
                return self._add_float(other)

    def __radd__(self, other: typing.Self | float) -> "Matrix":  # other + self
        return self + other

    def _add_matrix(self, other: typing.Self) -> "Matrix":
        assert self.rows == other.rows and self.cols == other.cols

        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)

        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] + other[row, col]
        return out

    def _add_float(self, other: float) -> "Matrix":
        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)

        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] + other
        return out

    def __mul__(self, other: float) -> "Matrix":
        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] * other

        return out

    def __rmul__(self, other: float) -> "Matrix":  # other * self
        return self * other

    def __pow__(self, other: float) -> "Matrix":
        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = self[row, col] * other

        return out

    def __neg__(self) -> "Matrix":  # -self
        return self * -1.0

    def __sub__(self, other: typing.Self | float) -> "Matrix":  # self - other
        return self + (-other)

    def __rsub__(self, other: typing.Self | float) -> "Matrix":  # other - self
        return other + (-self)

    def __truediv__(self, other: float) -> "Matrix":  # self / other
        return self * other**-1

    def __rtruediv__(self, other: float) -> "Matrix":  # other / self
        return other * self**-1

    def sum(self) -> "Matrix":
        total_sum = 0.0
        for row in range(self.rows):
            for col in range(self.cols):
                total_sum += self[row, col]

        out = Matrix.create_from_value(total_sum)
        return out

    def max(self) -> "Matrix":
        max_value = 0.0
        for row in range(self.rows):
            for col in range(self.cols):
                max_value = max(max_value, self[row, col])

        out = Matrix.create_from_value(max_value)
        return out

    def relu(self) -> "Matrix":
        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)
        for row in range(out.rows):
            for col in range(out.cols):
                out[row, col] = max(self[row, col], 0)

        return out

    def exp(self) -> "Matrix":
        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = math.exp(self[row, col])

        return out

    def log(self) -> "Matrix":
        out = Matrix.create_zeroed(rows=self.rows, cols=self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                out[row, col] = math.log(self[row, col])

        return out

    def argmax(self) -> int:
        self._is_one_dimensional()

        max_index_row = 0
        max_index_col = 0
        for row in range(self.rows):
            for col in range(self.cols):
                if self[row, col] > self[max_index_row, max_index_col]:
                    max_index_row = row
                    max_index_col = col

        # Because self can be both vector (vertical array) and matrix (horizontal array),
        # we return the bigger index because the other one is always equal to one.
        return max(max_index_col, max_index_row)

    def _is_scalar(self) -> None:
        assert self.rows == 1 and self.cols == 1

    def _is_one_dimensional(self) -> None:
        assert self.cols == 1 and self.rows == 1
