import typing


class Value:
    """stores a single scalar value and its gradient"""

    def __init__(
        self, data: int | float, _children: typing.Sequence["Value"] = (), _op: str = ""
    ) -> None:
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def backward(self) -> None:
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def relu(self) -> "Value":
        out = Value(max(self.data, 0), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __add__(self, other: "Value") -> "Value":
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Value") -> "Value":
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: int | float) -> "Value":
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> "Value":  # -self
        return self * Value(-1)

    def __radd__(self, other: "Value") -> "Value":  # other + self
        return self + other

    def __sub__(self, other: "Value") -> "Value":  # self - other
        return self + (-other)

    def __rsub__(self, other: "Value") -> "Value":  # other - self
        return other + (-self)

    def __rmul__(self, other: "Value") -> "Value":  # other * self
        return self * other

    def __truediv__(self, other: "Value") -> "Value":  # self / other
        return self * other**-1

    def __rtruediv__(self, other: "Value") -> "Value":  # other / self
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
