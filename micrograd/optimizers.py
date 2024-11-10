import typing

from micrograd.engine import Value


class Optimizer(typing.Protocol):
    def step(self) -> None: ...

    def zero_grad(self) -> None: ...


class SGD(Optimizer):
    def __init__(
        self, parameters: typing.Sequence[Value], learning_rate: float
    ) -> None:
        self.parameters = parameters
        self.original_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.steps = 0

    def step(self) -> None:
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad
        self.learning_rate = 1.0 - 0.9 * self.steps / 100
        self.steps += 1

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = 0

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"parameters_len={len(self.parameters)}, "
            f"learning_rate={self.original_learning_rate})"
        )
        return _repr
