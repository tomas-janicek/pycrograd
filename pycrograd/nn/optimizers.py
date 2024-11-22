import typing

from . import modules


class Optimizer(typing.Protocol):
    def step(self) -> None: ...

    def zero_grad(self) -> None: ...


class SGD(Optimizer):
    def __init__(
        self, parameters_dict: modules.ParametersDict, learning_rate: float
    ) -> None:
        self.parameters_dict = parameters_dict
        self.learning_rate = learning_rate

    def step(self) -> None:
        for parameters_sequence in self.parameters_dict.values():
            for parameters in parameters_sequence:
                assert parameters.grad is not None
                for row in range(parameters.rows):
                    for col in range(parameters.cols):
                        parameters[row, col] -= (
                            self.learning_rate * parameters.grad[row, col]
                        )

    def zero_grad(self) -> None:
        for parameters_sequence in self.parameters_dict.values():
            for parameters in parameters_sequence:
                assert parameters.grad is not None
                for row in range(parameters.rows):
                    for col in range(parameters.cols):
                        parameters.grad[row, col] = 0

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"parameters_len={len(self.parameters_dict)}, "
            f"learning_rate={self.learning_rate})"
        )
        return _repr


class SGDVariable(Optimizer):
    def __init__(
        self, parameters_dict: modules.ParametersDict, learning_rate: float
    ) -> None:
        self.parameters_dict = parameters_dict
        self.original_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.steps = 0

    def step(self) -> None:
        for parameters_sequence in self.parameters_dict.values():
            for parameters in parameters_sequence:
                assert parameters.grad is not None
                for row in range(parameters.rows):
                    for col in range(parameters.cols):
                        parameters[row, col] -= (
                            self.learning_rate * parameters.grad[row, col]
                        )
        self.learning_rate = 1.0 - 0.9 * self.steps / 100
        self.steps += 1

    def zero_grad(self) -> None:
        for parameters_sequence in self.parameters_dict.values():
            for parameters in parameters_sequence:
                assert parameters.grad is not None
                for row in range(parameters.rows):
                    for col in range(parameters.cols):
                        parameters.grad[row, col] = 0

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"parameters_len={len(self.parameters_dict)}, "
            f"learning_rate={self.original_learning_rate})"
        )
        return _repr
