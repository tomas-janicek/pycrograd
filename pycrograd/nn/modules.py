import typing

from pycrograd import enums, tensor

from . import init

ParametersDict = typing.Mapping[str, typing.Sequence[tensor.Matrix]]


class Module(typing.Protocol):
    def parameters(self) -> ParametersDict: ...

    def forward(self, input: tensor.Matrix) -> tensor.Matrix: ...

    def __call__(self, input: tensor.Matrix) -> tensor.Matrix:
        return self.forward(input)


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initialization: enums.Initialization = enums.Initialization.NORMAL,
    ) -> None:
        match initialization:
            case enums.Initialization.NORMAL:
                self.weights = init.create_normal_weights(
                    rows=out_features, cols=in_features
                )
                self.biases = init.create_normal_weights(rows=out_features, cols=1)
            case enums.Initialization.KAIMING:
                self.weights = init.create_kaiming_normal_weighta(
                    rows=out_features, cols=in_features
                )
                self.biases = init.create_kaiming_normal_weighta(
                    rows=out_features, cols=1
                )

    def forward(self, input: tensor.Matrix) -> tensor.Matrix:
        return self.weights @ input + self.biases

    def parameters(self) -> typing.Sequence[tensor.Matrix]:
        return (self.weights, self.biases)

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("  #
            f"weights={self.weights}, "
            f"bias={self.biases})"
        )
        return _repr
