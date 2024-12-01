from pycrograd import enums, tensor

from . import modules


class MLP(modules.Module[tensor.Tensor, tensor.Tensor]):
    def __init__(self) -> None:
        self.l1 = modules.Linear(
            in_features=2, out_features=16, initialization=enums.Initialization.KAIMING
        )
        self.l2 = modules.Linear(
            in_features=16, out_features=16, initialization=enums.Initialization.KAIMING
        )
        self.l3 = modules.Linear(
            in_features=16, out_features=1, initialization=enums.Initialization.KAIMING
        )

    def forward(self, input: tensor.Tensor) -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        return out

    def parameters(self) -> modules.ParametersDict:
        return {
            "l1": self.l1.parameters(),
            "l2": self.l2.parameters(),
            "l3": self.l3.parameters(),
        }

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"l1={self.l1}, "
            f"l2={self.l2}, "
            f"l3={self.l3})"
        )
        return _repr


class MLPDigits(modules.Module[tensor.Tensor, tensor.Tensor]):
    def __init__(self) -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=64,
            initialization=enums.Initialization.KAIMING,
        )
        self.l2 = modules.Linear(
            in_features=64, out_features=32, initialization=enums.Initialization.KAIMING
        )
        self.l3 = modules.Linear(
            in_features=32, out_features=10, initialization=enums.Initialization.KAIMING
        )

    def forward(self, input: tensor.Tensor) -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    def parameters(self) -> modules.ParametersDict:
        return {
            "l1": self.l1.parameters(),
            "l2": self.l2.parameters(),
            "l3": self.l3.parameters(),
        }

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"l1={self.l1}, "
            f"l2={self.l2}, "
            f"l3={self.l3})"
        )
        return _repr


class MLPDigitsBigger(modules.Module[tensor.Tensor, tensor.Tensor]):
    def __init__(self) -> None:
        self.l1 = modules.Linear(
            in_features=8 * 8,
            out_features=8192,
            initialization=enums.Initialization.KAIMING,
        )
        self.l2 = modules.Linear(
            in_features=8192,
            out_features=4096,
            initialization=enums.Initialization.KAIMING,
        )
        self.l3 = modules.Linear(
            in_features=4096,
            out_features=2048,
            initialization=enums.Initialization.KAIMING,
        )
        self.l4 = modules.Linear(
            in_features=2048,
            out_features=10,
            initialization=enums.Initialization.KAIMING,
        )

    def forward(self, input: tensor.Tensor) -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        out = out.relu()
        out = self.l4.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    def parameters(self) -> modules.ParametersDict:
        return {
            "l1": self.l1.parameters(),
            "l2": self.l2.parameters(),
            "l3": self.l3.parameters(),
            "l4": self.l4.parameters(),
        }

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"l1={self.l1}, "
            f"l2={self.l2}, "
            f"l3={self.l3}, "
            f"l4={self.l4})"
        )
        return _repr


class MLPDigitsLonger(modules.Module[tensor.Tensor, tensor.Tensor]):
    def __init__(self, n_layers: int) -> None:
        assert n_layers >= 2

        self.sequential_layers: list[modules.Linear] = []
        first = modules.Linear(
            in_features=8 * 8,
            out_features=256,
            initialization=enums.Initialization.KAIMING,
        )
        self.sequential_layers.append(first)
        for _ in range(n_layers - 2):
            lx = modules.Linear(
                in_features=256,
                out_features=256,
                initialization=enums.Initialization.KAIMING,
            )
            self.sequential_layers.append(lx)
        last = modules.Linear(
            in_features=256,
            out_features=10,
            initialization=enums.Initialization.KAIMING,
        )
        self.sequential_layers.append(last)

    def forward(self, input: tensor.Tensor) -> tensor.Tensor:
        out = input
        for i, layer in enumerate(self.sequential_layers):
            out = layer.forward(out)
            if i == len(self.sequential_layers) - 1:
                out = out.log_softmax()
            else:
                out = out.relu()
        return out

    def parameters(self) -> modules.ParametersDict:
        return {
            f"l{i}": layer.parameters()
            for i, layer in enumerate(self.sequential_layers)
        }

    def __repr__(self) -> str:
        layers_string = ", ".join(
            (f"l{i}={layer}" for i, layer in enumerate(self.sequential_layers))
        )
        _repr = f"{self.__class__.__name__}({layers_string})"
        return _repr


class MLPMnist(modules.Module[tensor.Tensor, tensor.Tensor]):
    def __init__(self) -> None:
        self.l1 = modules.Linear(
            in_features=28 * 28,
            out_features=64,
            initialization=enums.Initialization.KAIMING,
        )
        self.l2 = modules.Linear(
            in_features=64,
            out_features=32,
            initialization=enums.Initialization.KAIMING,
        )
        self.l3 = modules.Linear(
            in_features=32,
            out_features=10,
            initialization=enums.Initialization.KAIMING,
        )

    def forward(self, input: tensor.Tensor) -> tensor.Tensor:
        out = self.l1.forward(input)
        out = out.relu()
        out = self.l2.forward(out)
        out = out.relu()
        out = self.l3.forward(out)
        log_probabilities = out.log_softmax()
        return log_probabilities

    def parameters(self) -> modules.ParametersDict:
        return {
            "l1": self.l1.parameters(),
            "l2": self.l2.parameters(),
            "l3": self.l3.parameters(),
        }

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"l1={self.l1}, "
            f"l2={self.l2}, "
            f"l3={self.l3})"
        )
        return _repr
