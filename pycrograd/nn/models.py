from pycrograd import enums, tensor

from . import modules


class MLP(modules.Module):
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

    def forward(self, input: tensor.Matrix) -> tensor.Matrix:
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


class MLPDigits(modules.Module):
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

    def forward(self, input: tensor.Matrix) -> tensor.Matrix:
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


class MLPMnist(modules.Module):
    def __init__(self) -> None:
        self.l1 = modules.Linear(
            in_features=28 * 28,
            out_features=128,
            initialization=enums.Initialization.KAIMING,
        )
        self.l2 = modules.Linear(
            in_features=128,
            out_features=64,
            initialization=enums.Initialization.KAIMING,
        )
        self.l3 = modules.Linear(
            in_features=64,
            out_features=10,
            initialization=enums.Initialization.KAIMING,
        )

    def forward(self, input: tensor.Matrix) -> tensor.Matrix:
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
