import random
import typing

from micrograd.engine import Value


class Module(typing.Protocol):
    def parameters(self) -> typing.Sequence[Value]: ...

    def forward(self, input: typing.Sequence[Value]) -> typing.Sequence[Value]: ...

    def __call__(self, input: typing.Sequence[Value]) -> typing.Sequence[Value]:
        return self.forward(input)


class Neuron(Module):
    def __init__(self, in_features: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Value(0)

    def forward(self, input: typing.Sequence[Value]) -> typing.Sequence[Value]:
        out = sum((wi * xi for wi, xi in zip(self.w, input, strict=True)), self.b)
        return [out]

    def parameters(self) -> typing.Sequence[Value]:
        return [*self.w, self.b]

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("  #
            f"in_features={len(self.w)})"
        )
        return _repr


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def forward(self, input: typing.Sequence[Value]) -> typing.Sequence[Value]:
        out = [n.forward(input)[0] for n in self.neurons]
        return out

    def parameters(self) -> typing.Sequence[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        neurons = ", ".join(str(n) for n in self.neurons)
        _repr = (
            f"{self.__class__.__name__}("  #
            f"neurons={neurons})"
        )
        return _repr


class MLP(Module):
    def __init__(self) -> None:
        self.l1 = Linear(in_features=2, out_features=16)
        self.l2 = Linear(in_features=16, out_features=16)
        self.l3 = Linear(in_features=16, out_features=1)

    def forward(self, input: typing.Sequence[Value]) -> typing.Sequence[Value]:
        out = self.l1.forward(input)
        out = relu(out)
        out = self.l2.forward(out)
        out = relu(out)
        out = self.l3.forward(out)
        return out

    def parameters(self) -> typing.Sequence[Value]:
        return [
            *self.l1.parameters(),
            *self.l2.parameters(),
            *self.l3.parameters(),
        ]

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}("
            f"l1={self.l1}, "
            f"l2={self.l2}, "
            f"l3={self.l3})"
        )
        return _repr


def relu(input: typing.Sequence[Value]) -> typing.Sequence[Value]:
    return [i.relu() for i in input]
