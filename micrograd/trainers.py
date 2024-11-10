from datetime import datetime

from profilehooks import profile
from sklearn.datasets import make_moons

from micrograd import engine, nn, optimizers


class Data:
    def __init__(self, length: int) -> None:
        X, y = make_moons(n_samples=length, noise=0.1)
        self.data = X
        self.labels = y * 2 - 1  # make y be -1 or 1
        self._length = length

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={len(self)})"
        return _repr

    def __len__(self) -> int:
        return self._length


class MLPTrainer:
    def __init__(self, model: nn.Module, optimizer: optimizers.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    @profile(
        filename=f"profiling/mlp_{datetime.now().strftime("%Y-%m-%dT%H:%M")}.prof",
        stdout=False,
        dirs=True,
    )  # type: ignore
    def fit(self, epochs: int, data: Data) -> None:
        for k in range(epochs):
            # forward
            total_loss, acc = self.loss(data)

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()

            # update (with sgd)
            self.optimizer.step()

            if k % 1 == 0:
                print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    def loss(self, data: Data) -> tuple[engine.Value, float]:
        # inline DataLoader :)
        inputs = [[engine.Value(x) for x in xrow] for xrow in data.data]

        # forward the model to get scores
        scores = [self.model(input)[0] for input in inputs]

        # svm "max-margin" loss
        losses = [
            (engine.Value(1) + engine.Value(-yi) * scorei).relu()
            for yi, scorei in zip(data.labels, scores, strict=True)
        ]
        data_loss = sum(losses, start=engine.Value(0)) / engine.Value(len(losses))

        # L2 regularization
        alpha = engine.Value(1e-4)
        reg_loss = alpha * sum(
            (p * p for p in self.model.parameters()), engine.Value(0)
        )
        total_loss = data_loss + reg_loss

        # also get accuracy
        accuracy = [
            (yi > 0) == (scorei.data > 0)
            for yi, scorei in zip(data.labels, scores, strict=True)
        ]
        return total_loss, sum(accuracy) / len(accuracy)
