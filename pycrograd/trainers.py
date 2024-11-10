from datetime import datetime

import numpy as np
from profilehooks import profile

from pycrograd import datasets, nn, tensor


class MLPTrainer:
    def __init__(self, model: nn.Module, optimizer: nn.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    @profile(
        filename=f"profiling/mlp_tensor_{datetime.now().strftime("%Y-%m-%dT%H:%M")}.prof",
        stdout=False,
        dirs=True,
    )  # type: ignore
    def fit(self, epochs: int, data: datasets.MoonsData) -> None:
        for k in range(epochs):
            # forward
            # TODO: Create generic Dataloader and iterate over it
            total_loss, acc = self.loss(data)

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()

            # update (with sgd)
            self.optimizer.step()

            if k % 1 == 0:
                print(f"step {k} loss {total_loss.item()}, accuracy {acc*100}%")

    def loss(self, data: datasets.MoonsData) -> tuple[tensor.Matrix, float]:
        # forward the model to get scores
        scores = [self.model(input) for input in data.data]

        # svm "max-margin" loss
        one = tensor.Matrix.create_scalar(1)
        losses: list[tensor.Matrix] = []
        for i in range(len(data)):
            loss = (one + float(-data.labels[i]) * scores[i]).relu()
            losses.append(loss)

        data_loss = sum(losses, start=tensor.Matrix.create_scalar(0)) / len(losses)

        # L2 regularization
        reg_loss = self.get_reg_loss()
        total_loss = data_loss + reg_loss

        # also get accuracy
        accuracy: list[bool] = [
            (yi > 0.0) == (scorei.item() > 0.0)
            for yi, scorei in zip(data.labels, scores, strict=True)
        ]
        return total_loss, sum(accuracy) / len(accuracy)

    def get_reg_loss(self) -> tensor.Matrix:
        """L2 norm  places an outsize penalty on large components of the weight vector.
        This biases our learning algorithm towards models that **distribute weight evenly
        across a larger number of features**."""
        alpha = 1e-4
        summed_squared_parameters = tensor.Matrix.create_scalar(0, requires_grad=True)
        for parameters_sequence in self.model.parameters().values():
            for parameters in parameters_sequence:
                summed_squared_parameters += (parameters**2).sum()
        return alpha * summed_squared_parameters


class DigitsTrainer:
    def __init__(self, model: nn.Module, optimizer: nn.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def fit(self, epochs: int, data: datasets.DigitsData) -> None:
        for k in range(epochs):
            # forward
            # TODO: Create generic Dataloader and iterate over it
            total_loss, acc = self.loss(data)

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()

            # update (with sgd)
            self.optimizer.step()

            if k % 1 == 0:
                print(f"step {k} loss {total_loss.item()}, accuracy {acc*100}%")

    def loss(self, data: datasets.DigitsData) -> tuple[tensor.Matrix, float]:
        # forward the model to get scores
        predictions = []
        for input in data.data:
            prediction = self.model(input)
            predictions.append(prediction)

        loss = nn.cross_entropy_loss(
            input=predictions,
            target=data.labels,
            parameters_dict=self.model.parameters(),
        )

        # also get accuracy
        accuracy = nn.calculate_accuracy(input=predictions, target=data.labels)
        return loss, accuracy


class MnistTrainer:
    def __init__(self, model: nn.Module, optimizer: nn.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    @profile(
        filename=f"profiling/mlp_mnist_{datetime.now().strftime("%Y-%m-%dT%H:%M")}.prof",
        stdout=False,
        dirs=True,
    )  # type: ignore
    def fit(self, epochs: int, data: datasets.MnistData) -> None:
        for epoch in range(epochs):
            loss, accuracy = self.do_training(data)
            print(
                f"Training Epoch   {epoch}, loss {loss:3.4f}, accuracy {accuracy:.2%}"
            )

            loss, accuracy = self.do_validation(data)
            print(
                f"Validation Epoch {epoch}, loss {loss:3.4f}, accuracy {accuracy:.2%}"
            )

    def do_training(self, data: datasets.MnistData) -> tuple[float, float]:
        losses = []
        accuracies = []
        for inputs, targets in data.get_train_dataloader():
            # forward
            loss, accuracy = self.get_loss(inputs, targets)
            losses.append(loss.item())
            accuracies.append(accuracy)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # update (with sgd)
            self.optimizer.step()
        return np.array(losses).mean(), np.array(accuracies).mean()

    def do_validation(self, data: datasets.MnistData) -> tuple[float, float]:
        losses = []
        accuracies = []
        for inputs, targets in data.get_validation_dataloader():
            # forward
            loss, accuracy = self.get_loss(inputs, targets)
            losses.append(loss.item())
            accuracies.append(accuracy)

            # backward
            self.optimizer.zero_grad()
        return np.array(losses).mean(), np.array(accuracies).mean()

    def get_loss(
        self, inputs: list[tensor.Matrix], targets: list[tensor.Matrix]
    ) -> tuple[tensor.Matrix, float]:
        predictions = [self.model(input) for input in inputs]
        loss = nn.cross_entropy_loss(
            input=predictions,
            target=targets,
            parameters_dict=self.model.parameters(),
        )

        accuracy = nn.calculate_accuracy(input=predictions, target=targets)
        return loss, accuracy
