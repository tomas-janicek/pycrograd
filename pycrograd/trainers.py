import typing
from datetime import datetime

import numpy as np
from profilehooks import profile

from pycrograd import datasets, nn, tensor

DataT = typing.TypeVar("DataT")
TargetT = typing.TypeVar("TargetT")
OutputT = typing.TypeVar("OutputT")


class Trainer(typing.Generic[DataT, TargetT]):
    def __init__(
        self,
        model: nn.Module[DataT, OutputT],
        optimizer: nn.Optimizer,
        loss_function: nn.LossFunction[OutputT, TargetT],
        accuracy_function: nn.AccuracyFunction[OutputT, TargetT],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function

    @profile(
        filename=f"profiling/training_{datetime.now().strftime("%Y-%m-%dT%H:%M")}.prof",
        stdout=False,
        dirs=True,
    )  # type: ignore
    def fit(
        self,
        epochs: int,
        batch_size: int,
        data: datasets.Dataset[DataT, TargetT],
    ) -> None:
        dataloader = datasets.Dataloader(batch_size, dataset=data)

        for epoch in range(epochs):
            loss, accuracy = self.do_training(dataloader)
            print(
                f"Training Epoch   {epoch:3}, loss {loss:3.4f}, accuracy {accuracy:3.2%}"
            )

            loss, accuracy = self.do_validation(dataloader)
            print(
                f"Validation Epoch {epoch:3}, loss {loss:3.4f}, accuracy {accuracy:3.2%}"
            )

    def do_training(
        self, data: datasets.Dataloader[DataT, TargetT]
    ) -> tuple[float, float]:
        losses: list[float] = []
        accuracies: list[float] = []
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

    def do_validation(
        self, data: datasets.Dataloader[DataT, TargetT]
    ) -> tuple[float, float]:
        losses: list[float] = []
        accuracies: list[float] = []
        for inputs, targets in data.get_validation_dataloader():
            # forward
            loss, accuracy = self.get_loss(inputs, targets)
            losses.append(loss.item())
            accuracies.append(accuracy)
        return np.array(losses).mean(), np.array(accuracies).mean()

    def get_loss(
        self,
        inputs: typing.Sequence[DataT],
        targets: typing.Sequence[TargetT],
    ) -> tuple[tensor.Tensor, float]:
        predictions = [self.model(input) for input in inputs]
        loss = self.loss_function(
            input=predictions,
            target=targets,
            parameters_dict=self.model.parameters(),
        )

        accuracy = self.accuracy_function(input=predictions, target=targets)
        return loss, accuracy
