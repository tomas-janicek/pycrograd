import typing

import numpy as np

from pycrograd import tensor

from . import modules

DataT = typing.TypeVar("DataT")
TargetT = typing.TypeVar("TargetT")


class LossFunction(typing.Protocol[DataT, TargetT]):  # type: ignore
    def __call__(
        self,
        input: typing.Sequence[DataT],
        target: typing.Sequence[TargetT],
        parameters_dict: modules.ParametersDict | None = None,
    ) -> tensor.Tensor: ...


class AccuracyFunction(typing.Protocol[DataT, TargetT]):  # type: ignore
    def __call__(
        self,
        input: typing.Sequence[DataT],
        target: typing.Sequence[TargetT],
    ) -> float: ...


def max_margin_loss(
    input: typing.Sequence[tensor.Tensor],
    target: typing.Sequence[int],
    parameters_dict: modules.ParametersDict | None = None,
) -> tensor.Tensor:
    one = tensor.Tensor.create_scalar(1)
    losses: list[tensor.Tensor] = []
    for i in range(len(input)):
        loss = (one + float(-target[i]) * input[i]).relu()
        losses.append(loss)

    data_loss = sum(losses, start=tensor.Tensor.create_scalar(0)) / len(losses)

    # L2 regularization
    if parameters_dict:
        reg_loss = get_reg_loss(parameters_dict)
        return data_loss + reg_loss

    return data_loss


def cross_entropy_loss(
    input: typing.Sequence[tensor.Tensor],
    target: typing.Sequence[tensor.Tensor],
    parameters_dict: modules.ParametersDict | None = None,
) -> tensor.Tensor:
    losses: list[tensor.Tensor] = []
    for i in range(len(input)):
        loss = target[i] @ input[i]
        losses.append(loss)

    data_loss = -sum(losses, start=tensor.Tensor.create_scalar(0)) / len(losses)

    # L2 regularization
    if parameters_dict:
        reg_loss = get_reg_loss(parameters_dict)
        return data_loss + reg_loss

    return data_loss


def get_reg_loss(parameters_dict: modules.ParametersDict) -> tensor.Tensor:
    """L2 norm  places an outsize penalty on large components of the weight vector.
    This biases our learning algorithm towards models that **distribute weight evenly
    across a larger number of features**."""
    alpha = 1e-4
    summed_squared_parameters = 0

    for parameters_sequence in parameters_dict.values():
        for parameters in parameters_sequence:
            summed_squared_parameters += (parameters**2).sum().item()
    return tensor.Tensor.create_scalar(alpha * summed_squared_parameters)


def calculate_accuracy(
    input: typing.Sequence[tensor.Tensor],
    target: typing.Sequence[tensor.Tensor],
) -> float:
    correct_predictions = 0
    all_predations = 0
    for i, t in zip(input, target, strict=True):
        # Step 1: Convert probabilities to predicted classes
        predicted_classes = np.argmax(i.data, axis=0)

        # Step 2: Convert one-hot encoded targets to class labels
        true_classes = np.argmax(t.data, axis=1)

        # Step 3: Calculate accuracy
        if predicted_classes == true_classes:
            correct_predictions += 1
        all_predations += 1

    return correct_predictions / all_predations


def calculate_accuracy_binary(
    input: typing.Sequence[tensor.Tensor],
    target: typing.Sequence[int],
) -> float:
    accuracy: list[bool] = [
        (yi > 0.0) == (scorei.item() > 0.0)
        for yi, scorei in zip(target, input, strict=True)
    ]
    return sum(accuracy) / len(accuracy)
