import numpy as np

from pycrograd import tensor

from . import modules


def cross_entropy_loss(
    input: list[tensor.Matrix],
    target: list[tensor.Matrix],
    parameters_dict: modules.ParametersDict | None = None,
) -> tensor.Matrix:
    losses: list[tensor.Matrix] = []
    for i in range(len(input)):
        loss = target[i] @ input[i]
        losses.append(loss)

    data_loss = -sum(losses, start=tensor.Matrix.create_scalar(0)) / len(losses)

    # L2 regularization
    if parameters_dict:
        reg_loss = get_reg_loss(parameters_dict)
        return data_loss + reg_loss

    return data_loss


def get_reg_loss(parameters_dict: modules.ParametersDict) -> tensor.Matrix:
    """L2 norm  places an outsize penalty on large components of the weight vector.
    This biases our learning algorithm towards models that **distribute weight evenly
    across a larger number of features**."""
    alpha = 1e-4
    summed_squared_parameters = tensor.Matrix.create_scalar(0, requires_grad=True)
    for parameters_sequence in parameters_dict.values():
        for parameters in parameters_sequence:
            summed_squared_parameters += (parameters**2).sum()
    return alpha * summed_squared_parameters


def calculate_accuracy(
    input: list[tensor.Matrix],
    target: list[tensor.Matrix],
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
