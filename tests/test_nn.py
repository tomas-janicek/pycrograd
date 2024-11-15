import numpy as np
import torch
from torch.nn import functional as F

from pycrograd import nn, tensor


def test_get_parameters() -> None:
    layer = nn.Linear(in_features=2, out_features=2)
    weights_params, bias_params = layer.parameters()

    assert weights_params.grad is not None
    assert bias_params.grad is not None
    assert layer.weights.grad is not None
    assert layer.biases.grad is not None

    assert weights_params.grad[0, 0] == 0.0
    assert bias_params.grad[0, 0] == 0.0
    assert layer.weights.grad[0, 0] == 0.0
    assert layer.biases.grad[0, 0] == 0.0

    layer.weights.grad[0, 0] = 12.0
    layer.biases.grad[0, 0] = 12.0

    assert weights_params.grad[0, 0] == 12.0
    assert bias_params.grad[0, 0] == 12.0
    assert layer.weights.grad[0, 0] == 12.0
    assert layer.biases.grad[0, 0] == 12.0


def test_cross_entropy_loss() -> None:
    input = torch.tensor([[75.0, 5.0, 20.0]])
    target = torch.tensor([1])
    loss1 = F.cross_entropy(input, target)

    input = tensor.Tensor(np.array([[75.0], [5.0], [20.0]]))
    target = tensor.Tensor(np.array([[0, 1, 0]]))
    log_probabilities = input.log_softmax()
    loss2 = nn.cross_entropy_loss(input=[log_probabilities], target=[target])

    assert loss1.item() == loss2.item()


def test_cross_entropy_loss_2() -> None:
    input = torch.tensor([[75.0, 5.0, 20.0]])
    target = torch.tensor([0])
    loss1 = F.cross_entropy(input, target)

    input = tensor.Tensor(np.array([[75.0], [5.0], [20.0]]))
    target = tensor.Tensor(np.array([[1, 0, 0]]))
    log_probabilities = input.log_softmax()
    loss2 = nn.cross_entropy_loss(input=[log_probabilities], target=[target])

    assert loss1.item() == loss2.item()


def test_cross_entropy_loss_random() -> None:
    input = torch.tensor([[75.0, 0.0, 2000.0]])
    target = torch.tensor([0])
    loss1 = F.cross_entropy(input, target)

    input = tensor.Tensor(np.array([[75.0], [0.0], [2000.0]]))
    target = tensor.Tensor(np.array([[1, 0, 0]]))
    log_probabilities = input.log_softmax()
    loss2 = nn.cross_entropy_loss(input=[log_probabilities], target=[target])

    assert loss1.item() == loss2.item()
