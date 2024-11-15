import typing

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from numpy import typing as np_typing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from torch import Tensor
from torchvision import datasets as torch_datasets

from pycrograd import tensor

DataT = typing.TypeVar("DataT")
TargetT = typing.TypeVar("TargetT")


class Dataset(typing.Protocol[DataT, TargetT]):  # type: ignore
    def get_train_data(
        self,
    ) -> tuple[typing.Sequence[DataT], typing.Sequence[TargetT]]: ...

    def get_validation_data(
        self,
    ) -> tuple[typing.Sequence[DataT], typing.Sequence[TargetT]]: ...


class Dataloader(typing.Generic[DataT, TargetT]):
    def __init__(self, batch_size: int, dataset: Dataset[DataT, TargetT]) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_data, self.train_target = dataset.get_train_data()
        self.validation_data, self.validation_target = dataset.get_validation_data()

        self.train_length = len(self.train_data)
        self.validation_length = len(self.validation_data)

    def get_train_dataloader(
        self,
    ) -> typing.Iterator[tuple[typing.Sequence[DataT], typing.Sequence[TargetT]]]:
        for i in range(0, self.train_length, self.batch_size):
            start = i
            end = i + self.batch_size
            if end > self.train_length:
                end = self.train_length
            data_batch = self.train_data[start:end]
            target_batch = self.train_target[start:end]
            yield data_batch, target_batch

    def get_validation_dataloader(
        self,
    ) -> typing.Iterator[tuple[typing.Sequence[DataT], typing.Sequence[TargetT]]]:
        for i in range(0, self.validation_length, self.batch_size):
            start = i
            end = i + self.batch_size
            if end > self.validation_length:
                end = self.validation_length
            data_batch = self.validation_data[start:end]
            target_batch = self.validation_target[start:end]
            yield data_batch, target_batch


class MoonsData(Dataset[tensor.Tensor, int]):
    def __init__(self, length: int) -> None:
        self.length = length
        data, target = datasets.make_moons(n_samples=length, noise=0.1)
        data = [tensor.Tensor.create_vector(x) for x in data]
        target: np_typing.NDArray[np.int64] = target * 2 - 1

        (
            self.train_data,
            self.validation_data,
            self.train_target,
            self.validation_target,
        ) = train_test_split(data, target, test_size=0.2)

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={len(self)})"
        return _repr

    def __len__(self) -> int:
        return self.length

    def get_train_data(
        self,
    ) -> tuple[list[tensor.Tensor], list[int]]:
        return self.train_data, self.train_target

    def get_validation_data(
        self,
    ) -> tuple[list[tensor.Tensor], list[int]]:
        return self.validation_data, self.validation_target


class DigitsData(Dataset[tensor.Tensor, tensor.Tensor]):
    def __init__(self, length: int) -> None:
        self.length = length
        mnist: Bunch = datasets.load_digits()  # type: ignore

        data = mnist.data[:length]
        data = data / 16.0
        target = mnist.target[:length]

        train_data, validation_data, train_target, validation_target = train_test_split(
            data, target, test_size=0.2
        )

        self.train_data = [tensor.Tensor.create_vector(x) for x in train_data]
        self.validation_data = [tensor.Tensor.create_vector(x) for x in validation_data]

        self.train_target: list[tensor.Tensor] = []
        for target in train_target:
            self.train_target.append(_one_hot_encode(target))

        self.validation_target: list[tensor.Tensor] = []
        for target in validation_target:
            self.validation_target.append(_one_hot_encode(target))

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={self.length})"
        return _repr

    def get_train_data(
        self,
    ) -> tuple[list[tensor.Tensor], list[tensor.Tensor]]:
        return self.train_data, self.train_target

    def get_validation_data(
        self,
    ) -> tuple[list[tensor.Tensor], list[tensor.Tensor]]:
        return self.validation_data, self.validation_target


class MnistData(Dataset[tensor.Tensor, tensor.Tensor]):
    def __init__(self, length: int) -> None:
        self.length = length
        self.validation_length = length // 20

        mnist_train = torch_datasets.MNIST(root="data", train=True, download=True)
        mnist_validation = torch_datasets.MNIST(root="data", train=False, download=True)

        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        train_data: Tensor = mnist_train.data[:length]
        train_data = transform(train_data)
        self.train_data = [
            tensor.Tensor.create_vector(x.flatten())  # type: ignore
            for x in train_data
        ]

        validation_data: Tensor = mnist_validation.data[: self.validation_length]
        validation_data = transform(validation_data)
        self.validation_data = [
            tensor.Tensor.create_vector(x.flatten())  # type: ignore
            for x in validation_data
        ]

        self.train_labels: list[tensor.Tensor] = []
        for label in mnist_train.targets[:length]:
            self.train_labels.append(_one_hot_encode(label.item()))  # type: ignore

        self.validation_labels: list[tensor.Tensor] = []
        for label in mnist_validation.targets[: self.validation_length]:
            self.validation_labels.append(_one_hot_encode(label.item()))  # type: ignore

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={self.length})"
        return _repr

    def get_train_data(
        self,
    ) -> tuple[list[tensor.Tensor], list[tensor.Tensor]]:
        return self.train_data, self.train_labels

    def get_validation_data(
        self,
    ) -> tuple[list[tensor.Tensor], list[tensor.Tensor]]:
        return self.validation_data, self.validation_labels


def _one_hot_encode(value: int) -> tensor.Tensor:
    t = tensor.Tensor.create_zeroed(rows=1, cols=10)
    t[0, value] = 1.0
    return t
