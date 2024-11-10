import typing

import numpy as np
import torchvision.transforms as transforms
from numpy import typing as np_typing
from sklearn import datasets
from sklearn.utils import Bunch
from torch import Tensor
from torchvision import datasets as torch_datasets

from pycrograd import tensor

DataTargetTuple = tuple[list[tensor.Matrix], list[tensor.Matrix]]


class Dataset(typing.Protocol):
    def get_train_dataloader(self) -> typing.Iterator[DataTargetTuple]: ...

    def get_validation_dataloader(self) -> typing.Iterator[DataTargetTuple]: ...


class MoonsData:
    def __init__(self, length: int) -> None:
        X, y = datasets.make_moons(n_samples=length, noise=0.1)
        self.data = [tensor.Matrix.create_vector(x) for x in X]
        self.labels: np_typing.NDArray[np.int64] = y * 2 - 1
        self._length = length

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={len(self)})"
        return _repr

    def __len__(self) -> int:
        return self._length


class DigitsData:
    def __init__(self, length: int) -> None:
        self._length = length
        mnist: Bunch = datasets.load_digits()  # type: ignore
        X = mnist.data[:100]
        X = X / 16.0
        self.data = [tensor.Matrix.create_vector(x) for x in X]
        self.labels: list[tensor.Matrix] = []
        for label in mnist.target[:length]:
            self.labels.append(self._one_hot_encode(label))

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={len(self)})"
        return _repr

    def __len__(self) -> int:
        return self._length

    def _one_hot_encode(self, value: int) -> tensor.Matrix:
        t = tensor.Matrix.create_zeroed(rows=1, cols=10)
        t[0, value] = 1.0
        return t


class MnistData(Dataset):
    def __init__(self, length: int, batch_size: int) -> None:
        self.length = length
        self.validation_length = length // 20
        self.batch_size = batch_size

        transform = transforms.ToTensor()
        mnist_train = torch_datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )
        mnist_validation = torch_datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )

        X_train: Tensor = mnist_train.data[:length]
        X_train = X_train / 255
        self.train_X = [
            tensor.Matrix.create_vector(x.flatten())  # type: ignore
            for x in X_train
        ]

        X_validation: Tensor = mnist_validation.data[: self.validation_length]
        X_validation = X_validation / 255
        self.validation_X = [
            tensor.Matrix.create_vector(x.flatten())  # type: ignore
            for x in X_validation
        ]

        self.train_labels: list[tensor.Matrix] = []
        for label in mnist_train.targets[:length]:
            self.train_labels.append(self._one_hot_encode(label.item()))  # type: ignore

        self.validation_labels: list[tensor.Matrix] = []
        for label in mnist_validation.targets[: self.validation_length]:
            self.validation_labels.append(self._one_hot_encode(label.item()))  # type: ignore

    def get_train_dataloader(
        self,
    ) -> typing.Iterator[DataTargetTuple]:
        for i in range(0, self.length, self.batch_size):
            start = i
            end = i + self.batch_size
            if end > self.length:
                end = self.length
            data_batch = self.train_X[start:end]
            target_batch = self.train_labels[start:end]
            yield data_batch, target_batch

    def get_validation_dataloader(
        self,
    ) -> typing.Iterator[DataTargetTuple]:
        for i in range(0, self.validation_length, self.batch_size):
            start = i
            end = i + self.batch_size
            if end > self.validation_length:
                end = self.validation_length
            data_batch = self.validation_X[start:end]
            target_batch = self.validation_labels[start:end]
            yield data_batch, target_batch

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__}(length={self.length})"
        return _repr

    def _one_hot_encode(self, value: int) -> tensor.Matrix:
        t = tensor.Matrix.create_zeroed(rows=1, cols=10)
        t[0, value] = 1.0
        return t
