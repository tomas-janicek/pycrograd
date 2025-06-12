# Pycrograd

Pycrograd is a Python library designed for automatic differentiation and machine learning. It provides a simple and intuitive interface for building and training neural networks. This project also includes Micrograd, a minimalistic implementation for educational purposes, allowing you to compare and understand the differences between the two implementations. This implementation is mostly adopted from [Micrograd](https://github.com/karpathy/micrograd). Some changes were made to make it look more like Pycrograd and add benchmarking capabilities.

## 🎯 Motivation

I built this project to deepen my understanding of how autograd and PyTorch work, and to learn how to build a neural network from scratch. Additionally, I wanted to experiment with GPU and memory optimization, which is planned for future implementation. Another motivation was to create a baseline implementation that I can use to compare with other versions of the same code, including potential implementations in Mojo, C++, Rust, and CUDA.

## ✨ Features

- **Automatic Differentiation**: Compute gradients automatically for your models.
- **Neural Networks**: Build and train neural networks with ease.
- **Micrograd Comparison**: Includes Micrograd for performance comparison.
- **Matrix operations**: Perform matrix operations like sum, log, etc.

## 🛠️ Implementation

The cornerstone of implementations is `Tensor`, `Matrix` class and set of gradient functions. The `Tensor` class is used to store the value and gradient of a node in the computational graph. `Tensor` user `Matrix` class to hold both value and gradient data. The `Matrix` class is used to perform matrix operations like sum, log, etc. The gradient functions are used to calculate the gradients of the operations performed on the tensors.

## 🚀 Next Steps

1) Make the code run on NVIDIA and AMD GPUs.

## 📦 Installation

To install Pycrograd, you can use `uv`:

```sh
uv sync
```

## 📘 Usage

### 🏃‍♂️ Running Pycrograd

```sh
PYTHONPATH=. uv run pycrograd/cli.py train_mlp --epochs=100 --length=100
PYTHONPATH=. uv run pycrograd/cli.py train_digits --epochs=10 --length=100
PYTHONPATH=. uv run pycrograd/cli.py train_mnist --epochs=10 --length=100
```

### 🏃‍♂️ Running Micrograd

```sh
PYTHONPATH=. uv run micrograd/cli.py train_mlp --epochs=10 --length=100
```

## 🧪 Running Tests

To run the tests, use the following command:

```sh
PYTHONPATH=. uv run pytest tests/
```

## 📊 Benchmarks

All benchmarks are run on a MacBook Pro M1 with 32 GB of RAM.

### 🔍 Comparing Pycrograd with Micrograd

To run the benchmarks, use the following command:

```sh
PYTHONPATH=. uv run micrograd/cli.py train_mlp --epochs=10 --length=100
PYTHONPATH=. uv run pycrograd/cli.py train_mlp --epochs=10 --length=100
```

> [!NOTE]
> Micrograd and Pycrograd con not be compared with bigger length because Micrograd will run into recursion limit.

- Micrograd training took 94.781 seconds.
- Pycrograd training took 8.716 seconds.

#### 🚀 Why is Pycrograd faster than Micrograd?

Pycrograd is faster than Micrograd for several reasons:

- **Gradient Calculation**: In Micrograd, gradients are calculated using recursion, which can be slower and lead to issues with recursion limits. In contrast, Pycrograd uses a stack-based approach to calculate gradients, which is more efficient and avoids recursion-related problems.
- **Data Handling**: Pycrograd handles values and gradient data using arrays, allowing for more efficient operations on these lists. On the other hand, Micrograd uses a single value to hold both the value and gradient data, which can be less efficient.

These differences in implementation contribute to the performance improvements seen in Pycrograd compared to Micrograd.

### 📈 Comparing Pycrograd with Mocrograd

To run the benchmarks, use the following command:

```sh
PYTHONPATH=. uv run pycrograd/cli.py run_digits_benchmark
```

All benchmarks are run on a MacBook Pro M1 with 32 GB of RAM.

#### 🧠 Network size **[Normal]** 64 -> 64 -> 32 -> 10

- Training with Pycrograd: 121.291 seconds
- Training with Mocrograd: 20.281 seconds

#### 🧠 Network size **[Longer]** 64 [(-> 256) * 29] -> 10 (total of 30 layers)

- Training with Pycrograd: 4610.141 seconds (1 hour, 16 minutes, and 50.14 seconds)
- Training with Mocrograd: 56.842 seconds

#### 🧠 Network size **[Bigger]** 64 -> 8192 -> 4096 -> 2048 -> 10

- Training with Pycrograd: 13484.485 seconds (3 hours, 44 minutes, and 44.52 seconds)
- Training with Mocrograd: 94.953 seconds

Why is Mocrograd so much faster is explained in Mocrograd README.

# 📚 Sources

- [How Computational Graphs are Constructed in PyTorch](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
- [A Simple Introduction to Cross Entropy Loss](https://insidelearningmachines.com/cross_entropy_loss/)
- [Backpropagation through softmax layer](https://binpord.github.io/2021/09/26/softmax_backprop.html)
- [Distributed TensorFlow](https://www.oreilly.com/content/distributed-tensorflow/)
