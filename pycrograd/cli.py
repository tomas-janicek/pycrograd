import os

import typer

from pycrograd import datasets, digits, nn, trainers

cli = typer.Typer()


@cli.command(name="train_mlp")
def train_mlp(epochs: int = 100, length: int = 100) -> None:
    model = nn.MLP()
    optimizer = nn.SGD(learning_rate=0.01, parameters_dict=model.parameters())
    trainer = trainers.Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=nn.max_margin_loss,
        accuracy_function=nn.calculate_accuracy_binary,
    )
    data = datasets.MoonsData(length=length)
    trainer.fit(epochs=epochs, batch_size=10, data=data)


@cli.command(name="train_digits")
def train_digits(epochs: int = 5, length: int = 10) -> None:
    model = nn.MLPDigits()
    digits.train_digits_on_model(epochs=epochs, length=length, model=model)


@cli.command(name="run_digits_benchmark")
def run_digits_benchmark(epochs: int = 10, length: int = 1000) -> None:
    print(f"\nTraining on Normal Model with {length} samples")
    model = nn.MLPDigits()
    digits.train_digits_on_model(epochs=epochs, length=length, model=model)
    longer_length = length // 10
    print(f"\nTraining on Longer Model on {longer_length} samples")
    model = nn.MLPDigitsLonger(n_layers=30)
    digits.train_digits_on_model(epochs=epochs, length=longer_length, model=model)
    bigger_length = length // 100
    print(f"\nTraining on Bigger Model on {bigger_length} samples")
    model = nn.MLPDigitsBigger()
    digits.train_digits_on_model(epochs=epochs, length=bigger_length, model=model)


@cli.command(name="train_mnist")
def train_mnist(epochs: int = 5, length: int = 100) -> None:
    model = nn.MLPMnist()
    optimizer = nn.SGD(learning_rate=0.01, parameters_dict=model.parameters())
    trainer = trainers.Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    data = datasets.MnistData(length=length)
    trainer.fit(epochs=epochs, batch_size=32, data=data)


@cli.command(name="get_recommended_num_workers")
def get_recommended_num_workers() -> None:
    max_workers = min(32, (os.cpu_count() or 1))
    print(f"Recommendation for max number of workers is {max_workers}. ")


if __name__ == "__main__":
    cli()
