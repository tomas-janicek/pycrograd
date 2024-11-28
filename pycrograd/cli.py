import os

import typer

from pycrograd import datasets, nn, trainers

cli = typer.Typer()


@cli.command(name="train_mlp")
def train_mlp(epochs: int = 100) -> None:
    model = nn.MLP()
    optimizer = nn.SGD(learning_rate=0.01, parameters_dict=model.parameters())
    trainer = trainers.Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=nn.max_margin_loss,
        accuracy_function=nn.calculate_accuracy_binary,
    )
    data = datasets.MoonsData(length=1000)
    trainer.fit(epochs=epochs, batch_size=10, data=data)


@cli.command(name="train_digits")
def train_digits(epochs: int = 20) -> None:
    model = nn.MLPDigits()
    optimizer = nn.SGD(learning_rate=0.01, parameters_dict=model.parameters())
    trainer = trainers.Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    data = datasets.DigitsData(length=1000)
    trainer.fit(epochs=epochs, batch_size=32, data=data)


@cli.command(name="train_mnist")
def train_mnist(epochs: int = 5) -> None:
    model = nn.MLPMnist()
    optimizer = nn.SGD(learning_rate=0.01, parameters_dict=model.parameters())
    trainer = trainers.Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    data = datasets.MnistData(length=100)
    trainer.fit(epochs=epochs, batch_size=32, data=data)


@cli.command(name="get_recommended_num_workers")
def get_recommended_num_workers() -> None:
    max_workers = min(32, (os.cpu_count() or 1))
    print(f"Recommendation for max number of workers is {max_workers}. ")


if __name__ == "__main__":
    cli()
