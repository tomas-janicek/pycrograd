import os

import typer

from micrograd import nn, optimizers, trainers

cli = typer.Typer()


@cli.command(name="train_mlp")
def train_mlp(epochs: int = 100, length: int = 100) -> None:
    model = nn.MLP()
    optimizer = optimizers.SGD(learning_rate=1, parameters=model.parameters())
    trainer = trainers.MLPTrainer(model=model, optimizer=optimizer)
    data = trainers.Data(length)
    trainer.fit(epochs=epochs, data=data)


@cli.command(name="get_recommended_num_workers")
def get_recommended_num_workers() -> None:
    max_workers = min(32, (os.cpu_count() or 1))
    print(f"Recommendation for max number of workers is {max_workers}. ")


if __name__ == "__main__":
    cli()
