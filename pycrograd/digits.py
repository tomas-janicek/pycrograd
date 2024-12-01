from pycrograd import datasets, nn, tensor, trainers
from pycrograd.nn import modules


def train_digits_on_model(
    *,
    length: int,
    epochs: int,
    model: modules.Module[tensor.Tensor, tensor.Tensor],
) -> None:
    optimizer = nn.SGD(learning_rate=0.01, parameters_dict=model.parameters())
    trainer = trainers.Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=nn.cross_entropy_loss,
        accuracy_function=nn.calculate_accuracy,
    )
    data = datasets.DigitsData(length=length)
    trainer.fit(epochs=epochs, batch_size=32, data=data)
