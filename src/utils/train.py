import torch
from torch import nn
from torch.nn.modules.loss import _Loss as LossFunc
from torch.utils.data.dataloader import DataLoader
from torch import optim
from torch import device
import copy


def train_neural_network(
    train_data: DataLoader,
    val_data: DataLoader,
    model: nn.Module,
    loss_fn: LossFunc,
    optimiser: optim.Optimizer,
    epochs: int,
    patience: int,
    min_delta: float,
    device: device,
) -> tuple[nn.Module, list[float], list[float]]:

    best_loss: float = float("inf")
    patience_counter: int = 0
    best_state = None
    train_log = []
    val_log = []
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0
        for train_input, train_output in train_data:
            train_input, train_output = train_input.to(device), train_output.to(device)
            # reset the gradience
            optimiser.zero_grad(set_to_none=True)

            # Compute prediction error
            pred_train = model(train_input)
            loss_train = loss_fn(pred_train, train_output)

            # Backpropagation
            loss_train.backward()
            optimiser.step()

            train_loss += loss_train.item() * train_input.size(0)
        train_loss = train_loss / len(train_data.dataset)  # type: ignore
        train_log.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_input, val_output in val_data:
                val_input, val_output = val_input.to(device), val_output.to(device)
                pred_val = model(val_input)
                loss_val = loss_fn(pred_val, val_output)
                val_loss += loss_val.item() * val_input.size(0)

        val_loss = val_loss / len(val_data.dataset)  # type: ignore

        val_log.append(val_loss)
        if (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1} | train loss: {train_loss:.6f} | val loss {val_loss:.6f}"
            )

        # early stopping
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            # if patience_counter greater than patience level: stop training
            if patience_counter >= patience:
                print(f"Early Stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_log, val_log
