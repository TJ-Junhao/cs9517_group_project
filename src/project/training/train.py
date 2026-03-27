from __future__ import annotations
import logging
from typing import Any
from pathlib import Path

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch import optim


def train_neural_network(
    train_data: DataLoader,
    val_data: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
    checkpoint_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> tuple[nn.Module, list[float], list[float]]:
    logger = logging.getLogger("train logger")

    train_log: list[float] = []
    val_log: list[float] = []

    best_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_dir / "best.pt"
        last_ckpt_path = checkpoint_dir / "last.pt"
    else:
        best_ckpt_path = None
        last_ckpt_path = None

    # resume training
    if resume_path is not None:
        resume_path = Path(resume_path)
        checkpoint: dict[str, Any] = torch.load(
            resume_path, map_location=device, weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        train_log = checkpoint.get("train_log", [])
        val_log = checkpoint.get("val_log", [])

        logger.info(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # training
        model.train()
        train_loss = 0.0

        for train_X, train_y in train_data:
            train_X = train_X.to(device)
            train_y = train_y.to(device)

            optimiser.zero_grad(set_to_none=True)

            pred_train = model(train_X)
            loss_train = loss_fn(pred_train, train_y)

            loss_train.backward()
            optimiser.step()

            train_loss += loss_train.item() * train_X.size(0)

        train_loss /= len(train_data.dataset)  # type: ignore[arg-type]
        train_log.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_X, val_y in val_data:
                val_X = val_X.to(device)
                val_y = val_y.to(device)

                pred_val = model(val_X)
                loss_val = loss_fn(pred_val, val_y)

                val_loss += loss_val.item() * val_X.size(0)

        val_loss /= len(val_data.dataset)  # type: ignore[arg-type]
        val_log.append(val_loss)

        logger.info(
            f"epoch {epoch + 1} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}"
        )

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # save last checkpoint
        if last_ckpt_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "best_loss": best_loss,
                    "patience_counter": patience_counter,
                    "train_log": train_log,
                    "val_log": val_log,
                },
                last_ckpt_path,
            )

        # early stopping + save best
        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            patience_counter = 0

            if best_ckpt_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimiser.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler is not None else None
                        ),
                        "best_loss": best_loss,
                        "patience_counter": patience_counter,
                        "train_log": train_log,
                        "val_log": val_log,
                    },
                    best_ckpt_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # load best model before return
    if best_ckpt_path is not None and best_ckpt_path.exists():
        best_checkpoint = torch.load(
            best_ckpt_path, map_location=device, weights_only=False
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])

    return model, train_log, val_log
