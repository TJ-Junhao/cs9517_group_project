#!/usr/bin/env python3
import sys
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch import optim

from project.data.imageio import load_data
from project.training.train import train_neural_network
from project.training.loss import BCEDiceLoss
from project.processing.pipeline import ImagePipeline
from project.visualization.plot import plot_train_process
from project.utils.logger import setup_logger
from project.config.configuring import train_arg_parse, count_channels
from project.utils.random_setup import set_seed
from project.utils.constant import (
    SEED,
    DEVICE,
    REGULAR_CHANNEL_IN,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    get_checkpoint_path,
    get_plot_path,
)
from project.utils.registry import MODELS, FEATURE_BUILDERS
from project.utils.file_helper import ensure_dirs_exist


def train_model(
    model_name: str,
    pipe_train: ImagePipeline,
    pipe_val: ImagePipeline,
    lr: float = 1e-4,
    epochs: int = 40,
    patience: int = 5,
    min_delta: float = 1e-4,
    channel_in: int = 3,
    batch_size: int = 3,
    checkpoint_path: Path = get_checkpoint_path("model"),
    resume_path: str | None = None,
) -> tuple[nn.Module, list[float], list[float]]:
    # Get data
    train_data = pipe_train.get_data_loader(
        batch_size=batch_size, shuffle=True, seed=SEED
    )
    val_data = pipe_val.get_data_loader(batch_size=batch_size, shuffle=False, seed=SEED)

    # get model based on config
    model_factory = MODELS.get(model_name)
    if model_factory is None:
        raise NameError(f"model named {model_name} is not registered")
    model = model_factory(channel_in).to(DEVICE)

    # loss_func = alpha * BCELoss + beta * DICELoss
    loss_func = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    resume: Path | None = None
    if resume_path is not None:
        candidate = Path(resume_path)
        resume = candidate if candidate.is_absolute() else checkpoint_path / candidate

    time_start = time.perf_counter()
    model, train_log, val_log = train_neural_network(
        train_data,
        val_data,
        model,
        loss_func,
        optimizer,
        scheduler,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        device=DEVICE,
        checkpoint_dir=checkpoint_path,
        resume_path=resume,
    )
    time_end = time.perf_counter()
    logging.getLogger("train logger").info(
        f"Total Train Time: {(time_end - time_start):.4f}s"
    )
    return model, train_log, val_log


def save_model(model: nn.Module, save_to: Path) -> None:
    torch.save(model.state_dict(), save_to)


def apply_features(pipe: ImagePipeline, feature_configs: list[dict]) -> ImagePipeline:
    result = pipe

    for feature in feature_configs:
        name = feature["name"]
        params = feature.get("params", {})
        builder = FEATURE_BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"Unknown feature: {name}")
        result = result.concat(builder(pipe, **params))

    return result


def process_pipeline(
    pipe_train: ImagePipeline,
    pipe_val: ImagePipeline,
    pipe_test: ImagePipeline,
    features: list[dict],
) -> tuple[ImagePipeline, ImagePipeline, ImagePipeline]:
    ptr = apply_features(pipe_train, features)
    pv = apply_features(pipe_val, features)
    pte = apply_features(pipe_test, features)

    return ptr, pv, pte


def main() -> None:
    parameters = train_arg_parse(sys.argv[0])
    run_name = parameters.run

    plot_path = get_plot_path(run_name, None, None)
    checkpoint_path = get_checkpoint_path(run_name)
    ensure_dirs_exist(run_name)

    setup_logger(parameters.run)
    set_seed(SEED)
    pipe_train, pipe_val, pipe_test = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)

    channel_in = count_channels(parameters.features)

    if channel_in != REGULAR_CHANNEL_IN:
        pipe_train, pipe_val, pipe_test = process_pipeline(
            pipe_train, pipe_val, pipe_test, parameters.features
        )

    model, train_log, val_log = train_model(
        parameters.model,
        pipe_train,
        pipe_val,
        parameters.learning_rate,
        parameters.epoch,
        parameters.patience,
        parameters.min_delta,
        channel_in,
        parameters.batch_size,
        checkpoint_path,
        parameters.resume,
    )

    plot_train_process(
        train_log,
        val_log,
        save=True,
        save_to=plot_path,
        run=parameters.run,
        show=False,
    )

    save_model(model, checkpoint_path / "model.pt")


if __name__ == "__main__":
    main()
