#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import torch
from torch import nn

from project.config.configuring import eval_arg_parse, count_channels
from project.visualization.plot import plot_confusion_matrix
from project.processing.pipeline import ImagePipeline
from project.evaluation.metrics import evaluate_neural_network
from project.data.imageio import load_data
from project.utils.registry import MODELS
from project.utils.file_helper import ensure_dirs_exist
from project.utils.random_setup import set_seed
from project.utils.constant import (
    SEED,
    DEVICE,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    get_checkpoint_path,
    get_plot_path,
    get_performance_path,
)


def eval_model(
    model: nn.Module,
    pipe_test: ImagePipeline,
    save: bool,
    plot_save_to: Path,
    perf_save_to: Path,
    criteria: float = 0.6,
    title: str = "",
    mode: str = "test",
) -> None:
    test_data = pipe_test.get_data_loader(batch_size=1, shuffle=False, seed=SEED)
    confusion, report = evaluate_neural_network(
        model, test_data, device=DEVICE, criteria=criteria
    )

    plot_confusion_matrix(
        confusion,
        dpi=300,
        save=save,
        save_to=plot_save_to,
        title=f"{title} " if title != "" else "" + "Confusion Matrix",
        mode=mode,
    )

    with open(perf_save_to / f"performance_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pipe_test.set_nn_clf(model).nn_predict(criteria=criteria, device=DEVICE)


def load_model(model_name: str, run_name: str, channel_in: int = 3) -> nn.Module:
    state_dict = torch.load(
        get_checkpoint_path(run_name) / "model.pt",
        map_location=DEVICE,
    )
    model_factory = MODELS.get(model_name)
    if model_factory is None:
        raise NameError(f"model named {model_name} is not registered")
    model = model_factory(channel_in).to(DEVICE)
    model.load_state_dict(state_dict)
    return model


def main():
    set_seed(SEED)
    parameters = eval_arg_parse(sys.argv[0])
    mode = parameters.mode.lower()
    pipe = None

    if mode == "test":
        _, _, pipe = load_data(None, None, TEST_PATH)
    elif mode == "train":
        pipe, _, _ = load_data(TRAIN_PATH, None, None)
    else:
        _, pipe, _ = load_data(None, VAL_PATH, None)

    run_name = parameters.title

    model = load_model(
        parameters.model,
        run_name,
        count_channels(parameters.features),
    )

    ensure_dirs_exist(run_name)
    plot_path = get_plot_path(run_name)
    performance_path = get_performance_path(run_name)

    eval_model(
        model,
        pipe,
        save=True,
        plot_save_to=plot_path,
        perf_save_to=performance_path,
        criteria=parameters.criteria,
        mode=mode,
    )


if __name__ == "__main__":
    main()
