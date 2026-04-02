#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from project.config.configuring import eval_arg_parse, count_channels
from project.visualization.plot import plot_confusion_matrix
from project.processing.pipeline import ImagePipeline
from project.evaluation.metrics import evaluate_neural_network
from project.data.imageio import load_data
from project.utils.registry import MODELS, CORRUPTIONS
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
    get_output_path,
    get_failure_path,
)

Mode = Literal["test"] | Literal["train"] | Literal["validation"]


def eval_model(
    model: nn.Module,
    pipe_test: ImagePipeline,
    save: bool,
    plot_save_to: Path,
    perf_save_to: Path,
    criteria: float = 0.6,
    run: str = "",
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
        run=f"{run} " if run != "" else "" + "Confusion Matrix",
        mode=mode,
        show=False,
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


def save_prediction(pipe: ImagePipeline, save_to: Path):
    pipe.save(save_to)


def failure_analysis(pipe: ImagePipeline, save_to: Path):
    pipe.select_failures(10).save(save_to, True)


def robustness_test(
    pipeline: ImagePipeline,
    model: nn.Module,
    run_name: str,
    mode: str,
    criteria: float,
) -> None:
    for type_corruption, params in CORRUPTIONS.items():
        for i, p in enumerate(params, 1):
            corrupted: ImagePipeline = getattr(pipeline, type_corruption)(**p)
            eval_model(
                model,
                corrupted,
                save=True,
                plot_save_to=get_plot_path(run_name, type_corruption, i),
                perf_save_to=get_performance_path(run_name, type_corruption, i),
                criteria=criteria,
                run=run_name,
                mode=mode,
            )
            predicted_pipe = corrupted.set_nn_clf(model).nn_predict(criteria, DEVICE)
            failure_analysis(
                predicted_pipe, get_failure_path(run_name, mode, type_corruption, i)
            )


def normal_evaluation(
    pipeline: ImagePipeline,
    model: nn.Module,
    run_name: str,
    mode: str,
    criteria: float,
):
    plot_path = get_plot_path(run_name, None, None)
    performance_path = get_performance_path(run_name, None, None)
    output_path = get_output_path(run_name, mode, None, None)
    failure_path = get_failure_path(run_name, mode, None, None)

    eval_model(
        model,
        pipeline,
        save=True,
        plot_save_to=plot_path,
        perf_save_to=performance_path,
        criteria=criteria,
        mode=mode,
    )
    predicted_pipe = pipeline.set_nn_clf(model).nn_predict(criteria, DEVICE)
    save_prediction(predicted_pipe, output_path)
    failure_analysis(predicted_pipe, failure_path)


def get_pipe(mode: Mode) -> ImagePipeline:
    pipe = None
    if mode == "test":
        _, _, pipe = load_data(None, None, TEST_PATH)
    elif mode == "train":
        pipe, _, _ = load_data(TRAIN_PATH, None, None)
    elif mode == "validation":
        _, pipe, _ = load_data(None, VAL_PATH, None)
    return pipe


def main():
    set_seed(SEED)
    parameters = eval_arg_parse(sys.argv[0])
    mode: Mode = parameters.mode.lower()
    criteria = parameters.criteria

    pipe = get_pipe(mode)

    run_name = parameters.run

    model = load_model(
        parameters.model,
        run_name,
        count_channels(parameters.features),
    )

    ensure_dirs_exist(run_name)
    normal_evaluation(pipe, model, run_name, mode, criteria)
    robustness_test(pipe, model, run_name, mode, criteria)


if __name__ == "__main__":
    main()
