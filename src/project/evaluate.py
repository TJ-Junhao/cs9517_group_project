#!/usr/bin/env python3

import sys
import json
import time
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from project.config.configuring import eval_arg_parse, count_channels
from project.visualization.plot import plot_confusion_matrix
from project.processing.pipeline import ImagePipeline
from project.evaluation.metrics import predict, compute_metrics
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


def save_performance_json(perf_path: Path, mode: str, report: dict):
    with open(perf_path / f"performance_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def robustness_test(
    pipeline: ImagePipeline,
    model: nn.Module,
    run_name: str,
    mode: str,
    criteria: float,
    save: bool = True,
) -> None:
    for type_corruption, params in CORRUPTIONS.items():
        for i, p in enumerate(params, 1):
            plot_path = get_plot_path(run_name, type_corruption, i)
            perf_path = get_performance_path(run_name, type_corruption, i)
            fail_path = get_failure_path(run_name, mode, type_corruption, i)

            corrupted: ImagePipeline = getattr(pipeline, type_corruption)(**p)
            loader = corrupted.get_data_loader(batch_size=16, shuffle=False, seed=SEED)
            expected, predicted = predict(model, loader, DEVICE, criteria)
            confusion, report = compute_metrics(expected, predicted)

            ImagePipeline.from_arrays(
                corrupted.images,
                corrupted.gt,
                predicted,
            ).select_failures(10).invert().save(fail_path, True)

            plot_confusion_matrix(
                confusion,
                dpi=300,
                save=save,
                save_to=plot_path,
                run=(f"{run_name} " if run_name != "" else "") + "Confusion Matrix",
                mode=mode,
                show=False,
            )

            save_performance_json(perf_path, mode, report)


def normal_evaluation(
    pipeline: ImagePipeline,
    model: nn.Module,
    run_name: str,
    mode: str,
    criteria: float,
    save: bool = True,
):
    plot_path = get_plot_path(run_name, None, None)
    perf_path = get_performance_path(run_name, None, None)
    output_path = get_output_path(run_name, mode, None, None)
    fail_path = get_failure_path(run_name, mode, None, None)

    loader = pipeline.get_data_loader(batch_size=16, shuffle=False, seed=SEED)
    start = time.perf_counter()
    expected, predicted = predict(model, loader, DEVICE, criteria)
    confusion, report = compute_metrics(expected, predicted)
    inference_time = time.perf_counter() - start
    report["inference_time_seconds"] = inference_time

    plot_confusion_matrix(
        confusion,
        dpi=300,
        save=save,
        save_to=plot_path,
        run=(f"{run_name} " if run_name != "" else "") + "Confusion Matrix",
        mode=mode,
        show=False,
    )

    predicted_pipe = ImagePipeline.from_arrays(
        pipeline.images,
        pipeline.gt,
        predicted,
    )
    save_performance_json(perf_path, mode, report)
    predicted_pipe.invert().save(output_path)
    predicted_pipe.select_failures(10).invert().save(fail_path, True)


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
