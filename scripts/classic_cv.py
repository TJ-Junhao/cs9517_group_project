#!/usr/bin/env python3
import sys
from typing import Callable, Any
import time

from project.processing.pipeline import ImagePipeline
from project.utils.constant import (
    SEED,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    get_failure_path,
    get_plot_path,
    get_performance_path,
    get_output_path,
)

from project.utils.registry import CORRUPTIONS, TRADITIONAL_CV_METHODS
from project.evaluation.metrics import compute_metrics
from project.visualization.plot import plot_confusion_matrix
from project.data.json import save_performance_json
from project.config.configuring import traditional_cv_arg_parse
from project.utils.random_setup import set_seed
from project.utils.file_helper import ensure_dirs_exist


def main():
    set_seed(SEED)
    parameters = traditional_cv_arg_parse(sys.argv[0])
    ensure_dirs_exist(parameters.run)
    path = None
    if parameters.mode == "test":
        path = TEST_PATH
    elif parameters.mode == "train":
        path = TRAIN_PATH
    else:
        path = VAL_PATH

    pipe = ImagePipeline.read_from_path(path)
    if (method := TRADITIONAL_CV_METHODS.get(parameters.method)) is None:
        raise ValueError(f"The method {parameters.method} is not defined")

    normal_evaluation(
        pipeline=pipe,
        method=method,
        run_name=parameters.run,
        mode=parameters.mode,
        **parameters.kwargs,
    )

    set_seed(SEED)
    robustness_evaluation(
        pipeline=pipe,
        method=method,
        run_name=parameters.run,
        mode=parameters.mode,
        **parameters.kwargs,
    )


def robustness_evaluation(
    pipeline: ImagePipeline,
    method: Callable[..., ImagePipeline],
    run_name: str,
    mode: str,
    save: bool = True,
    **kwargs: dict[str, Any],
) -> None:
    for type_corruption, params in CORRUPTIONS.items():
        for i, p in enumerate(params, 1):
            plot_path = get_plot_path(run_name, type_corruption, i)
            perf_path = get_performance_path(run_name, type_corruption, i)
            fail_path = get_failure_path(run_name, mode, type_corruption, i)

            corrupted: ImagePipeline = getattr(pipeline, type_corruption)(**p)

            start = time.perf_counter()
            predicted, expected = method(corrupted, **kwargs).flatten()
            inference_time = time.perf_counter() - start

            confusion, report = compute_metrics(expected, predicted)
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

            if save:
                ImagePipeline.from_arrays(
                    corrupted.gt,
                    predicted,
                ).select_failures(10).invert().save(fail_path, True)
                save_performance_json(perf_path, mode, report)


def normal_evaluation(
    pipeline: ImagePipeline,
    method: Callable[..., ImagePipeline],
    run_name: str,
    mode: str,
    save: bool = True,
    **kwargs: dict[str, Any],
):
    plot_path = get_plot_path(run_name, None, None)
    perf_path = get_performance_path(run_name, None, None)
    output_path = get_output_path(run_name, mode, None, None)
    fail_path = get_failure_path(run_name, mode, None, None)

    start = time.perf_counter()
    output_pipe = method(pipeline, **kwargs)
    inference_time = time.perf_counter() - start

    predicted, expected = output_pipe.flatten()
    confusion, report = compute_metrics(expected, predicted)

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

    if save:
        predicted_pipe = ImagePipeline.from_arrays(
            pipeline.gt,
            predicted,
        )
        save_performance_json(perf_path, mode, report)
        predicted_pipe.invert().save(output_path)
        predicted_pipe.select_failures(10).invert().save(fail_path, True)


if __name__ == "__main__":
    main()
