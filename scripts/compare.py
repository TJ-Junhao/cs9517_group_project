#!/usr/bin/env python3

import sys

import pandas as pd

from project.config.configuring import compare_arg_parse
from project.utils.constant import (
    RUNS_PATH,
    CONFIG_PATH,
    COMPARISON_PATH,
    get_performance_path,
    get_plot_path,
)
from project.utils.string_process import snake_to_pascal
from project.visualization.plot import plot_line_plot, plot_bar_chart
from project.data.json import read_json


def compare_between_models(config_file: str, dataset_type: str, method: str) -> None:
    config = read_json(CONFIG_PATH / config_file)
    evaluating_runs = config.keys()
    runs = []
    plant_iou = []
    soil_iou = []
    plant_f1 = []
    accuracy = []
    for run in RUNS_PATH.iterdir():
        if run.name not in evaluating_runs:
            continue
        if not run.is_dir():
            continue
        runs.append(run.name)
        perf_path = (
            get_performance_path(run.name, None, None)
            / f"performance_{dataset_type}.json"
        )
        performance = read_json(perf_path)
        plant_metrics = performance["plant"]
        soil_metrics = performance["soil"]
        plant_iou.append(plant_metrics["iou"])
        soil_iou.append(soil_metrics["iou"])
        plant_f1.append(plant_metrics["f1-score"])
        accuracy.append(performance["accuracy"])
    dataframe = pd.DataFrame(
        {
            "Models": runs,
            "Plant IoU": plant_iou,
            "Soil IoU": soil_iou,
            "Plant F1": plant_f1,
            "Pixel Accuracy": accuracy,
        }
    ).sort_values("Plant IoU", ascending=True)

    file_name = None
    title = None
    if method == "dl":
        file_name = "deep_learning_comparisons"
        title = "Model Comparisons"
    elif method == "cv":
        file_name = "classic_cv_comparisons"
        title = "Algorithm Comparisons"
    else:
        file_name = "classic_ml_comparisons"
        title = "Model Comparisons"

    plot_bar_chart(
        dataframe,
        "Models",
        "Metrics",
        save=True,
        save_to=COMPARISON_PATH,
        show=False,
        title=title,
        file_name=file_name,
    )


def compare_between_levels(run_name: str, dataset_type: str) -> None:
    perf_path = get_performance_path(run_name)
    robustness_perf = perf_path / "robustness"
    if not robustness_perf.exists():
        raise FileNotFoundError(
            "Robustness performance path does not exist, run evaluation first"
        )
    for corrupt_type in robustness_perf.iterdir():
        levels = []
        plant_ious = []
        soil_ious = []
        for level in corrupt_type.iterdir():
            json_path = level / f"performance_{dataset_type}.json"
            performance = read_json(json_path)

            levels.append(int(level.name.split("_")[-1]))
            plant_ious.append(performance["plant"]["iou"])
            soil_ious.append(performance["soil"]["iou"])
        dataframe = pd.DataFrame(
            {"Levels": levels, "Plant_IoUs": plant_ious, "Soil_IoUs": soil_ious}
        )
        plot_path = get_plot_path(run_name, corrupt_type.name, None)
        plot_line_plot(
            dataframe,
            x="Levels",
            y="IoU",
            save=True,
            save_to=plot_path,
            show=False,
            run=run_name,
            title=f"IoU at Different Level of {snake_to_pascal(corrupt_type.name)}",
            file_name=f"robustness_iou_{dataset_type}",
            palette=["#0f671f", "#bd8a1c"],
        )


def main():
    parameters = compare_arg_parse(sys.argv[0])
    mode = parameters.mode
    dataset_type = parameters.dataset
    config = parameters.config
    method = parameters.method

    if mode == "cross":
        compare_between_models(config, dataset_type, method)
    elif mode == "robustness":
        run = parameters.run
        if run is None:
            raise ValueError(
                "When comparing between robustness levels, run name must be given"
            )
        compare_between_levels(run, dataset_type)
    else:
        raise ValueError(f"Mode {mode} not supported")


if __name__ == "__main__":
    main()
