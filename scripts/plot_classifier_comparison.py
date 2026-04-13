#!/usr/bin/env python3

import pandas as pd

from project.visualization.plot import plot_bar_chart
from project.data.json import read_json
from project.utils.constant import COMPARISON_PATH, get_performance_path

RUNS = {
    "RF": {
        "test": get_performance_path("RF_RGB_HSV_EXG", None, None)
        / "performance_test.json",
        "val": get_performance_path("RF_RGB_HSV_EXG", None, None)
        / "performance_validation.json",
    },
    "LR": {
        "test": get_performance_path("LR_RGB_HSV_EXG", None, None)
        / "performance_test.json",
        "val": get_performance_path("LR_RGB_HSV_EXG", None, None)
        / "performance_validation.json",
    },
}


def main():
    metrics = {
        model: {"test": read_json(paths["test"]), "val": read_json(paths["val"])}
        for model, paths in RUNS.items()
    }

    labels = list(metrics.keys())

    df = pd.DataFrame(
        {
            "Classifier": labels,
            "Plant IoU": [metrics[k]["test"]["plant"]["iou"] for k in labels],
            "Plant F1": [metrics[k]["test"]["plant"]["f1-score"] for k in labels],
            "Training Time (s)": [metrics[k]["val"]["train_time_seconds"] for k in labels],
            "Inference Time (s)": [metrics[k]["test"]["inference_time_seconds"] for k in labels],
        }
    )

    plot_bar_chart(
        df,
        x="Classifier",
        y="Metrics",
        save=True,
        save_to=COMPARISON_PATH,
        show=False,
        title="Classifier Comparisons",
        file_name="classifier_comparison.png",
    )

    print(f"[OK] Saved figure to {COMPARISON_PATH / 'classifier_comparison.png'}")


if __name__ == "__main__":
    main()
