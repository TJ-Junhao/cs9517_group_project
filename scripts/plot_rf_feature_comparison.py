#!/usr/bin/env python3

import pandas as pd

from project.visualization.plot import plot_bar_chart
from project.data.json import read_json
from project.utils.constant import COMPARISON_PATH, get_performance_path

RUNS = {
    "RGB": get_performance_path("RF_RGB", None, None) / "performance_test.json",
    "RGB+HSV": get_performance_path("RF_RGB_HSV", None, None) / "performance_test.json",
    "RGB+HSV+ExG": get_performance_path("RF_RGB_HSV_EXG", None, None)
    / "performance_test.json",
}


def main():
    metrics = {name: read_json(path) for name, path in RUNS.items()}

    labels = list(metrics.keys())

    df = pd.DataFrame(
        {
            "Feature Set": labels,
            "Plant IoU": [metrics[k]["plant"]["iou"] for k in labels],
            "Plant F1": [metrics[k]["plant"]["f1-score"] for k in labels],
            "Plant Recall": [metrics[k]["plant"]["recall"] for k in labels],
            "Pixel Accuracy": [metrics[k]["accuracy"] for k in labels],
        }
    )

    plot_bar_chart(
        df,
        x="Feature Set",
        y="Metrics",
        save=True,
        save_to=COMPARISON_PATH,
        show=False,
        title="Random Forest Feature Comparisons",
        file_name="rf_feature_comparison.png",
    )

    print("[OK] Saved figure to comparisons/rf_feature_comparison.png")


if __name__ == "__main__":
    main()
