from pathlib import Path
import json

from project.visualization.plot import plot_classifier_comparison


RUNS = {
    "RF": {
        "test": Path("runs/RF_RGB_HSV_EXG/performance/performance_test.json"),
        "val": Path("runs/RF_RGB_HSV_EXG/performance/performance_validation.json"),
    },
    "LR": {
        "test": Path("runs/LR_RGB_HSV_EXG/performance/performance_test.json"),
        "val": Path("runs/LR_RGB_HSV_EXG/performance/performance_validation.json"),
    },
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    metrics = {}
    for model_name, paths in RUNS.items():
        metrics[model_name] = {
            "test": load_json(paths["test"]),
            "val": load_json(paths["val"]),
        }

    labels = list(metrics.keys())

    plant_iou = [metrics[k]["test"]["plant"]["iou"] for k in labels]
    plant_f1 = [metrics[k]["test"]["plant"]["f1-score"] for k in labels]
    inference_time = [metrics[k]["test"]["inference_time_seconds"] for k in labels]
    train_time = [metrics[k]["val"]["train_time_seconds"] for k in labels]

    plot_classifier_comparison(
        labels=labels,
        plant_iou=plant_iou,
        plant_f1=plant_f1,
        train_time=train_time,
        inference_time=inference_time,
        save_path="comparisons/classifier_comparison.png",
    )

    print("[OK] Saved figure to comparisons/classifier_comparison.png")


if __name__ == "__main__":
    main()