from pathlib import Path
import json

from project.visualization.plot import plot_rf_feature_comparison


RUNS = {
    "RGB": Path("runs/RF_RGB/performance/performance_test.json"),
    "RGB+HSV": Path("runs/RF_RGB_HSV/performance/performance_test.json"),
    "RGB+HSV+ExG": Path("runs/RF_RGB_HSV_EXG/performance/performance_test.json"),
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    metrics = {name: load_json(path) for name, path in RUNS.items()}

    labels = list(metrics.keys())
    plant_iou = [metrics[k]["plant"]["iou"] for k in labels]
    plant_f1 = [metrics[k]["plant"]["f1-score"] for k in labels]
    plant_recall = [metrics[k]["plant"]["recall"] for k in labels]
    pixel_acc = [metrics[k]["accuracy"] for k in labels]

    plot_rf_feature_comparison(
        labels=labels,
        plant_iou=plant_iou,
        plant_f1=plant_f1,
        plant_recall=plant_recall,
        pixel_acc=pixel_acc,
        save_path="comparisons/rf_feature_comparison.png",
    )

    print("[OK] Saved figure to comparisons/rf_feature_comparison.png")


if __name__ == "__main__":
    main()