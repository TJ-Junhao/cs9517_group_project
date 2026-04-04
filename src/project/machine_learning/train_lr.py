from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from project.data.imageio import load_data
from project.models.logistic_regression import (
    LRConfig,
    train_logistic_regression,
    predict_pipeline,
)
from project.evaluation.metrics import compute_metrics
from project.utils.file_helper import ensure_dirs_exist
from project.utils.constant import (
    TRAIN_PATH,
    VAL_PATH,
    get_checkpoint_path,
    get_performance_path,
)


def save_performance_json(perf_path: Path, mode: str, report: dict) -> None:
    perf_path.mkdir(parents=True, exist_ok=True)
    with open(perf_path / f"performance_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def evaluate_predicted_pipe(pred_pipe):
    expected = (pred_pipe.gt.reshape(-1) == 0).astype(np.uint8)
    predicted = (pred_pipe.images.reshape(-1) > 0).astype(np.uint8)
    _, report = compute_metrics(expected, predicted)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train Logistic Regression segmentation.")
    parser.add_argument("--run", type=str, required=True, help="Run name")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="rgb_hsv_exg",
        choices=["rgb", "rgb_hsv", "rgb_hsv_exg"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs_exist(args.run)

    pipe_train, pipe_val, _ = load_data(TRAIN_PATH, VAL_PATH, None)
    assert pipe_train is not None
    assert pipe_val is not None

    cfg = LRConfig(
        max_iter=args.max_iter,
        random_state=42,
        samples_per_class=args.samples_per_class,
    )

    t0 = time.perf_counter()
    model = train_logistic_regression(
        pipe_train,
        cfg,
        feature_mode=args.feature_mode,
    )
    train_time = time.perf_counter() - t0

    ckpt_dir = get_checkpoint_path(args.run)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ckpt_dir / "lr.joblib")

    t1 = time.perf_counter()
    pred_val = predict_pipeline(
        model,
        pipe_val,
        title=f"{args.run} Validation LR",
        feature_mode=args.feature_mode,
    )
    infer_time = time.perf_counter() - t1

    report = evaluate_predicted_pipe(pred_val)
    report["train_time_seconds"] = train_time
    report["validation_inference_time_seconds"] = infer_time
    report["model_type"] = "logistic_regression"
    report["max_iter"] = args.max_iter
    report["samples_per_class"] = args.samples_per_class
    report["feature_mode"] = args.feature_mode

    save_performance_json(get_performance_path(args.run), "validation", report)

    print(f"[OK] Trained LR run={args.run}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Validation inference time: {infer_time:.2f}s")
    print("Validation accuracy:", report["accuracy"])


if __name__ == "__main__":
    main()