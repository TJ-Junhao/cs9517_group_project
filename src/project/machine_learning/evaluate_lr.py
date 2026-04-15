from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from project.models.logistic_regression import predict_pipeline
from project.evaluation.metrics import compute_metrics
from project.utils.file_helper import ensure_dirs_exist
from project.data.json import save_performance_json
from project.processing.pipeline import ImagePipeline
from project.utils.constant import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    get_checkpoint_path,
    get_output_path,
    get_performance_path,
    get_failure_path,
)

Mode = Literal["train", "validation", "test"]


def get_pipe(mode: Mode):
    if mode == "train":
        pipe, _, _ = ImagePipeline.load_data(TRAIN_PATH, None, None)
        return pipe
    if mode == "validation":
        _, pipe, _ = ImagePipeline.load_data(None, VAL_PATH, None)
        return pipe
    _, _, pipe = ImagePipeline.load_data(None, None, TEST_PATH)
    return pipe


def evaluate_predicted_pipe(pred_pipe):
    expected = (pred_pipe.gt.reshape(-1) == 0).astype(np.uint8)
    predicted = (pred_pipe.images.reshape(-1) > 0).astype(np.uint8)
    _, report = compute_metrics(expected, predicted)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Logistic Regression segmentation."
    )
    parser.add_argument("-R", "--run", type=str, required=True, help="Run name")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "-fm",
        "--feature-mode",
        type=str,
        default="rgb_hsv_exg",
        choices=["rgb", "rgb_hsv", "rgb_hsv_exg"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs_exist(args.run)

    ckpt = get_checkpoint_path(args.run) / "lr.joblib"
    if not ckpt.exists():
        raise FileNotFoundError(f"Model not found: {ckpt}")

    model = joblib.load(ckpt)
    pipe = get_pipe(args.mode)
    assert pipe is not None

    t0 = time.perf_counter()
    pred_pipe = predict_pipeline(
        model,
        pipe,
        title=f"{args.run} {args.mode} LR",
        feature_mode=args.feature_mode,
    )
    infer_time = time.perf_counter() - t0

    report = evaluate_predicted_pipe(pred_pipe)
    report["inference_time_seconds"] = infer_time
    report["model_type"] = "logistic_regression"
    report["feature_mode"] = args.feature_mode

    perf_path = get_performance_path(args.run)
    output_path = get_output_path(args.run, args.mode, None, None)
    failure_path = get_failure_path(args.run, args.mode, None, None)

    save_performance_json(perf_path, args.mode, report)
    pred_pipe.save(output_path)
    pred_pipe.select_failures(10).save(failure_path, True)

    print(f"[OK] Evaluated LR run={args.run}, mode={args.mode}")
    print(f"Inference time: {infer_time:.2f}s")
    print("Accuracy:", report["accuracy"])


if __name__ == "__main__":
    main()
