from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )


DEVICE = get_device()


def find_root() -> Path:
    p = Path(__file__).resolve()
    while not (p / "pyproject.toml").exists():
        p = p.parent
    return p


EVALUATION_MODE = ["train", "test", "validation"]
COMPARE_MODE = ["cross_model", "robustness_level"]

ROOT_PATH = find_root()
RUNS_PATH = ROOT_PATH / "runs"
COMPARISON_PATH = ROOT_PATH / "comparisons"
DATA_PATH = ROOT_PATH / "datasets" / "EWS-Dataset"
TRAIN_PATH = DATA_PATH / "train"
VAL_PATH = DATA_PATH / "validation"
TEST_PATH = DATA_PATH / "test"

CONFIG_PATH = ROOT_PATH / "configs"

SEED = 42
REGULAR_CHANNEL_IN = 3

INTENSITY_LEVEL = 256
INTENSITY_RANGE = (0, 256)

LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])


NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1, 0.2]

BLUR_LEVELS = [(3, 3), (5, 5), (7, 7), (11, 11), (15, 15)]
BRIGHTNESS_LEVELS = [-60, -30, -10, 10, 30, 60]
ROTATION_LEVELS = [15, 30, 45, 90, 135, 180]
JPEG_COMPRESSION_LEVEL = [80, 60, 40, 20, 10]


def get_run_path(run_name: str) -> Path:
    return RUNS_PATH / run_name


def get_checkpoint_path(run_name: str) -> Path:
    return get_run_path(run_name) / "checkpoints"


def get_plot_path(
    run_name: str, corruption: str | None = None, level: int | None = None
) -> Path:
    if corruption is None:
        return get_run_path(run_name) / "plots"
    if level is None:
        return get_run_path(run_name) / "plots" / "robustness" / corruption
    return (
        get_run_path(run_name) / "plots" / "robustness" / corruption / f"level_{level}"
    )


def get_performance_path(
    run_name: str, corruption: str | None = None, level: int | None = None
) -> Path:
    if corruption is None:
        return get_run_path(run_name) / "performance"
    return (
        get_run_path(run_name)
        / "performance"
        / "robustness"
        / corruption
        / f"level_{level}"
    )


def get_log_path(run_name: str) -> Path:
    return get_run_path(run_name) / "logs"


def get_output_path(
    run_name: str, mode: str, corruption: str | None = None, level: int | None = None
) -> Path:
    if corruption is None:
        return get_run_path(run_name) / "outputs" / mode
    return (
        get_run_path(run_name)
        / "outputs"
        / "robustness"
        / corruption
        / f"level_{level}"
        / mode
    )


def get_failure_path(
    run_name: str, mode: str, corruption: str | None = None, level: int | None = None
) -> Path:
    if corruption is None:
        return get_run_path(run_name) / "failures" / mode
    return (
        get_run_path(run_name)
        / "failures"
        / "robustness"
        / corruption
        / f"level_{level}"
        / mode
    )
