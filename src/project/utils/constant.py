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


ROOT_PATH = find_root()
RUNS_PATH = ROOT_PATH / "runs"

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


def get_run_path(title: str) -> Path:
    return RUNS_PATH / title


def get_checkpoint_path(title: str) -> Path:
    return get_run_path(title) / "checkpoints"


def get_plot_path(title: str) -> Path:
    return get_run_path(title) / "plots"


def get_performance_path(title: str) -> Path:
    return get_run_path(title) / "performance"


def get_log_path(title: str) -> Path:
    return get_run_path(title) / "logs"


def get_output_path(title: str) -> Path:
    return get_run_path(title) / "outputs"


def get_failure_path(title: str) -> Path:
    return get_run_path(title) / "failures"
