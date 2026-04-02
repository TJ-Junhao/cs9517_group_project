from project.utils.constant import (
    get_checkpoint_path,
    get_log_path,
    get_performance_path,
    get_plot_path,
    get_output_path,
    get_failure_path,
    EVALUATION_MODE,
)
from project.utils.registry import CORRUPTIONS


def ensure_dirs_exist(run_name: str) -> None:
    dirs = set(
        [
            get_checkpoint_path(run_name),
            get_log_path(run_name),
            *[
                get_plot_path(run_name, p, i)
                for p in CORRUPTIONS.keys()
                for i in range(1, len(CORRUPTIONS.get(p, [])) + 1)
            ],
            *[
                get_performance_path(run_name, p, i)
                for p in CORRUPTIONS.keys()
                for i in range(1, len(CORRUPTIONS.get(p, [])) + 1)
            ],
            *[
                get_output_path(run_name, p, m, i)
                for p in CORRUPTIONS.keys()
                for m in EVALUATION_MODE
                for i in range(1, len(CORRUPTIONS.get(p, [])) + 1)
            ],
            *[
                get_failure_path(run_name, p, m, i)
                for p in CORRUPTIONS.keys()
                for m in EVALUATION_MODE
                for i in range(1, len(CORRUPTIONS.get(p, [])) + 1)
            ],
        ]
    )

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
