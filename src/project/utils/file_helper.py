from project.utils.constant import (
    get_checkpoint_path,
    get_log_path,
    get_performance_path,
    get_plot_path,
)


def ensure_dirs_exist(run_name: str) -> None:
    dirs = [
        get_checkpoint_path(run_name),
        get_log_path(run_name),
        get_plot_path(run_name),
        get_performance_path(run_name),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
