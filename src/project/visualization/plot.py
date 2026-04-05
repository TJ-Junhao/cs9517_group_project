from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from project.utils.string_process import snake_to_pascal


def plot_line_plot(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    save: bool = False,
    save_to: Path | None = None,
    run: str = "undefined",
    dpi: int = 600,
    show: bool = True,
    title: str = "undefined",
    file_name="undefined",
    palette: list[str] | None = None,
) -> None:
    if palette is None:
        palette = ["#e71c22", "#0d61e8"]
    with sns.axes_style("darkgrid"), sns.plotting_context("paper"):
        fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True, dpi=dpi)

        df_melt = (
            dataframe.melt(id_vars=[x], value_name=y)
            .sort_values(x)
            .reset_index(drop=True)
        )
        sns.lineplot(
            x=x,
            y=y,
            hue="variable",
            data=df_melt,
            lw=1.7,
            ax=ax,
            palette=palette,
            errorbar=None,
        )
        for var, color in zip(df_melt["variable"].unique(), palette):
            subset = df_melt[df_melt["variable"] == var]
            ax.fill_between(subset[x], subset[y], color=color, alpha=0.2)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")
        ax.set_title(title, fontsize=15, fontweight="bold", y=1.04)
        ax.set_xlabel(x, fontsize=12, fontweight="bold")
        ax.set_xticks(sorted(dataframe[x].unique()))
        ax.set_ylabel(y, fontsize=12, fontweight="bold")
        ax.margins(x=0, y=0)
        plt.suptitle(run, fontsize=24, fontweight="bold", y=1.04)
        if save:
            assert save_to is not None
            save_to.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_to / file_name,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.3,
            )
        if show:
            plt.show()
        plt.close(fig)


def plot_bar_chart(
    dataframe: pd.DataFrame,
    x: str = "models",
    y: str = "metrics",
    save: bool = False,
    save_to: Path | None = None,
    dpi: int = 600,
    show: bool = True,
    title: str = "undefined",
    file_name="undefined",
):
    with sns.axes_style("darkgrid"), sns.plotting_context("paper"):
        fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True, dpi=dpi)
        df_melt = dataframe.melt(id_vars=[x], value_name=y, var_name="Metric Types")
        sns.barplot(data=df_melt, x=x, y=y, hue="Metric Types", ax=ax, errorbar="sd")
        ax.set_title(title, fontsize=24, fontweight="bold", y=1.04)
        ax.set_xlabel(x, fontsize=12, fontweight="bold")
        ax.set_ylabel(y, fontsize=12, fontweight="bold")
        min_value = df_melt["Metrics"].min()
        max_value = df_melt["Metrics"].max()
        ax.set_ylim(min_value - 0.02, max_value + 0.02)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f")  # type: ignore
        if save:
            assert save_to is not None
            save_to.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_to / file_name,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.3,
            )
        if show:
            plt.show()
        plt.close(fig)


def plot_train_process(
    train_log: list,
    val_log: list,
    save: bool,
    save_to: Path,
    run: str = "",
    show: bool = True,
) -> None:
    train_data = pd.DataFrame(
        {
            "Epoch": range(1, len(train_log) + 1),
            "Training Loss": train_log,
            "Validation Loss": val_log,
        }
    )
    plot_line_plot(
        train_data,
        "Epoch",
        "Loss",
        save=save,
        save_to=save_to,
        run=run,
        show=show,
        title="Training and Validation Loss Curves",
        file_name="training_and_validation_loss_curves",
        palette=["#e71c22", "#0d61e8"],
    )


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save: bool = False,
    save_to: Path | None = None,
    run: str = "undefined",
    mode: str = "test",
    dpi: int = 600,
    show: bool = True,
):
    with sns.axes_style("darkgrid"), sns.plotting_context("paper"):
        fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True, dpi=dpi)

        x_labels = ["Soil", "Plant"]
        y_labels = ["Soil", "Plant"]

        ax0 = sns.heatmap(
            confusion_matrix,
            annot=True,
            cmap="Blues",
            ax=ax,
            xticklabels=x_labels,
            yticklabels=y_labels,
        )
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")
        cbar1 = ax0.collections[0].colorbar
        cbar1.ax.set_ylabel("Number of Samples", rotation=270, labelpad=15, va="bottom")  # type: ignore

        plt.suptitle(run, fontsize=24, fontweight="bold", y=1.04)
        ax.set_title(snake_to_pascal(mode), fontsize=15, fontweight="bold", y=1.04)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(axis="x", labelsize=9, width=0.5)
        ax.tick_params(axis="y", labelsize=9, width=0.5)
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        if save:
            assert save_to is not None
            save_to.mkdir(exist_ok=True)
            fig.savefig(
                save_to / f"confusion_matrix_{mode}",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.3,
            )
        if show:
            plt.show()
        plt.close(fig)


def plot_rf_feature_comparison(
    labels: list[str],
    plant_iou: list[float],
    plant_f1: list[float],
    plant_recall: list[float],
    pixel_acc: list[float],
    save_path: str | Path,
) -> None:
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - 1.5 * width, plant_iou, width, label="Plant IoU")
    bars2 = ax.bar(x - 0.5 * width, plant_f1, width, label="Plant F1")
    bars3 = ax.bar(x + 0.5 * width, plant_recall, width, label="Plant Recall")
    bars4 = ax.bar(x + 1.5 * width, pixel_acc, width, label="Pixel Accuracy")

    ax.set_title("Random Forest Feature Comparisons", fontsize=18, fontweight="bold")
    ax.set_xlabel("Feature Sets", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.6, 1.0)
    ax.legend(title="Metric Types")

    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.003,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_comparison(
    labels: list[str],
    plant_iou: list[float],
    plant_f1: list[float],
    train_time: list[float],
    inference_time: list[float],
    save_path: str | Path,
) -> None:
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - 1.5 * width, plant_iou, width, label="Plant IoU")
    bars2 = ax.bar(x - 0.5 * width, plant_f1, width, label="Plant F1")
    bars3 = ax.bar(x + 0.5 * width, train_time, width, label="Training Time (s)")
    bars4 = ax.bar(x + 1.5 * width, inference_time, width, label="Inference Time (s)")

    ax.set_title("Classifier Comparisons", fontsize=18, fontweight="bold")
    ax.set_xlabel("Classifiers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Metric Types")

    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.01 if h > 1 else 0.003),
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
