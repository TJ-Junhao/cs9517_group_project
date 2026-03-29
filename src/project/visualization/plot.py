from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(
    data_frame: pd.DataFrame,
    save: bool = False,
    save_to: Path | None = None,
    title: str = "undefined",
    dpi: int = 600,
    show: bool = True,
) -> None:
    with sns.axes_style("darkgrid"), sns.plotting_context("paper"):
        _, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True, dpi=dpi)
        palette = ["#e71c22", "#0d61e8"]

        df_melt = pd.melt(data_frame, ["Epoch"])
        sns.lineplot(
            x="Epoch",
            y="value",
            hue="variable",
            data=df_melt,
            lw=1.7,
            ax=ax,
            palette=palette,
        )
        for var, color in zip(df_melt["variable"].unique(), palette):
            subset = df_melt[df_melt["variable"] == var]
            ax.fill_between(subset["Epoch"], subset["value"], color=color, alpha=0.2)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")
        ax.set_title(
            "Training and Validation Loss Curves", fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.margins(x=0, y=0)
        plt.suptitle(title, fontsize=24, fontweight="bold")
        if save:
            assert save_to is not None
            save_to.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_to / "training_and_validation_loss_curves",
                dpi=dpi,
            )
        if show:
            plt.show()


def plot_train_process(
    train_log: list,
    val_log: list,
    save: bool,
    save_to: Path,
    title: str = "",
    show: bool = True,
) -> None:
    train_data = pd.DataFrame(
        {
            "Epoch": range(1, len(train_log) + 1),
            "Training Loss": train_log,
            "Validation Loss": val_log,
        }
    )
    plot_loss_curve(train_data, save=save, save_to=save_to, title=title, show=show)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save: bool = False,
    save_to: Path | None = None,
    title: str = "undefined",
    mode: str = "test",
    dpi: int = 600,
    show: bool = True,
):
    with sns.axes_style("darkgrid"), sns.plotting_context("paper"):
        _, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True, dpi=dpi)

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

        ax.set_title(title, fontsize=15, fontweight="bold")

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
            plt.savefig(
                save_to / f"confusion_matrix_{mode}",
                dpi=dpi,
            )
        if show:
            plt.show()
