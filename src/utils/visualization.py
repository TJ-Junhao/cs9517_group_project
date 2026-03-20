import os
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
            os.makedirs("Figures", exist_ok=True)
            os.makedirs("Figures/Task-4-Safty-Shield-Training", exist_ok=True)
            plt.savefig(
                save_to,
                dpi=dpi,
            )
        plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    num_labels: int = 2,
    save: bool = False,
    save_to: Path | None = None,
    dpi: int = 600,
):
    with sns.axes_style("darkgrid"), sns.plotting_context("paper"):
        _, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True, dpi=dpi)

        x_labels = [f"Class {i}" for i in range(num_labels)]
        y_labels = [f"Class {i}" for i in range(num_labels)]

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

        ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(axis="x", labelsize=9, width=0.5)
        ax.tick_params(axis="y", labelsize=9, width=0.5)
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        if save:
            os.makedirs("Figures", exist_ok=True)
            os.makedirs("Figures/Task-4-Safty-Shield-Training", exist_ok=True)
            plt.savefig(
                save_to,
                dpi=dpi,
            )
        plt.show()
