"""Validation diagnostics from saved synthetic rankings."""

from pathlib import Path

import numpy as np
import pandas as pd


def plot_tuning_results(output):
    """Diagnostics from saved results only."""
    import matplotlib.pyplot as plt

    output = Path(output)
    val = pd.read_csv(output / "validation_ranking.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, group in val.groupby("method"):
        axes[0].scatter(group.drift, group.score, label=method, alpha=0.6)
        axes[1].scatter(group.alpha, group.score, label=method, alpha=0.6)
        axes[2].scatter(np.sqrt(group.H_mse), group.score, label=method, alpha=0.6)
    for ax, xlabel in zip(axes, ["Окно сноса", "α", "RMSE оценки H"]):
        ax.set(xlabel=xlabel, ylabel="Средняя лог-награда на validation")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_diagnostics.png", dpi=160)
    return fig
