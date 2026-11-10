"""Archived fixed-action diagnostic figure."""

import numpy as np

from market.action_audit import REGIMES


def plot_action_audit(summary, comparisons):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = {
        "cash": "Исходная область cash",
        "policy": "Исходная область policy",
        "hold": "Исходная область hold",
    }
    summary = summary.set_index("regime").reindex(REGIMES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(3)
    width = 0.24
    for j, (key, label, color) in enumerate(
        (
            ("cash", "Не покупать", "#8c939e"),
            ("hold", "Держать", "#219176"),
            ("policy", "Политика из 03 без границ", "#2869ac"),
        )
    ):
        axes[0].bar(x + (j - 1) * width, summary[f"mean_{key}"], width, label=label, color=color)
    axes[0].set(
        xticks=x,
        xticklabels=[f'{labels[k]}\nn={int(summary.loc[k,"n"])}' for k in REGIMES],
        ylabel="Средняя лог-награда за блок",
        title="Три фиксированных действия на одинаковых блоках",
    )
    axes[0].legend(frameon=False, fontsize=9)
    subset = (
        comparisons[comparisons.a.eq("policy") & comparisons.b.eq("hold")]
        .set_index("regime")
        .reindex(REGIMES)
    )
    axes[1].scatter(x, subset.mean_diff, color="#2869ac", zorder=3)
    axes[1].vlines(x, subset.boot_low, subset.boot_high, color="#2869ac", lw=2)
    axes[1].set(
        xticks=x,
        xticklabels=[k for k in REGIMES],
        ylabel="Разность средней лог-награды",
        title="Политика из 03 без границ − удержание\nБутстреп-интервалы по полной хронологии",
    )
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    return fig
