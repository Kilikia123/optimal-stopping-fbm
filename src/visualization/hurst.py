"""Complete-strategy advantage over holding by estimated H."""

import numpy as np


def plot_hurst_performance(rows, tables):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), layout="constrained")
    labels = {"model": "Стратегия: границы + модель"}
    colors = {"model": "#2869ac"}
    groups = tables["groups"].query("strategy == 'model'")
    level = 100 * groups.confidence.iloc[0]
    for ax, family, title in zip(
        axes[:2], ("distance", "side"), ("Удаление оценки H от 0,5", "Ниже и выше 0,5")
    ):
        subset = groups[groups.family.eq(family)]
        names = subset.group.drop_duplicates().tolist()
        strategies = subset.strategy.drop_duplicates().tolist()
        for j, strategy in enumerate(strategies):
            part = subset[subset.strategy.eq(strategy)].set_index("group").reindex(names)
            x = np.arange(len(names)) + (j - (len(strategies) - 1) / 2) * 0.16
            ax.scatter(x, part.mean_diff, color=colors[strategy], label=labels[strategy], zorder=3)
            ax.vlines(x, part.boot_low, part.boot_high, color=colors[strategy], lw=2)
        counts = subset.drop_duplicates("group").set_index("group").n
        ax.set(
            xticks=np.arange(len(names)),
            xticklabels=[f"{name}\nn={counts[name]}" for name in names],
            title=title,
            ylabel="Разница с buy-and-hold за блок",
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes[:2]:
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[2].hist(rows.H, bins=25, color="#6b9dbd", edgecolor="white")
    axes[2].axvline(0.5, color="#c54b42", ls="--", label="H = 0,5")
    axes[2].set(title="Распределение оценённого H", xlabel="Оценка H", ylabel="Число блоков")
    axes[2].legend(frameon=False)
    fig.suptitle(
        f"Оценённый H и результат стратегии: {level:g}%-е разведочные интервалы\n"
        "Пороги не подбираются; неопределённость оценки самого H не учитывается"
    )
    return fig
