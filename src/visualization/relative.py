"""Relative-boundary fits and localisation status."""

import matplotlib.pyplot as plt
import numpy as np


def plot_relative_boundaries(bounds, *, alpha):
    """One fitted value per H/side; hollow markers identify missing finite bands."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 5))
    for side, color in ((1, "#2a78d6"), (2, "#eb6834")):
        h = bounds.H.to_numpy()
        values = bounds[f"mu{side}_fit"].to_numpy()
        available = np.isfinite(values)
        supported = bounds[f"mu{side}_fit_status"].isin(["fitted", "analytic"]).to_numpy()
        for i in range(len(h) - 1):
            if available[i] and available[i + 1]:
                style = "-" if supported[i] and supported[i + 1] else "--"
                ax.plot(h[i : i + 2], values[i : i + 2], style, color=color)
        ax.scatter(h[available & supported], values[available & supported], color=color, s=20)
        ax.scatter(
            h[available & ~supported],
            values[available & ~supported],
            facecolors="none",
            edgecolors=color,
            s=28,
        )
    handles = [Line2D([], [], color=c, label=f"μ{s}") for s, c in ((1, "#2a78d6"), (2, "#eb6834"))]
    handles += [
        Line2D(
            [],
            [],
            color="gray",
            marker="o",
            linestyle="-",
            label="Есть интервал локализации / аналитический якорь",
        ),
        Line2D(
            [],
            [],
            color="gray",
            marker="o",
            markerfacecolor="none",
            linestyle="--",
            label="Оценка без ограниченного интервала",
        ),
    ]
    ax.legend(handles=handles)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0.5, color="gray", linewidth=0.7)
    ax.set(xlabel="H", ylabel="μ", title=f"Монотонные относительные границы, α={alpha:g}")
    fig.tight_layout()
    return fig
