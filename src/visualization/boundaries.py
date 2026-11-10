"""Frozen boundary curves and saved pointwise localisation intervals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.style import _style


def plot_final_boundaries(selection, *, output_dir=None, show_intervals=True):
    """Plot locked curves and saved pointwise intervals; never refit or interpolate CIs."""
    from matplotlib.lines import Line2D

    b = selection.boundaries
    c = selection.candidate
    h = np.asarray(b.H, dtype=float)
    if (
        len(h) < 2
        or np.any(np.diff(h) <= 0)
        or not np.isfinite(h).all()
        or not np.isfinite(b.mu1).all()
        or not np.isfinite(b.mu2).all()
    ):
        raise ValueError("Final boundary plot requires a complete ordered frozen grid")
    interval_frame = None
    if show_intervals and b.source:
        interval_frame = pd.read_csv(b.source).sort_values("H").reset_index(drop=True)
        if not np.array_equal(interval_frame.H.to_numpy(float), h) or not all(
            np.allclose(interval_frame[f"mu{s}"], getattr(b, f"mu{s}"), atol=1e-10, rtol=1e-10)
            for s in (1, 2)
        ):
            raise ValueError("Saved interval grid differs from the frozen boundaries")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(top=0.81, bottom=0.32, wspace=0.25)
    fig.suptitle(
        f"Итоговые практические границы · α={c.alpha:g} · {c.method.upper()}", y=0.98, fontsize=14
    )
    fig.text(
        0.5,
        0.90,
        f"Окна: H — {c.H}, снос a — {c.drift}, σ — {c.sigma} дней · выбор на validation",
        ha="center",
        fontsize=10,
    )
    zoom = (max(0.35, h[0]), min(0.65, h[-1]))
    notes = []
    unconfirmed = []
    plotted_intervals = []
    missing_intervals = []
    for side, values, color in (
        (1, np.asarray(b.mu1), "#2869ac"),
        (2, np.asarray(b.mu2), "#df713e"),
    ):
        status = np.asarray(b.statuses.get(side, np.full(len(h), "fitted")))
        imputed = status == "interpolated"
        uncertain = status == "fitted_unconfirmed"
        ordinary = ~imputed & ~uncertain
        weak = imputed | uncertain
        notes.append(
            f"μ{side}: " + (", ".join(f"{v:g}" for v in h[imputed]) if imputed.any() else "нет")
        )
        unconfirmed.append(
            f"μ{side}: " + (", ".join(f"{v:g}" for v in h[uncertain]) if uncertain.any() else "нет")
        )
        low = np.full(len(h), np.nan)
        high = low.copy()
        if interval_frame is not None and {f"mu{side}_low", f"mu{side}_high"} <= set(
            interval_frame
        ):
            low = interval_frame[f"mu{side}_low"].to_numpy(float)
            high = interval_frame[f"mu{side}_high"].to_numpy(float)
        bounded = np.isfinite(low) & np.isfinite(high) & ~weak
        if np.any(low[bounded] > high[bounded]):
            raise ValueError("Invalid saved boundary interval")
        missing_intervals.append(
            f"μ{side}: " + (", ".join(f"{v:g}" for v in h[~bounded]) if (~bounded).any() else "нет")
        )
        plotted_intervals.append((low, high, bounded))
        for ax in axes:
            if show_intervals and bounded.any():
                # Draw endpoints directly: these are saved localisation intervals,
                # not symmetric errors centred on the fitted practical curve.
                ax.vlines(
                    h[bounded],
                    low[bounded],
                    high[bounded],
                    color=color,
                    alpha=0.55,
                    lw=1.0,
                    zorder=2,
                    label=f"mu{side}_intervals",
                )
                ax.plot(
                    h[bounded], low[bounded], ls="none", marker="_", ms=5, color=color, alpha=0.7
                )
                ax.plot(
                    h[bounded], high[bounded], ls="none", marker="_", ms=5, color=color, alpha=0.7
                )
            for i in range(len(h) - 1):
                ax.plot(
                    h[i : i + 2],
                    values[i : i + 2],
                    "--" if weak[i] or weak[i + 1] else "-",
                    color=color,
                    lw=1.7,
                )
            # Each H has one marker for this curve. The three sets are disjoint.
            ax.scatter(h[ordinary], values[ordinary], s=16, color=color, zorder=3)
            ax.scatter(
                h[uncertain], values[uncertain], s=30, facecolors="none", edgecolors=color, zorder=4
            )
            ax.scatter(
                h[imputed],
                values[imputed],
                s=40,
                marker="D",
                facecolors="none",
                edgecolors=color,
                zorder=5,
            )
    for ax, limits, title in zip(
        axes, ((h[0], h[-1]), zoom), ("Полный диапазон H", "Увеличение около H = 0.5")
    ):
        ax.set(xlim=limits, xlabel="H", ylabel="μ", title=title)
        visible = (h >= limits[0]) & (h <= limits[1])
        ys = np.r_[
            b.mu1[visible], b.mu2[visible], np.interp(limits, h, b.mu1), np.interp(limits, h, b.mu2)
        ]
        if show_intervals:
            for low, high, bounded in plotted_intervals:
                ys = np.r_[ys, low[visible & bounded], high[visible & bounded]]
        pad = max(0.02, float(np.ptp(ys)) * 0.08)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.axhline(0, color="gray", lw=0.7)
        ax.axvline(0.5, color="gray", lw=0.7, ls=":")
        _style(ax)
    handles = [
        Line2D([], [], color="#2869ac", label="μ₁(H)"),
        Line2D([], [], color="#df713e", label="μ₂(H)"),
        Line2D(
            [], [], color="gray", marker="D", markerfacecolor="none", ls="--", label="Интерполяция"
        ),
        Line2D(
            [],
            [],
            color="gray",
            marker="o",
            markerfacecolor="none",
            ls="--",
            label="Нет ограниченного интервала",
        ),
    ]
    if show_intervals:
        levels = (
            []
            if interval_frame is None or "confidence" not in interval_frame
            else interval_frame.confidence.dropna().unique()
        )
        level = f"{100*levels[0]:g}%-е " if len(levels) == 1 else ""
        handles.append(
            Line2D(
                [], [], color="gray", marker="_", ls="-", lw=1, label=f"{level}поточечные интервалы"
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.20),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(0.02, 0.16, "Интерполированные H — " + "; ".join(notes), fontsize=9)
    fig.text(0.02, 0.12, "Без ограниченного интервала — " + "; ".join(unconfirmed), fontsize=9)
    if show_intervals:
        fig.text(
            0.02,
            0.075,
            "Сохранённые интервалы локализации; для интерполяции новые интервалы не строятся.",
            fontsize=9,
        )
        fig.text(
            0.02,
            0.035,
            "Нет сохранённого ограниченного интервала — " + "; ".join(missing_intervals),
            fontsize=8,
            wrap=True,
        )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "final_boundaries.png", dpi=180)
    return fig
