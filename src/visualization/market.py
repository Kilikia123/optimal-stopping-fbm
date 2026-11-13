"""Market performance, parameter histories and regime diagnostics from saved tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from market.validation import _classification, validate_saved_results
from visualization.boundaries import plot_final_boundaries
from visualization.style import COLORS, LABELS, STRATEGIES, _style, _time


def plot_performance(result, title=""):
    dates = pd.to_datetime(result.end_date)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 2, 0.85)},
        layout="constrained",
    )
    for name, (label, color) in STRATEGIES.items():
        values = result[f"r_{name}"].cumsum().to_numpy()
        axes[0].plot(
            dates,
            values,
            label=label,
            color=color,
            linewidth=1.6 if name in ("model", "buy_hold") else 1.0,
        )
        if name in ("model", "buy_hold"):
            peaks = np.maximum.accumulate(np.r_[0.0, values])[1:]
            drawdown = np.expm1(values - peaks)
            axes[1].plot(dates, drawdown, color=color, label=label, linewidth=1.0)
            axes[1].fill_between(dates, drawdown, 0, color=color, alpha=0.08)
    axes[0].set(ylabel="Накопленная лог-доходность", title=title or "Доходность и риск")
    axes[0].legend(ncol=2, frameon=False, loc="upper left")
    axes[1].set(ylabel="Просадка от максимума")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    regimes = result.regime.to_numpy()
    edges = np.r_[0, np.flatnonzero(regimes[1:] != regimes[:-1]) + 1, len(regimes)]
    decisions = pd.to_datetime(result.decision_date)
    for a, b in zip(edges[:-1], edges[1:]):
        axes[2].fill_between(
            [decisions.iloc[a], dates.iloc[b - 1]], 0, 1, color=COLORS[regimes[a]], alpha=0.85, lw=0
        )
    axes[2].set(ylim=(0, 1.7), yticks=[], ylabel="Режим", xlabel="Дата")
    axes[2].legend(
        handles=[Patch(facecolor=COLORS[k], label=LABELS[k]) for k in COLORS],
        ncol=3,
        frameon=False,
        loc="upper center",
    )
    for ax in axes:
        _style(ax)
    _time(axes[-1])
    return fig


def plot_parameters(result, *, windows, intervals=None, title=""):
    """H, drift a, sigma and normalized drift; alpha itself is fixed."""
    dates = pd.to_datetime(result.decision_date)
    entries = [
        ("H", f"Ĥ — память, окно {windows.H} дней", "#2869ac", 0.5),
        ("a", f"â — снос за блок, окно {windows.drift} дней", "#9065a8", 0),
        ("sigma", f"σ̂ — масштаб за блок, окно {windows.sigma} дней", "#219176", None),
        ("mu", "μ̂ = â / σ̂ — нормированный снос", "#df713e", 0),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, layout="constrained")
    for ax, (key, label, color, reference) in zip(axes, entries):
        ax.plot(dates, result[key], color=color, linewidth=1.0, label="Точечная оценка")
        if intervals is not None and not intervals.empty:
            required = {"decision_date", f"{key}_low", f"{key}_high"}
            if required <= set(intervals):
                ax.vlines(
                    pd.to_datetime(intervals.decision_date),
                    intervals[f"{key}_low"],
                    intervals[f"{key}_high"],
                    color=color,
                    alpha=0.25,
                    linewidth=1.0,
                    label="Бутстреп-интервал в выбранную дату",
                )
        if reference is not None:
            ax.axhline(reference, color="#62666c", lw=0.8, ls="--")
        if key in ("a", "sigma"):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(ylabel=key, title=label)
        _style(ax)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xlabel("Дата")
    if title:
        fig.suptitle(title, fontsize=13)
    _time(axes[-1])
    return fig


def plot_boundary_diagnostics(result, boundaries):
    low, high, labels = _classification(result, boundaries)
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.4, 1), width_ratios=(2, 1))
    plane = fig.add_subplot(grid[0, 0])
    shares = fig.add_subplot(grid[0, 1])
    history = fig.add_subplot(grid[1, :])
    xmin = max(boundaries.H.min(), result.H.min() - 0.03)
    xmax = min(boundaries.H.max(), result.H.max() + 0.03)
    if xmin == xmax:
        xmin, xmax = boundaries.H.min(), boundaries.H.max()
    h = np.linspace(xmin, xmax, 500)
    b1, b2 = boundaries.mu1_at(h), boundaries.mu2_at(h)
    finite = np.r_[result.mu.to_numpy(), b1[np.isfinite(b1)], b2[np.isfinite(b2)]]
    ymin, ymax = finite.min(), finite.max()
    margin = max(0.05, (ymax - ymin) * 0.05)
    ymin -= margin
    ymax += margin
    plane.fill_between(h, ymin, b1, color=COLORS["cash"], alpha=0.12)
    plane.fill_between(h, b1, b2, color=COLORS["policy"], alpha=0.12)
    plane.fill_between(h, b2, ymax, color=COLORS["hold"], alpha=0.12)
    weak_statuses = ("interpolated", "fitted_unconfirmed")
    weak_present = False
    for side, values, color, label in (
        (1, boundaries.mu1, "#2869ac", "μ₁(H)"),
        (2, boundaries.mu2, "#df713e", "μ₂(H)"),
    ):
        status = boundaries.statuses.get(side, np.full(len(boundaries.H), "fitted"))
        first = True
        for i in range(len(boundaries.H) - 1):
            x0 = max(xmin, boundaries.H[i])
            x1 = min(xmax, boundaries.H[i + 1])
            if x1 <= x0:
                continue
            weak = status[i] in weak_statuses or status[i + 1] in weak_statuses
            weak_present |= weak
            plane.plot(
                [x0, x1],
                boundaries.mu1_at([x0, x1]) if side == 1 else boundaries.mu2_at([x0, x1]),
                "--" if weak else "-",
                color=color,
                lw=1.6,
                label=label if first else None,
            )
            first = False
    if weak_present:
        plane.text(
            0.02,
            0.02,
            "Пунктир: интерполяция / нет ограниченного интервала",
            transform=plane.transAxes,
            fontsize=8,
            color="#555555",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )
    dates = pd.to_datetime(result.decision_date)
    counts = pd.Series(labels).value_counts().reindex(COLORS, fill_value=0)
    for key, color in COLORS.items():
        mask = labels == key
        plane.scatter(
            result.H[mask],
            result.mu[mask],
            s=13,
            color=color,
            alpha=0.6,
            label=f"{LABELS[key]}: {counts[key]} ({counts[key]/len(result):.1%})",
            edgecolors="none",
        )
        history.scatter(dates[mask], result.mu[mask], s=10, color=color, zorder=3)
    plane.set(
        xlabel="Оценка H",
        ylabel="Оценка μ",
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        title="Каждый блок — точка; цвет показывает выбранный режим",
    )
    plane.legend(frameon=False, fontsize=9, loc="best")
    proportions = counts / len(result)
    shares.barh(range(3), proportions, color=list(COLORS.values()), height=0.55)
    shares.set(
        yticks=range(3),
        yticklabels=list(LABELS.values()),
        xlim=(0, max(0.7, proportions.max() * 1.3)),
        xlabel="Доля всех блоков",
        title=f"Сколько блоков в каждой области: N = {len(result)}",
    )
    shares.invert_yaxis()
    shares.xaxis.set_major_formatter(PercentFormatter(1))
    for i, key in enumerate(COLORS):
        shares.text(
            proportions[key] + 0.015, i, f"{proportions[key]:.1%} · {counts[key]}", va="center"
        )
    history.plot(dates, low, color="#2869ac", lw=1.1, label="μ₁(Ĥ): ниже — не покупать")
    history.plot(dates, high, color="#df713e", lw=1.1, label="μ₂(Ĥ): выше — держать")
    history.fill_between(
        dates, low, high, color=COLORS["policy"], alpha=0.12, label="Область стратегии остановки"
    )
    history.axhline(0, color="gray", lw=0.6)
    history.set(xlabel="Дата", ylabel="μ", title="Оценка μ и границы для текущего Ĥ во времени")
    history.legend(frameon=False, fontsize=9, ncol=3)
    for ax in (plane, shares, history):
        _style(ax)
    _time(history)
    return fig


def plot_regime_shares(result):
    years = pd.to_datetime(result.decision_date).dt.year // 10 * 10
    table = pd.crosstab(years, result.regime).reindex(columns=list(COLORS), fill_value=0)
    proportions = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    bottom = np.zeros(len(table))
    for key in COLORS:
        heights = proportions[key].to_numpy()
        ax.bar(
            range(len(table)),
            heights,
            bottom=bottom,
            color=COLORS[key],
            label=LABELS[key],
            width=0.7,
        )
        for i, v in enumerate(heights):
            if v > 0.09:
                ax.text(
                    i,
                    bottom[i] + v / 2,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )
        bottom += heights
    ax.set(
        xticks=range(len(table)),
        xticklabels=[f"{y}-е\nn={n}" for y, n in zip(table.index, table.sum(axis=1))],
        ylabel="Доля блоков",
        ylim=(0, 1),
        title="Как распределение режимов менялось по десятилетиям",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _style(ax)
    return fig


def plot_calibration(calibration):
    fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
    ax.errorbar(
        calibration.pred_mean,
        calibration.real_mean,
        yerr=calibration.real_se,
        fmt="o",
        color="#2869ac",
        capsize=3,
        label="Группы прогноза, ±1 SE",
    )
    limits = np.r_[calibration.pred_mean, calibration.real_mean]
    lo = min(0.0, limits.min())
    hi = limits.max()
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="Прогноз совпадает с фактом")
    ax.set(
        xlabel="Прогноз средней лог-награды",
        ylabel="Фактическая средняя лог-награда",
        title="Калибровка прогноза",
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    _style(ax)
    return fig


def plot_saved_market_results(output, selection, *, show_boundary_intervals=True):
    """Rebuild figures from CSV only, including already saved parameter intervals."""
    output = Path(output)
    result = pd.read_csv(output / "blocks.csv", parse_dates=["decision_date", "end_date"])
    validate_saved_results(result, selection)
    c = selection.candidate
    interval_path = output / "parameter_intervals.csv"
    intervals = pd.read_csv(interval_path) if interval_path.exists() else None
    figures = {
        "cumulative": plot_performance(
            result, f"Зафиксированная модель · α={c.alpha:g} · {c.method.upper()}"
        ),
        "parameter_history": plot_parameters(
            result, windows=selection.windows, intervals=intervals
        ),
        "boundary_diagnostics": plot_boundary_diagnostics(result, selection.boundaries),
        "regime_shares": plot_regime_shares(result),
        "final_boundaries": plot_final_boundaries(
            selection, show_intervals=show_boundary_intervals
        ),
    }
    if (output / "calibration.csv").exists():
        figures["calibration"] = plot_calibration(pd.read_csv(output / "calibration.csv"))
    for name, fig in figures.items():
        fig.savefig(output / f"{name}.png", dpi=160)
    return figures
