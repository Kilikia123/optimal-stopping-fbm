"""Shared plot colours, regime labels and axis formatting."""

import matplotlib.dates as mdates

COLORS = {"cash": "#8c939e", "policy": "#d89a24", "hold": "#219176"}


LABELS = {"cash": "Не покупать", "policy": "Стратегия остановки", "hold": "Держать"}


STRATEGIES = {
    "model": ("Модель", "#2869ac"),
    "buy_hold": ("Buy-and-hold", "#df713e"),
    "regimes_only": ("Только режимы", "#219176"),
    "sign_drift": ("Знак сноса", "#9065a8"),
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def _time(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
