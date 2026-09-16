"""Plot saved audit results without generating paths or fitting policies."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from osfbm.confidence import load_audit


STATUS_LABELS = {
    "localized_grid": "Локализована на сетке",
    "expand_window": "Нужно расширить диапазон μ",
    "multiple_transitions": "Повторные переходы",
    "incomplete_test": "Тест не завершён",
    "source_not_localized": "Нет исходной границы",
    "source_incomplete": "Исходная сетка неполна",
}


def plot_audit(audit_dir, H=0.3, output_dir=None, epsilon=None):
    fresh = (Path(audit_dir) / "experiment.json").exists()
    if fresh:
        from osfbm.experiment import load_experiment
        manifest, nodes, boundaries = load_experiment(audit_dir, epsilon=epsilon)
    else:
        manifest, nodes, boundaries = load_audit(audit_dir)
    threshold = manifest["settings"]["epsilon"] if epsilon is None else epsilon
    output = Path(output_dir) if output_dir else Path(audit_dir) / "analysis"
    if fresh and output_dir is None:
        output = output / f"epsilon_{threshold:g}"
    output.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(output / "values.csv", index=False)
    boundaries.to_csv(output / "boundaries.csv", index=False)
    epsilon = threshold
    level = 100 * manifest["settings"]["confidence"]
    view = nodes[nodes.H.eq(H) & nodes.complete].sort_values("mu")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, prefix, value, title in zip(
        axes, ("", "excess_"), ("V", "excess"),
        ("Стоимость фиксированной стратегии", "Превышение над удержанием"),
    ):
        # Separate original windows, rather than drawing a band across untested mu.
        mu = view.mu.to_numpy()
        splits = np.flatnonzero(np.diff(mu) > 1.5 * np.min(np.diff(mu))) + 1 if len(mu) > 1 else []
        for i, part in enumerate(np.split(np.arange(len(view)), splits)):
            if not len(part):
                continue
            group = view.iloc[part]
            x = group.mu.to_numpy()
            ax.fill_between(x, group[prefix + "band_low"].to_numpy(),
                            group[prefix + "band_high"].to_numpy(), color="C0", alpha=0.12,
                            label=f"Общая полоса {level:g}%" if i == 0 else None)
            ax.fill_between(x, group[prefix + "ci_low"].to_numpy(),
                            group[prefix + "ci_high"].to_numpy(), color="C0", alpha=0.3,
                            label=f"Точечный ДИ {level:g}%" if i == 0 else None)
            ax.plot(x, group[value].to_numpy(), ".-", color="C0", label="Новый тест" if i == 0 else None)
        ax.axhline(epsilon, color="black", linestyle="--", label=f"ε={epsilon:g}")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set(xlabel="μ", ylabel="Награда", title=title)
        if view.empty:
            ax.text(0.5, 0.5, "Нет завершённых тестовых узлов", ha="center", transform=ax.transAxes)
        ax.legend(fontsize=8)
    for _, row in boundaries[boundaries.H.eq(H)].iterrows():
        ax = axes[int(row.side) - 1]
        if row.status == "localized_grid":
            ax.axvspan(row.mu_low, row.mu_high, color="C1", alpha=0.25)
            ax.axvline(row.estimate, color="C1", linestyle=":")
        ax.text(0.02, 0.02, STATUS_LABELS[row.status], transform=ax.transAxes, fontsize=8)
    domain = "полная сетка μ" if fresh else "локальные окна μ"
    fig.suptitle(f"H={H:g}; обученные стратегии, {domain}")
    fig.tight_layout()
    fig.savefig(output / f"values_H_{H:g}.png", dpi=160, bbox_inches="tight")

    grid_fig, grid_axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for side, ax in enumerate(grid_axes, 1):
        group = boundaries[boundaries.side.eq(side)].sort_values("H")
        old = group.old_mu.to_numpy(dtype=float)
        if np.isfinite(old).any():
            ax.plot(group.H, old, "--", color="gray", label="Исходная оценка")
        valid = group.status.eq("localized_grid")
        y = np.where(valid, group.estimate.to_numpy(dtype=float), np.nan)
        ax.plot(group.H, y, ".-", color=f"C{side}", label="Независимый тест")
        good = group[valid]
        if len(good):
            ax.errorbar(good.H, good.estimate,
                        yerr=np.array([good.estimate - good.mu_low, good.mu_high - good.estimate]),
                        fmt="none", capsize=3, color=f"C{side}", label="Интервал сеточной μ")
        missing = group[~valid]
        if len(missing):
            ax.scatter(missing.H, np.full(len(missing), 0.025), marker="x", color="red",
                       transform=ax.get_xaxis_transform(), label="Не локализована / тест не завершён")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set(xlabel="H", ylabel="μ", title=f"Практическая граница μ{side}, ε={epsilon:g}")
        ax.legend(fontsize=8)
    grid_fig.suptitle(f"Общее приближённое покрытие {level:g}%; {domain}")
    grid_fig.tight_layout()
    grid_fig.savefig(output / "boundaries_ci.png", dpi=180, bbox_inches="tight")
    return nodes, boundaries, (fig, grid_fig)
