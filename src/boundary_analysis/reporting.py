"""Export saved relative-boundary estimates, diagnostics and sensitivity plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from boundary_analysis.monotone import fit_monotone_boundaries
from boundary_analysis.relative import relative_boundaries
from experiments.statistics import load_experiment_statistics
from infrastructure.identity import digest
from visualization.relative import plot_relative_boundaries

ACTIONS = {
    "incomplete_paired_test": "Завершить парный пересчёт сохранённых стратегий в необязательном разделе парной статистики notebook 03",
    "missing_zero": "Добавить узел μ=0",
    "incomplete_test": "Завершить сетку и тестовую выборку",
    "insufficient_precision": "Увеличить независимую тестовую выборку; проверить обучение",
    "multiple_transitions": "Проверить обучение с другими seed и точность оценок",
    "expand_window": "Проверить концы сетки: расширить диапазон или увеличить точность",
    "incompatible_shape": "Проверить обучение, точность и гипотезу монотонности",
    "solver_error": "Проверить численное решение задачи ограниченной аппроксимации",
}


def analyze_relative_run(
    run_dir,
    *,
    alpha=0.2,
    confidence=0.95,
    sensitivity=(0.1, 0.2, 0.3, 0.5),
    include_unconfirmed=True,
    gain_mode="legacy",
):
    """Load once; never simulate or train. Return nodes, boundaries, diagnostics, output."""
    run_dir = Path(run_dir)
    manifest, nodes = load_experiment_statistics(
        run_dir, confidence=confidence, gain_mode=gain_mode
    )
    T = manifest["config"]["T"]
    alphas = sorted(set([alpha, *sensitivity]))
    tables = {
        a: fit_monotone_boundaries(
            relative_boundaries(nodes, T=T, alpha=a, confidence=confidence, gain_mode=gain_mode),
            include_unconfirmed=include_unconfirmed,
        )
        for a in alphas
    }
    bounds = tables[alpha]
    output = (
        run_dir
        / "analysis"
        / ("relative_paired" if gain_mode == "paired" else "relative")
        / f"alpha_{alpha:g}"
        / f"confidence_{confidence:g}"
    )
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for _, row in bounds.iterrows():
        for side in (1, 2):
            status = row[f"mu{side}_status"]
            fit = row[f"mu{side}_fit_status"]
            action = ACTIONS.get(fit, ACTIONS.get(status, ""))
            if (
                not action
                and status == "localized_grid"
                and row[f"mu{side}_low"] < row[f"mu{side}_high"]
            ):
                action = "При необходимости уточнить сетку около перехода"
            diagnostics.append(
                dict(H=row.H, side=side, status=status, fit_status=fit, recommendation=action)
            )
    diagnostics = pd.DataFrame(diagnostics)
    bounds.to_csv(output / "boundaries.csv", index=False)
    diagnostics.to_csv(output / "diagnostics.csv", index=False)
    pd.concat(tables.values(), ignore_index=True).to_csv(output / "sensitivity.csv", index=False)
    metadata = dict(
        schema_version=3,
        gain_mode=gain_mode,
        paired_identity=nodes.attrs.get("paired_identity"),
        include_unconfirmed=include_unconfirmed,
        run_dir=str(run_dir.resolve()),
        experiment_identity=digest(manifest),
        alpha=alpha,
        confidence=confidence,
        sensitivity=alphas,
        T=T,
        family_size=manifest["family_size"],
        solver="CLARABEL",
        interpretation="Policy-mean diagnostic, not certified optimal-value boundaries",
        bands="Bonferroni Student-t, approximate for non-Gaussian rewards",
        coverage="Separate simultaneous coverage for each boundary; not joint coverage of both",
        shape="Separate monotone branches; no symmetry; analytic anchor at H=0.5",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    fig = plot_relative_boundaries(bounds, alpha=alpha)
    fig.savefig(output / "boundaries.png", dpi=160)
    # A separate diagnostic view: raw and fitted estimates must not look like
    # two values of the same boundary in the main plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for side, ax, color in zip((1, 2), axes, ("#2a78d6", "#eb6834")):
        ax.plot(bounds.H, bounds[f"mu{side}_raw"], "x", color="gray", label="Исходная оценка")
        ax.plot(
            bounds.H,
            bounds[f"mu{side}_fit"],
            ".-",
            color=color,
            label="После ограничения монотонности",
        )
        ax.fill_between(
            bounds.H, bounds[f"mu{side}_low"], bounds[f"mu{side}_high"], color=color, alpha=0.15
        )
        ax.set(xlabel="H", ylabel=f"μ{side}", title="Диагностика корректировки и интервалов")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "boundary_comparison.png", dpi=160)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bounds.H, bounds.epsilon, ".-")
    ax.fill_between(bounds.H, bounds.epsilon_low, bounds.epsilon_high, alpha=0.2)
    ax.set(xlabel="H", ylabel="ε(H)", title="Допуск и неопределённость его оценки")
    fig.tight_layout()
    fig.savefig(output / "epsilon.png", dpi=160)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a, table in tables.items():
        for side, ax in enumerate(axes, 1):
            ax.plot(table.H, table[f"mu{side}_fit"], ".-", label=f"α={a:g}")
            ax.set(xlabel="H", ylabel=f"μ{side}", title="Чувствительность монотонной оценки")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "sensitivity.png", dpi=160)
    return nodes, bounds, diagnostics, output
