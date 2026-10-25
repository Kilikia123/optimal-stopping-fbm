# optimal-stopping-fbm

Оптимальная остановка fractional Brownian motion со сносом.
[Постановка и теория](docs/README.md).

1. В [03-grid.ipynb](notebooks/03-grid.ipynb) запустите **новый эксперимент**:
   100 000 train и 100 000 test в каждой точке всей сетки, с доверительными
   интервалами сразу. Задайте новый `RUN_ID`; повторный запуск продолжает расчёт.
2. В [04-grid-analysis.ipynb](notebooks/04-grid-analysis.ipynb) укажите тот же
   `RUN_ID`: графики стоимости и границ μ с интервалами строятся без обучения.
   [Как считаются доверительные интервалы](docs/05-confidence-intervals.md).

Практическая часть ([методика](docs/06-empirical.md)). Нужен завершённый прогон с
21 датой решения (например, `daily21_21_train20k_test20k_v1`) и его границы из
[04-grid-analysis.ipynb](notebooks/04-grid-analysis.ipynb) при выбранном `EPSILON`.

1. `uv sync` — ставит `yfinance`.
2. [06-estimation.ipynb](notebooks/06-estimation.ipynb): задайте `RUN_ID` и `EPSILON`;
   загрузка S&P 500 в `data/`, скользящие оценки a, σ, H, μ с бутстреп-интервалами
   на фоне границ μ₁(H), μ₂(H).
3. [07-backtest.ipynb](notebooks/07-backtest.ipynb) с теми же `RUN_ID` и `EPSILON`:
   синтетическая проверка, бэктест против buy-and-hold, калибровка, выбор окна,
   чувствительность. Результаты — в `result_optimal_stopping/empirical/<RUN_ID>/`.

Прежний расчёт сетки также доступен:

1. В [03-grid.ipynb](notebooks/03-grid.ipynb) выберите `MODE="grid"`, задайте `mu_grid`, `H_grid` и
   параметры Monte Carlo. Запуск сохраняет каждый узел и полный конфиг в
   `result_optimal_stopping/runs/<RUN_ID>/`. Для продолжения укажите прежний `RUN_ID`.
2. В [04-grid-analysis.ipynb](notebooks/04-grid-analysis.ipynb) выберите `ANALYSIS_MODE="legacy"`, запуск
   и `EPSILON`. Графики и практические границы строятся
   по сохранённым данным без повторного обучения.
3. [05-checks.ipynb](notebooks/05-checks.ipynb) содержит численные проверки.

Практические ε-границы определены на всей оси μ. Их разность может быть
отрицательной: это перекрытие областей близости стоимости к двум опорным
стратегиям. Эти оценки пока не подтверждены dual; выбор в нулевой момент
использует существующую поправку по тестовому среднему в исходном скане.
Независимая проверка фиксирует это старое решение до нового теста.

Автоматические тесты в установленном окружении проекта:

```bash
python -m unittest discover -s tests -v
```
