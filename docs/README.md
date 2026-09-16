# План магистерской диссертации

**Тема:** нахождение оптимального момента продажи актива с учётом автокорреляционной структуры цены.

> **Идея работы в одном предложении.**
> Исследуется, как автокорреляционная структура fractional Brownian motion влияет на оптимальный момент продажи актива: исследуются точная граница немедленной остановки и практические границы близости стоимости к двум опорным стратегиям при фиксированном допуске; численная часть использует signature-based optimal stopping.

---

## Навигация

| Файл | О чём | Тип работы |
|---|---|---|
| [01-problem-setup.md](01-problem-setup.md) | Модель цены, fBm, постановка задачи, нормировка | постановка |
| [02-theory-boundaries.md](02-theory-boundaries.md) | Свойства $V(\mu,H)$, точные границы и практические $\mu_i^\varepsilon$, перекрытие областей | **теория (новизна)** |
| [03-method-signatures.md](03-method-signatures.md) | Snell envelope, signatures, Longstaff–Schwartz, primal/dual | метод |
| [04-numerics.md](04-numerics.md) | Численная оценка границ, сетки, валидация | вычисления |
| [05-confidence-intervals.md](05-confidence-intervals.md) | Независимый тест стратегий, интервалы стоимости и сеточных границ μ | статистическая проверка |

Новый рабочий процесс: [06 — новый эксперимент 100k train / 100k test](06-fresh-experiment.md).

---

## Основные источники

- **Primal and dual optimal stopping with signatures** — Bayer, Pelizzari, Schoenmakers.
  <https://arxiv.org/abs/2312.03444>
- Репозиторий авторов: **Optimal Stopping with Signatures**.
  <https://github.com/lucapelizzari/Optimal_Stopping_with_signatures>
- **Ton Dieker — Simulation of fractional Brownian motion**.
  <https://www.columbia.edu/~ad3217/fbm/thesis.pdf>

---

## Общая схема работы

```text
                 S_t = S_0 exp(a t + σ B_t^H)
                              │
                     log-utility + нормировка
                              │
                              ▼
                 V(μ,H) = sup E[μτ + B_τ^H]
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
           ТЕОРИЯ                      ЧИСЛЕННЫЙ МЕТОД
    свойства V, границы             fBm → Signature →
       μ_1^ε(H), μ_2^ε(H)               Continuation Value →
              │                     Bellman → STOP/CONTINUE
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   СИНТЕТИЧЕСКАЯ ПРОВЕРКА
              сетка (μ, H) → τ* и V(μ,H) →
           практические границы при заданном ε
```

---

## Соглашения об обозначениях

| Символ | Значение |
|---|---|
| $S_t$ | цена актива, $S_t = S_0\exp\{at + \sigma B^H_t\}$ |
| $a$ | **ненормированный** drift исходной модели цены |
| $\sigma$ | масштаб случайной компоненты |
| $B^H_t$ | fractional Brownian motion, параметр Херста $H\in(0,1)$ |
| $\mathcal F_t$ | естественная фильтрация $\sigma(B^H_s,\ s\le t)$ |
| $\mu = a/\sigma$ | **нормированный** drift — единственный параметр сноса в задаче остановки |
| $X_t = \mu t + B^H_t$ | нормированный наблюдаемый процесс |
| $Z_t = X_t$ | reward конкретной траектории при остановке в момент $t$ |
| $Y_t$ | Snell envelope, $C_t$ — continuation value |
| $V(\mu,H)$ | значение задачи, $\sup_\tau E[Z_\tau]$ |
| $\mu_1(H),\mu_2(H)$ | Точные границы; $\mu_2=+\infty$ при $H\ne1/2$, $N\ge2$ |
| $\mu_i^\varepsilon, D_\varepsilon$ | Практические границы на всей оси и их разность, допускающая отрицательные значения |
| $T=1$ | горизонт (один месяц), $N=20$ интервалов, 21 момент решения |
| $K$ | уровень усечения signature |
| $M_{\text{train}},M_{\text{test}}$ | число обучающих и тестовых MC-траекторий |

> **Про $a$ и $\mu$.** Это разные величины с разной размерностью: $a$ — только ненормированный drift цены, $\mu$ — только нормированный. Точные границы обозначаются $\mu_1(H),\mu_2(H)$, практические — $\mu_i^\varepsilon(H)$.

## Ноутбуки

- [01 — генерация](../notebooks/01-fbm.ipynb).
- [02 — один узел](../notebooks/02-single-node.ipynb).
- [03 — запуск сетки](../notebooks/03-grid.ipynb).
- [04 — анализ сохранённого запуска](../notebooks/04-grid-analysis.ipynb).
- [05 — проверки](../notebooks/05-checks.ipynb).
