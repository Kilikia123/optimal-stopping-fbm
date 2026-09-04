# План магистерской диссертации

**Тема:** нахождение оптимального момента продажи актива с учётом автокорреляционной структуры цены.

> **Идея работы в одном предложении.**
> Исследуется, как автокорреляционная структура fractional Brownian motion влияет на оптимальный момент продажи актива: теоретически определяются области немедленной остановки, нетривиального optimal timing и удержания до конца; эти результаты проверяются численно с помощью signature-based optimal stopping.

---

## Навигация

| Файл | О чём | Тип работы |
|---|---|---|
| [01-problem-setup.md](01-problem-setup.md) | Модель цены, fBm, постановка задачи, нормировка | постановка |
| [02-theory-boundaries.md](02-theory-boundaries.md) | Свойства $V(\mu,H)$, границы $\mu_1(H)$, $\mu_2(H)$, три режима | **теория (новизна)** |
| [03-method-signatures.md](03-method-signatures.md) | Snell envelope, signatures, Longstaff–Schwartz, primal/dual | метод |
| [04-numerics.md](04-numerics.md) | Численная оценка границ, сетки, валидация | вычисления |

Номера `05` и `06` зарезервированы под эмпирическую часть (реальные данные S&P 500, сравнение стратегий) — она будет добавлена позже.

---

## Основные источники

- **Primal and dual optimal stopping with signatures** — Bayer, Pelizzari, Schoenmakers.
  <https://arxiv.org/abs/2312.03444>
- Репозиторий авторов: **Optimal Stopping with Signatures**.
  <https://github.com/lucapelizzari/Optimal_Stopping_with_signatures>
- **Tom Dieker — Simulation of fractional Brownian motion**.
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
       μ_1(H), μ_2(H)               Continuation Value →
              │                     Bellman → STOP/CONTINUE
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   СИНТЕТИЧЕСКАЯ ПРОВЕРКА
              сетка (μ, H) → τ* и V(μ,H) →
           численные границы vs теоретические
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
| $\mu_1(H),\mu_2(H)$ | границы трёх режимов, $\Delta(H)=\mu_2-\mu_1$ |
| $T=1$ | горизонт (один месяц), $N=20$ интервалов, 21 момент решения |
| $K$ | уровень усечения signature |
| $M_{\text{train}},M_{\text{test}}$ | число обучающих и тестовых MC-траекторий |

> **Про $a$ и $\mu$.** Это разные величины с разной размерностью: $a$ — только ненормированный drift цены, $\mu$ — только нормированный. Границы обозначаются $\mu_1(H),\mu_2(H)$.
