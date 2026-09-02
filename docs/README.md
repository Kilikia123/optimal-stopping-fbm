# План магистерской диссертации

**Тема:** нахождение оптимального момента продажи актива с учётом автокорреляционной структуры цены.

> **Идея работы в одном предложении.**
> Исследуется, как автокорреляционная структура fractional Brownian motion влияет на оптимальный момент продажи актива: теоретически определяются области немедленной остановки, нетривиального optimal timing и удержания до конца; эти результаты проверяются численно с помощью signature-based optimal stopping и переносятся на реальные данные S&P 500.

---

## Навигация

| Файл | О чём | Тип работы |
|---|---|---|
| [01-problem-setup.md](01-problem-setup.md) | Модель цены, fBm, постановка задачи, нормировка, связь со статьёй | постановка |
| [02-theory-boundaries.md](02-theory-boundaries.md) | Свойства $V(\mu,H)$, границы $\mu_1(H)$, $\mu_2(H)$, три режима, программа теоретического исследования | **теория (новизна)** |
| [03-method-signatures.md](03-method-signatures.md) | Snell envelope, немарковость, signatures, Longstaff–Schwartz, primal/dual | метод |
| [04-numerics.md](04-numerics.md) | Что уже есть в репозитории, численная оценка границ, сетки, валидация | вычисления |
| [05-real-data.md](05-real-data.md) | S&P 500, оценка $a,\sigma,H$, приведение к модельной форме, walk-forward | эмпирика |
| [06-evaluation.md](06-evaluation.md) | Сравнение трёх подходов, метрики, окно оценивания, издержки, риски | эмпирика |
| [07-contribution-and-roadmap.md](07-contribution-and-roadmap.md) | Что является новым результатом, этапы работы, чек-лист | планирование |

---

## Основные источники

- **Primal and dual optimal stopping with signatures** — Bayer, Pelizzari, Schoenmakers.
  <https://arxiv.org/abs/2312.03444>
- Репозиторий авторов: **Optimal Stopping with Signatures**.
  <https://github.com/lucapelizzari/Optimal_Stopping_with_signatures>
- **Tom Dieker — Simulation of fractional Brownian motion**.
  <https://www.columbia.edu/~ad3217/fbm/thesis.pdf>

Соответствие разделов статьи и файлов плана:

| Раздел статьи | Где используется |
|---|---|
| Section 2 (signatures, Theorem 2.7, Corollary 2.8) | [03](03-method-signatures.md) |
| Section 3.2 / 3.2.1 (Snell envelope, primal, LS) | [03](03-method-signatures.md) |
| Section 3.3 (dual, upper bound) | [03](03-method-signatures.md) |
| Section 4.1 (fBm, эксперимент $\sup_\tau E[B^H_\tau]$) | [01](01-problem-setup.md), [04](04-numerics.md) |

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
                              │
                              ▼
                       РЕАЛЬНЫЕ ДАННЫЕ
        S&P 500 → â, σ̂, Ĥ → μ̂ → режим по границам
                              │
                              ▼
              Signature stopping model на месяц
                     → реальный момент τ*
                              │
                              ▼
              Теория vs Signature vs Buy-and-Hold
```

Более подробная линейная цепочка — в конце [07-contribution-and-roadmap.md](07-contribution-and-roadmap.md).

---

## Соглашения об обозначениях

Единые по всем файлам:

| Символ | Значение |
|---|---|
| $S_t$ | цена актива, $S_t = S_0\exp\{at + \sigma B^H_t\}$ |
| $a$ | **ненормированный** drift исходной модели цены |
| $\sigma$ | масштаб случайной компоненты |
| $B^H_t$ | fractional Brownian motion, параметр Херста $H\in(0,1)$ |
| $\mu = a/\sigma$ | **нормированный** drift — единственный параметр сноса в задаче остановки |
| $X_t = \mu t + B^H_t$ | нормированный наблюдаемый процесс |
| $Z_t = X_t$ | reward конкретной траектории при остановке в момент $t$ |
| $V(\mu,H)$ | значение задачи, $\sup_\tau E[Z_\tau]$ |
| $\mu_1(H),\mu_2(H)$ | границы трёх режимов |
| $T=1$ | горизонт (один месяц), $N=20$ моментов решения |

> **Важно.** В исходной версии плана нормированный drift обозначался то $\mu$, то снова $a$ — это главный источник путаницы, особенно в разделах про реальные данные, где $a$ и $\mu$ имеют разный масштаб и разную размерность. Здесь и далее: $a$ — только ненормированный drift цены, $\mu$ — только нормированный. Границы обозначаются $\mu_1(H),\mu_2(H)$ (в исходном плане — $a_1,a_2$).

Пометкой **➕ Дополнение** отмечены фрагменты, которых не было в исходном плане.
