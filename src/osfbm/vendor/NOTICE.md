# Сторонний код

Файлы в этом каталоге скопированы **без изменений** из репозитория авторов статьи.

- Источник: <https://github.com/lucapelizzari/Optimal_Stopping_with_signatures>
- Коммит: `c1b73eebb5a2457b28e6652868748d7df1a7aaf6` (ветка `main`)
- Дата копирования: 2026-09-04
- Автор: Luca Pelizzari

Статья: Bayer, Pelizzari, Schoenmakers, *Primal and dual optimal stopping with
signatures*, <https://arxiv.org/abs/2312.03444>

| Файл | Исходный путь | SHA-256 |
|---|---|---|
| `FBM_package.py` | `FBM_package.py` | `cd18d6c8…6b0e40f1` |
| `Signature_computer.py` | `Signature_computer.py` | `54b479aa…cd4f1a4f` |
| `Linear_signature_optimal_stopping.py` | `Linear signature optimal stopping/Linear_signature_optimal_stopping.py` | `d38e0c7b…ffc50b2b` |

## Правила работы с этим каталогом

**Файлы не редактируются.** Пока они побайтово совпадают с оригиналом, сверка
наших результатов с числами Section 4.1 статьи остаётся честной. Все поправки,
нужные нашей задаче, живут в `src/osfbm/adapter.py`.

Проверка целостности:

```bash
shasum -a 256 src/osfbm/vendor/*.py
```

## Лицензия

В исходном репозитории **нет файла LICENSE**. Код скопирован с полной
атрибуцией для академического использования. Перед публикацией репозитория
диссертации нужно либо получить у авторов явное разрешение, либо заменить эти
файлы собственной реализацией — интерфейс `adapter.py` это допускает без
переделки остального кода.
