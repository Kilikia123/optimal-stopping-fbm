# Данные

- `gspc_daily.csv`: дневные закрытия `^GSPC` c Yahoo Finance.
  Загружено 2026-10-24, строк: 24821,
  период 1927-12-30 — 2026-10-23.

Файл читается функцией `osfbm.data.load_prices`; повторная загрузка —
`osfbm.data.fetch_prices(refresh=True)`.
