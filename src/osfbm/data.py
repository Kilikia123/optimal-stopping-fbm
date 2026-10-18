"""Daily price data for the empirical part: download, cache, monthly blocks.

Time convention. One model month is ``D`` trading days (D=21 by default). A
block consists of D+1 consecutive closes P_0, ..., P_D: the decision point t_0
is the close of the day before the block, selling is allowed at the closes of
days 1..D. Blocks are consecutive and do not overlap, so they do not coincide
with calendar months.
"""
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_TICKER = "^GSPC"
DEFAULT_START = "1927-01-01"


def _slug(ticker):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_")


def cache_path(ticker=DEFAULT_TICKER, data_dir=DATA_DIR):
    return Path(data_dir) / f"{_slug(ticker)}_daily.csv"


def download_prices(ticker=DEFAULT_TICKER, start=DEFAULT_START, end=None):
    """Daily closes from Yahoo Finance as a float Series indexed by date.

    For an index such as ^GSPC the adjusted close equals the close. The result
    is cleaned: non-positive and missing prices are dropped, dates are unique
    and sorted.
    """
    import yfinance as yf  # imported lazily: only the download needs the network

    frame = yf.download(ticker, start=start, end=end, auto_adjust=False,
                        progress=False, actions=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {ticker!r}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    column = "Adj Close" if "Adj Close" in frame.columns else "Close"
    series = frame[column].astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[np.isfinite(series) & (series > 0)]
    series.name = "close"
    series.index.name = "date"
    return series


def save_prices(series, path, ticker=DEFAULT_TICKER):
    """Write the cache CSV and a README noting the download date."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(path, index_label="date", float_format="%.6f")
    readme = path.parent / "README.md"
    lines = [
        "# Данные",
        "",
        f"- `{path.name}`: дневные закрытия `{ticker}` c Yahoo Finance.",
        f"  Загружено {dt.date.today().isoformat()}, строк: {len(series)},",
        f"  период {series.index[0].date()} — {series.index[-1].date()}.",
        "",
        "Файл читается функцией `osfbm.data.load_prices`; повторная загрузка —",
        "`osfbm.data.fetch_prices(refresh=True)`.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_prices(path):
    """Read a cached CSV written by ``save_prices``."""
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    series = frame["close"].astype(float).sort_index()
    if series.index.has_duplicates or not np.all(np.isfinite(series)) or (series <= 0).any():
        raise ValueError(f"Corrupted price cache: {path}")
    series.name = "close"
    return series


def fetch_prices(ticker=DEFAULT_TICKER, *, data_dir=DATA_DIR, refresh=False,
                 start=DEFAULT_START, end=None):
    """Cached prices; download only when the cache is missing or refresh=True."""
    path = cache_path(ticker, data_dir)
    if path.exists() and not refresh:
        return load_prices(path)
    series = download_prices(ticker, start=start, end=end)
    save_prices(series, path, ticker)
    return series


def month_blocks(n_obs, D=21, warmup=0):
    """Consecutive non-overlapping blocks as an int array of (decision, end).

    ``decision`` is the index of P_0, ``end = decision + D`` the index of P_D.
    The first decision index is ``warmup``: an estimation window of W daily
    returns needs W+1 prices, so pass ``warmup=W``.
    """
    if D < 1 or warmup < 0:
        raise ValueError("D must be >= 1 and warmup >= 0")
    starts = np.arange(warmup, n_obs - D, D)
    return np.column_stack([starts, starts + D]).astype(int)


def block_frame(dates, blocks):
    """Decision and end dates for each block, for reporting."""
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame({"decision_date": dates[blocks[:, 0]],
                         "end_date": dates[blocks[:, 1]],
                         "decision_idx": blocks[:, 0], "end_idx": blocks[:, 1]})
