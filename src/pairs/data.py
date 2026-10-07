"""Download, cache, and clean daily adjusted closes from yfinance."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

import config


def download_prices(
    tickers: list[str] = config.TICKERS,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str = config.DATA_DIR,
) -> pd.DataFrame:
    """Download adjusted closes and cache to parquet.

    Returns a DataFrame indexed by date with one column per ticker.
    Missing tickers (delistings, typos) are silently dropped.
    """
    cache_path = Path(cache_dir) / "prices.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    start = start or str(config.DOWNLOAD_START)
    end = end or str(config.DOWNLOAD_END)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives adjusted close in the "Close" column
        threads=True,
    )

    # yf.download returns MultiIndex columns: (field, ticker).
    # With auto_adjust=True, "Close" already contains the adjusted close.
    prices = raw["Close"].copy()

    prices = _clean(prices)

    os.makedirs(cache_dir, exist_ok=True)
    prices.to_parquet(cache_path)

    return prices


def _clean(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers with too many NaNs, forward-fill small gaps."""
    # Drop tickers missing more than 5% of trading days
    threshold = len(prices) * 0.05
    prices = prices.dropna(axis=1, thresh=len(prices) - int(threshold))

    # Forward-fill gaps up to 5 days (holidays, halts), then drop any remaining NaN rows
    prices = prices.ffill(limit=5)
    prices = prices.dropna()

    return prices


def split_in_out(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prices into in-sample and out-of-sample periods."""
    is_mask = (prices.index >= pd.Timestamp(config.IN_SAMPLE_START)) & (
        prices.index <= pd.Timestamp(config.IN_SAMPLE_END)
    )
    oos_mask = (prices.index >= pd.Timestamp(config.OUT_OF_SAMPLE_START)) & (
        prices.index <= pd.Timestamp(config.OUT_OF_SAMPLE_END)
    )
    return prices.loc[is_mask], prices.loc[oos_mask]
