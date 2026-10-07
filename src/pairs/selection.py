"""Pair selection via Engle-Granger cointegration tests."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import config


def engle_granger_screen(
    prices: pd.DataFrame,
    p_threshold: float = config.COINT_PVALUE,
) -> pd.DataFrame:
    """Run pairwise Engle-Granger tests on all ticker combinations.

    Returns a DataFrame with columns:
        ticker_a, ticker_b, t_stat, p_value, half_life
    sorted by t_stat (most negative = strongest cointegration).
    """
    tickers = prices.columns.tolist()
    results = []

    for a, b in combinations(tickers, 2):
        t_stat, p_value, _ = coint(prices[a], prices[b])
        if p_value < p_threshold:
            hl = _half_life(prices[a], prices[b])
            results.append(
                {
                    "ticker_a": a,
                    "ticker_b": b,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "half_life": hl,
                }
            )

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("t_stat").reset_index(drop=True)
    return df


def _half_life(y: pd.Series, x: pd.Series) -> float:
    """Mean-reversion half-life of the OLS spread via the Ornstein-Uhlenbeck AR(1) fit."""
    spread = _ols_spread(y, x)
    lag = spread.shift(1)
    delta = spread.diff()

    # Drop NaNs and align
    valid = lag.notna() & delta.notna()
    lag, delta = lag[valid], delta[valid]

    if len(lag) < 10:
        return np.inf

    phi = np.polyfit(lag, delta, 1)[0]
    if phi >= 0:
        return np.inf
    return -np.log(2) / phi


def _ols_spread(y: pd.Series, x: pd.Series) -> pd.Series:
    """OLS hedge ratio β and spread: s = y - β·x."""
    beta = np.polyfit(x, y, 1)[0]
    return y - beta * x


def filter_pairs(
    pairs: pd.DataFrame,
    top_n: int = config.TOP_N_PAIRS,
    hl_min: float = config.HALFLIFE_MIN,
    hl_max: float = config.HALFLIFE_MAX,
) -> pd.DataFrame:
    """Keep pairs whose half-life is in [hl_min, hl_max], take top_n by t_stat."""
    filtered = pairs[
        (pairs["half_life"] >= hl_min) & (pairs["half_life"] <= hl_max)
    ].copy()
    return filtered.head(top_n).reset_index(drop=True)
