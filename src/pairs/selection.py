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

    For every pair (a, b) in the universe:
      1. Call statsmodels.tsa.stattools.coint(prices[a], prices[b])
         which returns (t_stat, p_value, crit_values).
      2. If p_value < p_threshold, compute the half-life and keep the pair.

    Returns a DataFrame with columns:
        ticker_a, ticker_b, t_stat, p_value, half_life
    sorted by t_stat ascending (most negative = strongest cointegration).

    Hints:
      - Use itertools.combinations(tickers, 2) for all pairs.
      - Call _half_life(prices[a], prices[b]) for the half-life.
    """
    raise NotImplementedError("TODO: implement")


def _half_life(y: pd.Series, x: pd.Series) -> float:
    """Mean-reversion half-life of the OLS spread.

    Steps:
      1. Compute the OLS spread: s = y - beta * x  (use _ols_spread).
      2. Compute lag = s shifted by 1 period, delta = s.diff().
      3. Align them (drop NaNs from both so they have equal length).
      4. Fit AR(1): delta_t = phi * lag_t + epsilon
         via np.polyfit(lag, delta, 1) — the slope is phi.
      5. If phi >= 0, the spread doesn't mean-revert — return np.inf.
      6. Otherwise return -ln(2) / phi.

    The math: this is the discrete Ornstein-Uhlenbeck process.
    half-life = -ln(2) / phi tells you how many days until a deviation
    from the mean is expected to shrink by half.
    """
    raise NotImplementedError("TODO: implement")


def _ols_spread(y: pd.Series, x: pd.Series) -> pd.Series:
    """OLS hedge ratio and spread.

    Fit y = beta * x + intercept via np.polyfit(x, y, 1).
    Return the spread: s = y - beta * x.
    """
    raise NotImplementedError("TODO: implement")


def filter_pairs(
    pairs: pd.DataFrame,
    top_n: int = config.TOP_N_PAIRS,
    hl_min: float = config.HALFLIFE_MIN,
    hl_max: float = config.HALFLIFE_MAX,
) -> pd.DataFrame:
    """Keep pairs whose half-life is in [hl_min, hl_max], take top_n by t_stat.

    The input DataFrame is already sorted by t_stat (most negative first).
    Filter rows where half_life is within the range, then take the first top_n.
    """
    raise NotImplementedError("TODO: implement")
