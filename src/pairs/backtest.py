"""Vectorized backtest: positions, transaction costs, and PnL."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def backtest_pair(
    y: pd.Series,
    x: pd.Series,
    positions: pd.Series,
    beta: pd.Series,
    fill_delay: int = config.FILL_DELAY,
    cost_frac: float = config.COST_FRAC,
) -> pd.DataFrame:
    """Backtest a single pair.

    Steps:
      1. Delay positions and beta by `fill_delay` bars using .shift().
         This models execution delay: signal at close t -> fill at close t+1.
      2. Compute spread return:
         spread_ret = y.pct_change() - delayed_beta * x.pct_change()
      3. Gross PnL = delayed_position * spread_return
      4. Transaction costs: charged on every position change.
         pos_change = delayed_pos.diff().abs()
         costs = pos_change * cost_frac * 2  (2 legs per pair trade)
      5. Net PnL = gross - costs
      6. Cumulative PnL = net_pnl.cumsum()

    Returns a DataFrame with columns:
        position, spread_return, gross_pnl, costs, net_pnl, cumulative_pnl
    """
    raise NotImplementedError("TODO: implement")


def aggregate_pairs(
    pair_results: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Combine PnL across all pairs (equal weight, additive returns).

    Steps:
      1. Extract the 'net_pnl' column from each pair's DataFrame.
      2. Take the mean across all pairs for each day (equal-weight portfolio).
      3. Compute cumulative PnL.

    Returns a DataFrame with columns: daily_pnl, cumulative_pnl.
    """
    raise NotImplementedError("TODO: implement")
