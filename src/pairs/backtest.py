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

    The position signal is lagged by fill_delay bars to model execution delay.
    Returns a DataFrame with columns:
        spread_return, gross_pnl, costs, net_pnl, cumulative_pnl
    """
    # Delayed positions: signal at t, filled at t + fill_delay
    delayed_pos = positions.shift(fill_delay)
    delayed_beta = beta.shift(fill_delay)

    # Spread return: r_t = (y_t / y_{t-1} - 1) - β_{t-1} * (x_t / x_{t-1} - 1)
    y_ret = y.pct_change()
    x_ret = x.pct_change()
    spread_ret = y_ret - delayed_beta * x_ret

    gross_pnl = delayed_pos * spread_ret

    # Costs: charged on position changes (entry, exit, flip)
    pos_change = delayed_pos.diff().abs()
    # Each unit of position change costs cost_frac per leg, 2 legs
    costs = pos_change * cost_frac * 2

    net_pnl = gross_pnl - costs

    result = pd.DataFrame(
        {
            "position": delayed_pos,
            "spread_return": spread_ret,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
        },
        index=y.index,
    )
    result["cumulative_pnl"] = result["net_pnl"].cumsum()

    return result


def aggregate_pairs(
    pair_results: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Combine PnL across all pairs (equal weight, additive returns).

    Returns a DataFrame with portfolio-level daily and cumulative PnL.
    """
    if not pair_results:
        return pd.DataFrame()

    all_pnl = pd.DataFrame(
        {f"{a}_{b}": df["net_pnl"] for (a, b), df in pair_results.items()}
    )

    portfolio = pd.DataFrame(index=all_pnl.index)
    portfolio["daily_pnl"] = all_pnl.mean(axis=1)  # equal-weight average
    portfolio["cumulative_pnl"] = portfolio["daily_pnl"].cumsum()

    return portfolio
