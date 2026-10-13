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
    """PnL for one pair with fill delay and transaction costs."""
    delayed_pos = positions.shift(fill_delay)
    delayed_beta = beta.shift(fill_delay)

    spread_ret = y.pct_change() - delayed_beta * x.pct_change()
    gross_pnl = delayed_pos * spread_ret

    pos_change = delayed_pos.diff().abs()
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
        index=y.index
    )
    result["cumulative_pnl"] = net_pnl.cumsum()
    
    return result 


def aggregate_pairs(
    pair_results: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Equal-weight daily PnL across all pairs."""
    all_pnl = pd.DataFrame({
        f"{a}_{b}": df["net_pnl"] for (a, b), df in pair_results.items()
    })
    portfolio = pd.DataFrame(index=all_pnl.index)
    portfolio["daily_pnl"] = all_pnl.mean(axis=1)
    portfolio["cumulative_pnl"] = portfolio["daily_pnl"].cumsum()

    return portfolio