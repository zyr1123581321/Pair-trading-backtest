"""Performance metrics for the backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(daily_pnl: pd.Series, annual_factor: float = 252) -> float:
    """Annualized Sharpe ratio (assuming zero risk-free rate).

    Formula: (mean daily return / std daily return) * sqrt(252)

    Edge case: if std is effectively zero (< 1e-15), return np.inf
    if mean > 0, else 0.0.
    """
    raise NotImplementedError("TODO: implement")


def annualized_return(daily_pnl: pd.Series, annual_factor: float = 252) -> float:
    """Annualized return from daily additive returns.

    Formula: mean_daily_return * 252
    """
    raise NotImplementedError("TODO: implement")


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum drawdown as a positive number.

    Steps:
      1. Compute the running maximum: cumulative_pnl.cummax()
      2. Drawdown at each point = running_max - cumulative_pnl
      3. Return the maximum drawdown value.
    """
    raise NotImplementedError("TODO: implement")


def win_rate(daily_pnl: pd.Series) -> float:
    """Fraction of active trading days with positive PnL.

    Only count days where PnL != 0 (skip days with no position).
    Return 0.0 if there are no active days.
    """
    raise NotImplementedError("TODO: implement")


def turnover(positions: pd.Series) -> float:
    """Average daily turnover: mean absolute change in position."""
    raise NotImplementedError("TODO: implement")


def summary_table(
    daily_pnl: pd.Series,
    cumulative_pnl: pd.Series,
    positions: pd.Series,
) -> pd.Series:
    """Compute all metrics and return as a labeled Series.

    Keys: 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown',
          'Win Rate', 'Avg Daily Turnover'
    """
    raise NotImplementedError("TODO: implement")
