"""Performance metrics for the backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(daily_pnl: pd.Series, annual_factor: float = 252) -> float:
    """Annualized Sharpe ratio (assuming zero risk-free rate)."""
    std = daily_pnl.std()
    if std < 1e-15:
        return np.inf if daily_pnl.mean() > 0 else 0.0
    return daily_pnl.mean() / std * np.sqrt(annual_factor)


def annualized_return(daily_pnl: pd.Series, annual_factor: float = 252) -> float:
    """Annualized return from daily additive returns."""
    return daily_pnl.mean() * annual_factor


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum drawdown (as a positive number)."""
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    return drawdown.max()


def win_rate(daily_pnl: pd.Series) -> float:
    """Fraction of trading days with positive PnL."""
    active = daily_pnl[daily_pnl != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).mean()


def turnover(positions: pd.Series) -> float:
    """Average daily turnover: mean absolute change in position."""
    return positions.diff().abs().mean()


def summary_table(
    daily_pnl: pd.Series,
    cumulative_pnl: pd.Series,
    positions: pd.Series,
) -> pd.Series:
    """Compute all metrics and return as a labeled Series."""
    return pd.Series(
        {
            "Annualized Return": annualized_return(daily_pnl),
            "Sharpe Ratio": sharpe_ratio(daily_pnl),
            "Max Drawdown": max_drawdown(cumulative_pnl),
            "Win Rate": win_rate(daily_pnl),
            "Avg Daily Turnover": turnover(positions),
        }
    )
