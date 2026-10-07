"""Rolling hedge ratio, spread, z-score, and position signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """Rolling OLS β: y_t = β_t · x_t + ε, estimated on [t-window, t-1].

    The hedge ratio at time t uses data up to t-1 to avoid look-ahead.
    """
    beta = pd.Series(index=y.index, dtype=float)
    for i in range(window, len(y)):
        y_win = y.iloc[i - window : i]
        x_win = x.iloc[i - window : i]
        beta.iloc[i] = np.polyfit(x_win, y_win, 1)[0]
    return beta


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Spread: s_t = y_t - β_t · x_t."""
    return y - beta * x


def compute_zscore(
    spread: pd.Series,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """Z-score using rolling mean and std on [t-window, t-1].

    At time t the z-score uses statistics up to t-1 (shifted window).
    """
    roll_mean = spread.rolling(window).mean().shift(1)
    roll_std = spread.rolling(window).std().shift(1)
    return (spread - roll_mean) / roll_std


def generate_positions(
    zscore: pd.Series,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_LOSS_Z,
) -> pd.Series:
    """State-machine position signal.

    Returns +1 (long spread), -1 (short spread), or 0 (flat).
    Transitions:
        flat  → short spread  when z > +entry_z
        flat  → long  spread  when z < -entry_z
        short → flat           when z <= exit_z  or z > stop_z
        long  → flat           when z >= -exit_z or z < -stop_z
    """
    pos = pd.Series(0.0, index=zscore.index)
    state = 0.0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            pos.iloc[i] = 0.0
            state = 0.0
            continue

        if state == 0.0:
            if z > entry_z:
                state = -1.0    # spread is rich → short spread
            elif z < -entry_z:
                state = 1.0     # spread is cheap → long spread
        elif state == -1.0:
            if z <= exit_z or z > stop_z:
                state = 0.0
        elif state == 1.0:
            if z >= -exit_z or z < -stop_z:
                state = 0.0

        pos.iloc[i] = state

    return pos
