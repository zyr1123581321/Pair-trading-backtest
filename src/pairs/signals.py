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
    """Rolling OLS hedge ratio on [t-window, t-1]. NaN for the first `window` entries."""
    hedge_ratio = pd.Series(data=None, index=y.index, dtype=float)
    for i in range(window, len(y)):
        y_window = y.iloc[i-window : i]
        x_window = x.iloc[i-window : i]
        beta = np.polyfit(x_window, y_window, 1)[0]
        hedge_ratio.iloc[i] = beta
    return hedge_ratio
        



def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Spread: s_t = y_t - beta_t * x_t."""
    return y - beta * x


def compute_zscore(
    spread: pd.Series,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """Z-score with shifted rolling statistics: z_t = (s_t - mu_{t-1}) / sigma_{t-1}."""

    mu = spread.rolling(window).mean().shift(1)
    sigma = spread.rolling(window).std().shift(1)

    return (spread - mu) / sigma


def generate_positions(
    zscore: pd.Series,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_LOSS_Z,
) -> pd.Series:
    """State-machine position signal with hysteresis: enter at |z| > entry_z, exit at mean reversion or stop-loss."""

    positions = pd.Series(data=None, index=zscore.index, dtype=float)

    state = 0
    
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z): 
            state = 0            
        elif state == 0:
            if z > entry_z:
                state = -1 
            elif z < -entry_z:
                state = 1
        elif state == -1:
            if z <= exit_z or z > stop_z:
                state = 0
        else:
            if z >= -exit_z or z < -stop_z:
                state = 0
        positions.iloc[i] = state
    return positions
