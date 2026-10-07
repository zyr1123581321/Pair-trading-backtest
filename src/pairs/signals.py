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
    """Rolling OLS hedge ratio: y_t = beta_t * x_t + epsilon.

    At each time t (starting from index `window`), fit OLS on the
    PAST `window` days: y[t-window : t] and x[t-window : t].

    CRITICAL: The slice must NOT include day t. Use y.iloc[i-window : i],
    which gives indices [i-window, i-1]. This prevents look-ahead bias.

    Returns a Series of the same length as y, with NaN for the first
    `window` entries (not enough history).

    Hint: np.polyfit(x_window, y_window, 1)[0] gives the slope (beta).
    """
    raise NotImplementedError("TODO: implement")


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    """Spread: s_t = y_t - beta_t * x_t.

    Simple element-wise subtraction. beta comes from rolling_hedge_ratio.
    """
    raise NotImplementedError("TODO: implement")


def compute_zscore(
    spread: pd.Series,
    window: int = config.LOOKBACK_WINDOW,
) -> pd.Series:
    """Z-score of the spread using rolling statistics.

    z_t = (s_t - mu_{t-1}) / sigma_{t-1}

    where mu and sigma are the rolling mean and std over `window` days.

    CRITICAL: The rolling stats must be SHIFTED by 1 day (.shift(1)).
    Without the shift, you'd use today's spread value to compute today's
    z-score normalization, which is a subtle form of look-ahead bias.

    Use: spread.rolling(window).mean().shift(1) for mu
         spread.rolling(window).std().shift(1)  for sigma
    """
    raise NotImplementedError("TODO: implement")


def generate_positions(
    zscore: pd.Series,
    entry_z: float = config.ENTRY_Z,
    exit_z: float = config.EXIT_Z,
    stop_z: float = config.STOP_LOSS_Z,
) -> pd.Series:
    """State-machine position signal.

    Returns +1 (long spread), -1 (short spread), or 0 (flat).

    State transitions:
        flat  -> short spread (-1)  when z > +entry_z   (spread is rich)
        flat  -> long spread  (+1)  when z < -entry_z   (spread is cheap)
        short -> flat         (0)   when z <= exit_z     (mean reverted)
        short -> flat         (0)   when z > stop_z      (stop-loss)
        long  -> flat         (0)   when z >= -exit_z    (mean reverted)
        long  -> flat         (0)   when z < -stop_z     (stop-loss)

    If z is NaN, reset to flat (state = 0).

    Why a state machine: Without hysteresis, you'd flip in and out of
    positions every day near the threshold, paying transaction costs
    each time. The state machine enters at +/-2 and stays until mean
    reversion (z crosses 0) or stop-loss (|z| > 3).

    Implementation: Loop through the z-score series, track `state` as
    a float (0, +1, or -1), and apply the transitions above.
    """
    raise NotImplementedError("TODO: implement")
