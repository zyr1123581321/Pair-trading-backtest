"""Look-ahead bias test.

Truncates data at day t and verifies that signals computed from the
truncated series match those from the full series.  This proves the
backtest never peeks at future data.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.pairs.signals import rolling_hedge_ratio, compute_spread, compute_zscore, generate_positions
import config


def _make_synthetic_pair(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Create a synthetic cointegrated pair."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = 100 + np.cumsum(rng.randn(n) * 0.5)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.randn()
    y = 0.7 * x + noise + 50
    return pd.Series(y, index=dates, name="Y"), pd.Series(x, index=dates, name="X")


@pytest.mark.parametrize("t", [150, 200, 250])
def test_signal_no_look_ahead(t: int) -> None:
    """Signal at day t must be identical whether computed from data[:t+1] or data[:300]."""
    y_full, x_full = _make_synthetic_pair(300)

    # Full-sample computation
    beta_full = rolling_hedge_ratio(y_full, x_full)
    spread_full = compute_spread(y_full, x_full, beta_full)
    z_full = compute_zscore(spread_full)
    pos_full = generate_positions(z_full)

    # Truncated computation
    y_trunc, x_trunc = y_full.iloc[: t + 1], x_full.iloc[: t + 1]
    beta_trunc = rolling_hedge_ratio(y_trunc, x_trunc)
    spread_trunc = compute_spread(y_trunc, x_trunc, beta_trunc)
    z_trunc = compute_zscore(spread_trunc)
    pos_trunc = generate_positions(z_trunc)

    # The signal at day t must match
    assert beta_trunc.iloc[t] == pytest.approx(beta_full.iloc[t], abs=1e-10), \
        f"Hedge ratio at t={t} differs between truncated and full data"
    assert z_trunc.iloc[t] == pytest.approx(z_full.iloc[t], abs=1e-10), \
        f"Z-score at t={t} differs between truncated and full data"
    assert pos_trunc.iloc[t] == pos_full.iloc[t], \
        f"Position at t={t} differs between truncated and full data"
