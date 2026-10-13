"""Tests for the backtest module."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.pairs.backtest import backtest_pair, aggregate_pairs


def _make_prices(n: int = 10) -> tuple[pd.Series, pd.Series]:
    """Two simple price series."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    y = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109.0], index=dates, name="Y")
    x = pd.Series([50, 51, 50.5, 51.5, 52, 51, 52, 53, 52.5, 53.5], index=dates, name="X")
    return y, x


def test_backtest_pair_returns_dataframe() -> None:
    """backtest_pair should return a DataFrame with the expected columns."""
    y, x = _make_prices()
    positions = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 0, 0.0], index=y.index)
    beta = pd.Series([1.0] * 10, index=y.index)

    result = backtest_pair(y, x, positions, beta, fill_delay=1, cost_frac=0.0005)

    assert isinstance(result, pd.DataFrame)
    for col in ["position", "spread_return", "gross_pnl", "costs", "net_pnl", "cumulative_pnl"]:
        assert col in result.columns, f"Missing column: {col}"
    assert len(result) == len(y)


def test_fill_delay_shifts_position() -> None:
    """The position used in PnL should be delayed by fill_delay days."""
    y, x = _make_prices()
    # Position enters on day 2 (index 2)
    positions = pd.Series([0, 0, 1, 1, 0, 0, 0, 0, 0, 0.0], index=y.index)
    beta = pd.Series([1.0] * 10, index=y.index)

    result = backtest_pair(y, x, positions, beta, fill_delay=1, cost_frac=0.0)

    # With fill_delay=1, position at day 2 should appear in result at day 3
    assert result["position"].iloc[2] == 0.0  # not yet filled
    assert result["position"].iloc[3] == 1.0  # filled one day later


def test_flat_position_zero_pnl() -> None:
    """A flat position (all zeros) should produce zero PnL everywhere."""
    y, x = _make_prices()
    positions = pd.Series(0.0, index=y.index)
    beta = pd.Series([1.0] * 10, index=y.index)

    result = backtest_pair(y, x, positions, beta, fill_delay=1, cost_frac=0.0005)

    assert (result["gross_pnl"].dropna() == 0).all()
    assert (result["costs"].dropna() == 0).all()


def test_costs_only_on_position_change() -> None:
    """Costs should be zero when position doesn't change."""
    y, x = _make_prices()
    # Position is constant at 1 after day 3 (after fill delay)
    positions = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 1, 1.0], index=y.index)
    beta = pd.Series([1.0] * 10, index=y.index)

    result = backtest_pair(y, x, positions, beta, fill_delay=1, cost_frac=0.0005)

    # After the initial entry, costs should be zero for holding
    # Days 4+ in delayed_pos are all 1.0, so diff is 0 -> no cost
    costs_after_entry = result["costs"].iloc[5:]  # well after entry
    assert (costs_after_entry == 0).all()


def test_costs_charged_on_entry_and_exit() -> None:
    """Costs should be nonzero when entering and exiting a position."""
    y, x = _make_prices()
    # Enter on day 2, exit on day 5
    positions = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0.0], index=y.index)
    beta = pd.Series([1.0] * 10, index=y.index)
    cost_frac = 0.01  # large costs so they're easy to check

    result = backtest_pair(y, x, positions, beta, fill_delay=1, cost_frac=cost_frac)

    # With fill_delay=1, entry shows at day 3, exit at day 6
    # Entry cost: |1-0| * 0.01 * 2 = 0.02
    assert result["costs"].iloc[3] == pytest.approx(0.02)
    # Exit cost: |0-1| * 0.01 * 2 = 0.02
    assert result["costs"].iloc[6] == pytest.approx(0.02)


def test_cumulative_pnl_is_cumsum() -> None:
    """cumulative_pnl should be the running sum of net_pnl."""
    y, x = _make_prices()
    positions = pd.Series([0, 0, 1, 1, -1, -1, 0, 0, 1, 0.0], index=y.index)
    beta = pd.Series([1.0] * 10, index=y.index)

    result = backtest_pair(y, x, positions, beta, fill_delay=1, cost_frac=0.0005)

    expected_cum = result["net_pnl"].cumsum()
    pd.testing.assert_series_equal(result["cumulative_pnl"], expected_cum, check_names=False)


def test_aggregate_pairs_equal_weight() -> None:
    """aggregate_pairs should average net_pnl across pairs."""
    dates = pd.bdate_range("2020-01-01", periods=5)

    pair1 = pd.DataFrame({"net_pnl": [0.01, -0.02, 0.03, 0.00, 0.01]}, index=dates)
    pair2 = pd.DataFrame({"net_pnl": [0.03, 0.02, -0.01, 0.00, -0.01]}, index=dates)

    result = aggregate_pairs({("A", "B"): pair1, ("C", "D"): pair2})

    assert "daily_pnl" in result.columns
    assert "cumulative_pnl" in result.columns
    # Mean of pair1 and pair2 net_pnl
    expected_daily = pd.Series([0.02, 0.00, 0.01, 0.00, 0.00], index=dates)
    pd.testing.assert_series_equal(result["daily_pnl"], expected_daily, check_names=False)


def test_aggregate_pairs_empty() -> None:
    """aggregate_pairs with no pairs should return empty DataFrame."""
    result = aggregate_pairs({})
    assert isinstance(result, pd.DataFrame)
    assert result.empty
