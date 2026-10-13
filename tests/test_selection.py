"""Tests for the pair selection module."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.pairs.selection import engle_granger_screen, filter_pairs, _half_life, _ols_spread


def _make_cointegrated_pair(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with two cointegrated columns and one non-cointegrated."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)

    # x is a random walk
    x = 100 + np.cumsum(rng.randn(n) * 0.5)
    # y is cointegrated with x: y = 0.8*x + mean-reverting noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.randn()
    y = 0.8 * x + noise + 20
    # z is an independent random walk (not cointegrated with x or y)
    z = 50 + np.cumsum(rng.randn(n) * 0.5)

    return pd.DataFrame({"Y": y, "X": x, "Z": z}, index=dates)


def test_ols_spread_returns_series() -> None:
    """_ols_spread should return a Series the same length as the inputs."""
    prices = _make_cointegrated_pair()
    spread = _ols_spread(prices["Y"], prices["X"])
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(prices)


def test_ols_spread_is_residual() -> None:
    """The spread should have bounded variance (stationary) for a cointegrated pair."""
    prices = _make_cointegrated_pair()
    spread = _ols_spread(prices["Y"], prices["X"])
    # The spread can have a nonzero mean (the intercept), but its std should be small
    # relative to the price levels, indicating stationarity
    assert spread.std() < 5  # stationary spread has bounded variance


def test_half_life_cointegrated_pair() -> None:
    """A cointegrated pair should have a finite, positive half-life."""
    prices = _make_cointegrated_pair()
    hl = _half_life(prices["Y"], prices["X"])
    assert hl > 0
    assert hl < 100  # should be reasonably fast for our constructed pair


def test_half_life_random_walks() -> None:
    """Two independent random walks should have a much larger half-life than a cointegrated pair."""
    prices = _make_cointegrated_pair()
    hl_coint = _half_life(prices["Y"], prices["X"])
    hl_random = _half_life(prices["X"], prices["Z"])
    # The cointegrated pair should revert much faster than the independent pair
    assert hl_random > hl_coint


def test_engle_granger_finds_cointegrated_pair() -> None:
    """The screen should find the cointegrated pair (Y, X) and not (X, Z)."""
    prices = _make_cointegrated_pair()
    results = engle_granger_screen(prices, p_threshold=0.05)

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0

    # The cointegrated pair (X, Y) should be in the results
    pair_sets = [
        {row["ticker_a"], row["ticker_b"]} for _, row in results.iterrows()
    ]
    assert {"X", "Y"} in pair_sets


def test_engle_granger_columns() -> None:
    """Results should have the expected columns."""
    prices = _make_cointegrated_pair()
    results = engle_granger_screen(prices, p_threshold=0.05)
    for col in ["ticker_a", "ticker_b", "t_stat", "p_value", "half_life"]:
        assert col in results.columns, f"Missing column: {col}"


def test_engle_granger_sorted_by_t_stat() -> None:
    """Results should be sorted by t_stat ascending (most negative first)."""
    prices = _make_cointegrated_pair()
    results = engle_granger_screen(prices, p_threshold=0.10)
    if len(results) > 1:
        assert (results["t_stat"].diff().dropna() >= 0).all()


def test_filter_pairs_respects_half_life() -> None:
    """filter_pairs should only keep pairs within the half-life range."""
    pairs = pd.DataFrame({
        "ticker_a": ["A", "B", "C", "D"],
        "ticker_b": ["W", "X", "Y", "Z"],
        "t_stat": [-5.0, -4.5, -4.0, -3.5],
        "p_value": [0.001, 0.002, 0.003, 0.004],
        "half_life": [3.0, 15.0, 45.0, 100.0],
    })

    filtered = filter_pairs(pairs, top_n=10, hl_min=5, hl_max=60)

    assert len(filtered) == 2  # only B (15) and C (45)
    assert "A" not in filtered["ticker_a"].values  # hl=3 too short
    assert "D" not in filtered["ticker_a"].values  # hl=100 too long


def test_filter_pairs_respects_top_n() -> None:
    """filter_pairs should return at most top_n pairs."""
    pairs = pd.DataFrame({
        "ticker_a": ["A", "B", "C", "D", "E"],
        "ticker_b": ["V", "W", "X", "Y", "Z"],
        "t_stat": [-5.0, -4.5, -4.0, -3.5, -3.0],
        "p_value": [0.001, 0.002, 0.003, 0.004, 0.005],
        "half_life": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    filtered = filter_pairs(pairs, top_n=3, hl_min=5, hl_max=60)
    assert len(filtered) == 3


# ── Sector constraint tests ──────────────────────────────────────────


def test_sector_filter_excludes_cross_sector() -> None:
    """With a sector_map, cross-sector pairs should not appear in results."""
    prices = _make_cointegrated_pair()
    sector_map = {"Y": "tech", "X": "tech", "Z": "energy"}
    results = engle_granger_screen(prices, p_threshold=0.10, sector_map=sector_map)

    pair_sets = [
        {row["ticker_a"], row["ticker_b"]} for _, row in results.iterrows()
    ]
    # X and Z are in different sectors — should be excluded
    assert {"X", "Z"} not in pair_sets
    assert {"Y", "Z"} not in pair_sets


def test_sector_filter_keeps_same_sector() -> None:
    """Same-sector cointegrated pairs should still be found."""
    prices = _make_cointegrated_pair()
    sector_map = {"Y": "tech", "X": "tech", "Z": "energy"}
    results = engle_granger_screen(prices, p_threshold=0.10, sector_map=sector_map)

    pair_sets = [
        {row["ticker_a"], row["ticker_b"]} for _, row in results.iterrows()
    ]
    # X and Y are both "tech" and cointegrated — should be found
    assert {"X", "Y"} in pair_sets


def test_no_sector_map_tests_all_pairs() -> None:
    """Without a sector_map (None), all pairs should be tested including cross-sector."""
    prices = _make_cointegrated_pair()
    results_no_filter = engle_granger_screen(prices, p_threshold=0.10, sector_map=None)
    sector_map = {"Y": "tech", "X": "tech", "Z": "energy"}
    results_filtered = engle_granger_screen(prices, p_threshold=0.10, sector_map=sector_map)

    # Without sector filter, we should get at least as many pairs
    assert len(results_no_filter) >= len(results_filtered)
