"""Unit tests for the signal state machine."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.pairs.signals import generate_positions


def _zscore_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_entry_short_spread() -> None:
    """Z > entry_z should trigger short spread (-1)."""
    z = _zscore_series([0, 0, 2.5, 2.5, 0.5, -0.1])
    pos = generate_positions(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    assert pos.iloc[2] == -1.0
    assert pos.iloc[3] == -1.0
    assert pos.iloc[5] == 0.0   # exited at z <= 0


def test_entry_long_spread() -> None:
    """Z < -entry_z should trigger long spread (+1)."""
    z = _zscore_series([0, 0, -2.5, -2.5, -0.5, 0.1])
    pos = generate_positions(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    assert pos.iloc[2] == 1.0
    assert pos.iloc[3] == 1.0
    assert pos.iloc[5] == 0.0


def test_stop_loss() -> None:
    """Z beyond stop_z should exit the position."""
    z = _zscore_series([0, 0, 2.5, 3.5])
    pos = generate_positions(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    assert pos.iloc[2] == -1.0
    assert pos.iloc[3] == 0.0   # stopped out


def test_nan_resets_to_flat() -> None:
    """NaN z-scores should result in flat position."""
    z = _zscore_series([0, 2.5, float("nan"), 0])
    pos = generate_positions(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    assert pos.iloc[2] == 0.0
    assert pos.iloc[3] == 0.0
