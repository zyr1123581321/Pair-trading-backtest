"""Sanity checks for performance metrics."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.pairs.metrics import sharpe_ratio, max_drawdown, win_rate, annualized_return


def test_sharpe_positive() -> None:
    pnl = pd.Series([0.01] * 252)
    assert sharpe_ratio(pnl) == np.inf  # constant positive returns → infinite Sharpe


def test_sharpe_zero_vol() -> None:
    pnl = pd.Series([0.0] * 100)
    assert sharpe_ratio(pnl) == 0.0


def test_max_drawdown_no_loss() -> None:
    cum = pd.Series([1, 2, 3, 4, 5])
    assert max_drawdown(cum) == 0.0


def test_max_drawdown_simple() -> None:
    cum = pd.Series([0, 1, 2, 0.5, 1])
    assert max_drawdown(cum) == pytest.approx(1.5)


def test_win_rate_half() -> None:
    pnl = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert win_rate(pnl) == pytest.approx(0.5)


def test_annualized_return() -> None:
    pnl = pd.Series([0.001] * 252)
    assert annualized_return(pnl) == pytest.approx(0.252)
