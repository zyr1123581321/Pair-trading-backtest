"""Visualization: equity curve, drawdown, spread examples."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import config


def plot_equity_curve(
    portfolio: pd.DataFrame,
    title: str = "Portfolio Equity Curve",
    save_path: str | None = None,
) -> None:
    """Plot cumulative PnL over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(portfolio.index, portfolio["cumulative_pnl"], linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Cumulative PnL (fraction of capital)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_drawdown(
    portfolio: pd.DataFrame,
    title: str = "Portfolio Drawdown",
    save_path: str | None = None,
) -> None:
    """Plot drawdown from the running peak."""
    cum = portfolio["cumulative_pnl"]
    drawdown = cum.cummax() - cum

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown, alpha=0.5, color="salmon")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_pair_spread(
    spread: pd.Series,
    zscore: pd.Series,
    positions: pd.Series,
    pair_name: str,
    save_path: str | None = None,
) -> None:
    """Plot the spread, z-score, and position signals for a single pair."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(spread.index, spread, linewidth=0.8)
    axes[0].set_title(f"{pair_name} — Spread")
    axes[0].set_ylabel("Spread")

    axes[1].plot(zscore.index, zscore, linewidth=0.8, color="tab:orange")
    axes[1].axhline(config.ENTRY_Z, color="red", linestyle="--", linewidth=0.7)
    axes[1].axhline(-config.ENTRY_Z, color="green", linestyle="--", linewidth=0.7)
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[1].set_title("Z-Score")
    axes[1].set_ylabel("Z")

    axes[2].plot(positions.index, positions, linewidth=0.8, color="tab:purple")
    axes[2].set_title("Position")
    axes[2].set_ylabel("Signal")
    axes[2].set_xlabel("Date")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
