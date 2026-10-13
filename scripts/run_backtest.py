"""End-to-end pairs trading backtest."""

from __future__ import annotations

import os
import sys

# Add project root to path so config and src are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import config
from src.pairs.data import download_prices, split_in_out
from src.pairs.selection import engle_granger_screen, filter_pairs
from src.pairs.signals import (
    rolling_hedge_ratio,
    compute_spread,
    compute_zscore,
    generate_positions,
)
from src.pairs.backtest import backtest_pair, aggregate_pairs
from src.pairs.metrics import summary_table
from src.pairs.plots import plot_equity_curve, plot_drawdown, plot_pair_spread


def select_and_backtest(
    selection_prices: pd.DataFrame,
    trade_prices: pd.DataFrame,
    sector_map: dict | None = None,
) -> pd.DataFrame:
    """Select pairs on selection_prices, backtest on trade_prices.

    Returns portfolio DataFrame with 'daily_pnl' and 'cumulative_pnl',
    or an empty DataFrame if no pairs are tradeable.
    """
    all_pairs = engle_granger_screen(selection_prices, sector_map=sector_map)
    selected = filter_pairs(all_pairs)

    if selected.empty:
        return pd.DataFrame()

    print(f"  Selected {len(selected)} pairs")

    pair_results: dict[tuple[str, str], pd.DataFrame] = {}

    for _, row in selected.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        if a not in trade_prices.columns or b not in trade_prices.columns:
            continue

        y = trade_prices[a]
        x = trade_prices[b]

        beta = rolling_hedge_ratio(y, x)
        spread = compute_spread(y, x, beta)
        zscore = compute_zscore(spread)
        positions = generate_positions(zscore)

        bt = backtest_pair(y, x, positions, beta)
        pair_results[(a, b)] = bt

    portfolio = aggregate_pairs(pair_results)
    if not portfolio.empty:
        portfolio = portfolio.dropna()

    return portfolio


def report_and_plot(
    portfolio: pd.DataFrame,
    label: str,
    suffix: str = "",
) -> None:
    """Compute metrics, save CSV, and generate plots for a portfolio."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    metrics = summary_table(
        portfolio["daily_pnl"],
        portfolio["cumulative_pnl"],
        pd.Series(0.0, index=portfolio.index),
    )
    print(f"\n  Metrics ({label}{suffix}):")
    print(metrics.to_string())
    metrics.to_csv(os.path.join(config.RESULTS_DIR, f"metrics_{label}{suffix}.csv"))

    plot_equity_curve(
        portfolio,
        title=f"Equity Curve ({label}{suffix})",
        save_path=os.path.join(config.RESULTS_DIR, f"equity_{label}{suffix}.png"),
    )
    plot_drawdown(
        portfolio,
        title=f"Drawdown ({label}{suffix})",
        save_path=os.path.join(config.RESULTS_DIR, f"drawdown_{label}{suffix}.png"),
    )


def run_strategy(
    in_sample: pd.DataFrame,
    out_of_sample: pd.DataFrame,
    sector_map: dict | None = None,
    tag: str = "",
) -> None:
    """Run pair selection + backtest on fixed in-sample / out-of-sample split."""
    suffix = f"_{tag}" if tag else ""
    label_prefix = f"[{tag}] " if tag else ""

    print(f"\n{'#'*60}")
    print(f"  {label_prefix}Running cointegration tests...")
    print(f"{'#'*60}")

    for label, trade_prices in [("in_sample", in_sample), ("out_of_sample", out_of_sample)]:
        print(f"\n{'='*60}")
        print(f"  {label_prefix}Backtesting: {label}")
        print(f"{'='*60}")

        portfolio = select_and_backtest(in_sample, trade_prices, sector_map=sector_map)

        if portfolio.empty:
            print("  No tradeable pairs in this period.")
            continue

        report_and_plot(portfolio, label, suffix)

    print(f"\n  {label_prefix}Done.")


def generate_windows(
    prices: pd.DataFrame,
    lookback_years: int = 2,
    trade_months: int = 12,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Rolling walk-forward windows: (lookback_start, lookback_end, trade_start, trade_end) tuples."""
    date_start = pd.Timestamp(prices.index[0])
    date_end = pd.Timestamp(prices.index[-1])    
    sliding_windows = []

    current_lookback_start = date_start

    while True:
        lookback_start = current_lookback_start
        lookback_end = (
            lookback_start 
            + pd.DateOffset(years=lookback_years)
            - pd.DateOffset(days=1)
        )

        trade_start = lookback_start + pd.DateOffset(years=lookback_years)
        trade_end = (
            trade_start 
            + pd.DateOffset(months=trade_months)
            - pd.DateOffset(days=1)
        )
        if trade_end > date_end:
            break

        sliding_windows.append((lookback_start, lookback_end, trade_start, trade_end))
        current_lookback_start += pd.DateOffset(years=1)
    return sliding_windows

def run_walkforward(
    prices: pd.DataFrame,
    sector_map: dict | None = None,
    lookback_years: int = 2,
    trade_months: int = 12,
    tag: str = "walkforward",
) -> None:
    """Walk-forward backtest: re-select pairs each window, stitch results."""
    suffix = f"_{tag}" if tag else ""
    
    all_portfolios = []

    windows = generate_windows(prices, lookback_years, trade_months)
    for lookback_start, lookback_end, trade_start, trade_end in windows:
        lookback_prices = prices.loc[lookback_start:lookback_end]
        trade_prices = prices.loc[trade_start:trade_end]
        portfolio = select_and_backtest(lookback_prices, trade_prices, sector_map)
        if not portfolio.empty:
            all_portfolios.append(portfolio)
    combined_portfolio = pd.concat(all_portfolios)
    combined_portfolio["cumulative_pnl"] = combined_portfolio["daily_pnl"].cumsum()
    report_and_plot(combined_portfolio, tag, "")


def main() -> None:
    # ── 1. Data ──────────────────────────────────────────────────────
    print("Downloading / loading price data...")
    prices = download_prices()
    in_sample, out_of_sample = split_in_out(prices)
    print(f"  In-sample:      {in_sample.index[0].date()} → {in_sample.index[-1].date()}  "
          f"({len(in_sample)} days, {len(in_sample.columns)} tickers)")
    print(f"  Out-of-sample:  {out_of_sample.index[0].date()} → {out_of_sample.index[-1].date()}  "
          f"({len(out_of_sample)} days)")

    # ── 2. Baseline (no sector filter) ───────────────────────────────
    run_strategy(in_sample, out_of_sample, sector_map=None, tag="")

    # ── 3. Sector-constrained ────────────────────────────────────────
    run_strategy(in_sample, out_of_sample, sector_map=config.SECTOR_MAP, tag="sector")

    # ── 4. Walkforward Pair Selection (with sector map) ──────────────
    run_walkforward(prices, sector_map=config.SECTOR_MAP, lookback_years=2, trade_months=12, tag="walkforward")

    print("\nAll done. Results saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
