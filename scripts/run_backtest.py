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


def main() -> None:
    # ── 1. Data ──────────────────────────────────────────────────────
    print("Downloading / loading price data...")
    prices = download_prices()
    in_sample, out_of_sample = split_in_out(prices)
    print(f"  In-sample:      {in_sample.index[0].date()} → {in_sample.index[-1].date()}  "
          f"({len(in_sample)} days, {len(in_sample.columns)} tickers)")
    print(f"  Out-of-sample:  {out_of_sample.index[0].date()} → {out_of_sample.index[-1].date()}  "
          f"({len(out_of_sample)} days)")

    # ── 2. Pair selection (in-sample only) ───────────────────────────
    print("\nRunning cointegration tests on in-sample data...")
    all_pairs = engle_granger_screen(in_sample)
    print(f"  Significant pairs (p < {config.COINT_PVALUE}): {len(all_pairs)}")

    selected = filter_pairs(all_pairs)
    print(f"  After half-life filter [{config.HALFLIFE_MIN}, {config.HALFLIFE_MAX}]: {len(selected)}")
    print(selected[["ticker_a", "ticker_b", "t_stat", "p_value", "half_life"]].to_string(index=False))

    # ── 3. Backtest each pair on BOTH periods ────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for label, period_prices in [("in_sample", in_sample), ("out_of_sample", out_of_sample)]:
        print(f"\n{'='*60}")
        print(f"  Backtesting: {label}")
        print(f"{'='*60}")

        pair_results: dict[tuple[str, str], pd.DataFrame] = {}

        for _, row in selected.iterrows():
            a, b = row["ticker_a"], row["ticker_b"]
            if a not in period_prices.columns or b not in period_prices.columns:
                continue

            y = period_prices[a]
            x = period_prices[b]

            beta = rolling_hedge_ratio(y, x)
            spread = compute_spread(y, x, beta)
            zscore = compute_zscore(spread)
            positions = generate_positions(zscore)

            bt = backtest_pair(y, x, positions, beta)
            pair_results[(a, b)] = bt

        portfolio = aggregate_pairs(pair_results)

        if portfolio.empty:
            print("  No tradeable pairs in this period.")
            continue

        # Drop NaN rows from warm-up period
        portfolio = portfolio.dropna()

        # ── Metrics ──────────────────────────────────────────────
        # Build an aggregate position series for turnover
        all_positions = pd.DataFrame(
            {f"{a}_{b}": pair_results[(a, b)]["position"]
             for (a, b) in pair_results}
        ).mean(axis=1)

        metrics = summary_table(
            portfolio["daily_pnl"],
            portfolio["cumulative_pnl"],
            all_positions,
        )
        print(f"\n  Metrics ({label}):")
        print(metrics.to_string())

        metrics.to_csv(os.path.join(config.RESULTS_DIR, f"metrics_{label}.csv"))

        # ── Plots ────────────────────────────────────────────────
        plot_equity_curve(
            portfolio,
            title=f"Equity Curve ({label})",
            save_path=os.path.join(config.RESULTS_DIR, f"equity_{label}.png"),
        )
        plot_drawdown(
            portfolio,
            title=f"Drawdown ({label})",
            save_path=os.path.join(config.RESULTS_DIR, f"drawdown_{label}.png"),
        )

        # Plot the top pair's spread
        if pair_results:
            (a, b) = list(pair_results.keys())[0]
            y = period_prices[a]
            x = period_prices[b]
            beta = rolling_hedge_ratio(y, x)
            spread = compute_spread(y, x, beta)
            zscore = compute_zscore(spread)
            positions = generate_positions(zscore)
            plot_pair_spread(
                spread,
                zscore,
                positions,
                pair_name=f"{a} / {b}",
                save_path=os.path.join(config.RESULTS_DIR, f"spread_{a}_{b}_{label}.png"),
            )

    print("\nDone. Results saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
