"""Central configuration for the pairs trading backtest."""

from datetime import date

# ── Universe ──────────────────────────────────────────────────────────
# ~50 liquid S&P 500 names across sectors.
# Survivorship bias is acknowledged in the README.
TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ADBE", "CRM", "INTC", "CSCO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABT", "MRK", "TMO", "ABBV", "LLY",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST",
    # Industrials
    "CAT", "HON", "UPS", "BA", "GE", "MMM",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Materials
    "LIN", "APD",
    # Utilities
    "NEE", "DUK",
    # Communication Services
    "DIS", "NFLX", "CMCSA",
]

# ── Date ranges ───────────────────────────────────────────────────────
IN_SAMPLE_START = date(2015, 1, 1)
IN_SAMPLE_END = date(2021, 12, 31)
OUT_OF_SAMPLE_START = date(2022, 1, 1)
OUT_OF_SAMPLE_END = date(2024, 12, 31)

DOWNLOAD_START = IN_SAMPLE_START
DOWNLOAD_END = OUT_OF_SAMPLE_END

# ── Pair selection ────────────────────────────────────────────────────
COINT_PVALUE = 0.05
TOP_N_PAIRS = 20
HALFLIFE_MIN = 5
HALFLIFE_MAX = 60

# ── Signal parameters ─────────────────────────────────────────────────
LOOKBACK_WINDOW = 60          # rolling window for hedge ratio, z-score
ENTRY_Z = 2.0                 # enter when |z| > ENTRY_Z
EXIT_Z = 0.0                  # exit when z crosses zero (mean reversion)
STOP_LOSS_Z = 3.0             # stop-loss when |z| > STOP_LOSS_Z

# ── Execution & costs ─────────────────────────────────────────────────
FILL_DELAY = 1                # signal at close t, fill at close t + FILL_DELAY
COST_BPS = 5                  # one-way cost per leg, in basis points
COST_FRAC = COST_BPS / 10_000

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
RESULTS_DIR = "results"
