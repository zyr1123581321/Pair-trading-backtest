"""Pair selection via Engle-Granger cointegration tests."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import coint

import config


def engle_granger_screen(
    prices: pd.DataFrame,
    p_threshold: float = config.COINT_PVALUE,
    sector_map: dict = None
) -> pd.DataFrame:
    """Test all ticker pairs for cointegration, return significant pairs sorted by t-stat."""
    row_list = []
    
    tickers = prices.columns.tolist()
    
    for a, b in combinations(tickers, 2):
        if (sector_map) and (sector_map[a] != sector_map[b]):
            continue            
        t_stat, p_value, _ = coint(prices[a], prices[b])
        if p_value < p_threshold:
            half_life = _half_life(prices[a], prices[b])
            dict_row = {'ticker_a': a, 
                        'ticker_b': b, 
                        't_stat': t_stat, 
                        'p_value': p_value, 
                        'half_life': half_life}
            row_list.append(dict_row)

    df = pd.DataFrame(row_list)     
    return df.sort_values(by='t_stat')   


def _half_life(y: pd.Series, x: pd.Series) -> float:
    """Half-life of mean reversion via AR(1) on the OLS spread. Returns np.inf if non-reverting."""
    s = _ols_spread(y, x)
    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    phi = np.polyfit(lag, delta, 1)[0]
    if phi >=0:
        return np.inf
    else: 
        return -math.log(2) / phi


def _ols_spread(y: pd.Series, x: pd.Series) -> pd.Series:
    """Return the OLS spread s = y - beta * x."""
    beta, _ = np.polyfit(x, y, 1)
    return y - beta * x
    

def filter_pairs(
    pairs: pd.DataFrame,
    top_n: int = config.TOP_N_PAIRS,
    hl_min: float = config.HALFLIFE_MIN,
    hl_max: float = config.HALFLIFE_MAX,
) -> pd.DataFrame:
    """Filter to pairs with half-life in [hl_min, hl_max], take top_n by t-stat."""
    pairs_filtered = pairs[(pairs["half_life"] >= hl_min) & (pairs["half_life"] <= hl_max)]
    
    return pairs_filtered.iloc[:top_n]