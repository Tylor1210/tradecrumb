"""Tests for the market trend filter (risk/trend_filter.py)."""

import numpy as np
import pandas as pd
import pytest

from signal_engine.risk.trend_filter import TrendFilterRule, apply_trend_filter, uptrend_mask


def _df(prices, idx):
    return pd.DataFrame({"close": pd.Series(prices, index=idx, dtype=float)})


def test_from_config_off_by_default():
    assert TrendFilterRule.from_config(None) is None
    assert TrendFilterRule.from_config({"enabled": False}) is None
    rule = TrendFilterRule.from_config({"enabled": True, "sma_days": 50, "index_symbol": "QQQ"})
    assert rule == TrendFilterRule(index_symbol="QQQ", sma_days=50, require_ticker_above_sma=False)


def test_uptrend_mask_false_until_average_exists():
    idx = pd.bdate_range("2024-01-01", periods=6)
    mask = uptrend_mask(pd.Series([1, 2, 3, 4, 5, 6], index=idx), 3)
    assert list(mask) == [False, False, True, True, True, True]


def test_uptrend_mask_point_in_time():
    # Changing a later bar must not change earlier mask values.
    idx = pd.bdate_range("2024-01-01", periods=30)
    base = pd.Series(np.linspace(100, 130, 30), index=idx)
    later = base.copy()
    later.iloc[25:] = 50.0
    a = uptrend_mask(base, 10)
    b = uptrend_mask(later, 10)
    assert a.iloc[:25].equals(b.iloc[:25])


def test_filter_drops_signals_when_market_below_average():
    idx = pd.bdate_range("2024-01-01", periods=12)
    # market rises, then falls hard from bar 8 on
    spy = [100, 101, 102, 103, 104, 105, 106, 107, 90, 85, 80, 75]
    price_data = {"SPY": _df(spy, idx), "AAA": _df([10] * 12, idx)}
    sig = pd.Series(True, index=idx)
    rule = TrendFilterRule(sma_days=3)
    filtered, removed = apply_trend_filter({"AAA": sig}, price_data, rule)
    out = filtered["AAA"]
    assert not out.iloc[:2].any()          # no average yet
    assert out.iloc[2:8].all()             # uptrend
    assert not out.iloc[8:].any()          # below the 3-day average
    assert removed == 2 + 4


def test_filter_can_require_ticker_uptrend():
    idx = pd.bdate_range("2024-01-01", periods=8)
    price_data = {
        "SPY": _df(np.linspace(100, 110, 8), idx),
        "UP": _df(np.linspace(10, 20, 8), idx),
        "DOWN": _df(np.linspace(20, 10, 8), idx),
    }
    sigs = {"UP": pd.Series(True, index=idx), "DOWN": pd.Series(True, index=idx)}
    filtered, _ = apply_trend_filter(sigs, price_data, TrendFilterRule(sma_days=3, require_ticker_above_sma=True))
    assert filtered["UP"].iloc[2:].all()
    assert not filtered["DOWN"].any()


def test_filter_never_adds_signals():
    idx = pd.bdate_range("2024-01-01", periods=10)
    price_data = {"SPY": _df(np.linspace(100, 120, 10), idx)}
    sig = pd.Series([False, True] * 5, index=idx)
    filtered, _ = apply_trend_filter({"X": sig}, price_data, TrendFilterRule(sma_days=2))
    assert not (filtered["X"] & ~sig).any()


def test_missing_index_raises():
    idx = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(ValueError):
        apply_trend_filter({"X": pd.Series(True, index=idx)}, {}, TrendFilterRule())
