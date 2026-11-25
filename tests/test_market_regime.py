"""Market regime: fear/greed + pressure on synthetic bars, policy, store, no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from signal_engine.market_regime.policy import RegimePolicy
from signal_engine.market_regime.regime import compute_regime
from signal_engine.market_regime.store import RegimeStore

CFG = {
    "symbols": {"benchmark": "SPY", "bonds_long": "TLT", "junk": "HYG", "investment_grade": "LQD",
                "offensive": ["XLY", "XLK"], "defensive": ["XLP", "XLU"], "sectors": ["XLK", "XLE", "XLP", "XLU", "XLF"]},
    "fear_greed": {}, "pressure": {},
}


def bars(n=200, drift=0.001, close_loc=0.5, vol=1_000_000, seed=0, noise=0.004):
    """close_loc: where each bar closes inside its range (1 = at high = buying, 0 = at low = selling)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + drift + rng.normal(0, noise, n))
    high, low = close * 1.005, close * 0.995
    close = low + (high - low) * close_loc
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": float(vol)}, index=idx)


def world(drift, close_loc, defensive_drift=None):
    d = {"SPY": bars(drift=drift, close_loc=close_loc, seed=1),
         "TLT": bars(drift=0.0, seed=2), "HYG": bars(drift=drift, seed=3), "LQD": bars(drift=0.0, seed=4)}
    for i, s in enumerate(["XLY", "XLK", "XLE", "XLF"]):
        d[s] = bars(drift=drift, close_loc=close_loc, seed=10 + i)
    for i, s in enumerate(["XLP", "XLU"]):
        d[s] = bars(drift=defensive_drift if defensive_drift is not None else drift, close_loc=close_loc, seed=20 + i)
    for i in range(8):
        d[f"S{i}"] = bars(drift=drift, close_loc=close_loc, seed=30 + i)
    return d, [f"S{i}" for i in range(8)]


def test_uptrend_with_close_near_highs_is_greedy_and_bullish():
    d, br = world(drift=0.002, close_loc=0.9)
    r = compute_regime(d, CFG, br)
    assert r.fear_greed > 55 and r.pressure > 0.15 and r.pressure_label == "BULLISH"


def test_downtrend_with_close_near_lows_is_fearful_and_bearish():
    d, br = world(drift=-0.003, close_loc=0.1)
    r = compute_regime(d, CFG, br)
    assert r.fear_greed < 45 and r.pressure < -0.15 and r.pressure_label == "BEARISH"
    assert r.breadth_ad < 0.5


def test_no_lookahead_as_of_excludes_that_day():
    d, br = world(drift=0.001, close_loc=0.5)
    cutoff = d["SPY"].index[150]
    r = compute_regime(d, CFG, br, as_of=cutoff)
    assert r.as_of == str(d["SPY"].index[149].date())


def test_none_when_benchmark_missing_and_unavailable_when_few_components():
    assert compute_regime({}, CFG, []) is None
    only_spy = {"SPY": bars(n=30)}  # too short for SMA125, no breadth/bonds/sectors
    r = compute_regime(only_spy, CFG, [])
    assert r.fear_greed is None and r.fear_greed_label == "Unavailable"


def test_sector_flow_ranks_buying_first_and_summary_names_it():
    d, br = world(drift=0.0, close_loc=0.5)
    d["XLE"] = bars(drift=0.0, close_loc=0.95, seed=99)
    d["XLF"] = bars(drift=0.0, close_loc=0.05, seed=98)
    r = compute_regime(d, CFG, br)
    assert r.sectors[0]["symbol"] == "XLE" and r.sectors[-1]["symbol"] == "XLF"
    assert any("Buying concentrated in XLE" in s for s in r.summary)


def _regime(pressure, fg):
    return {"pressure": pressure, "fear_greed": fg}


def test_policy_disabled_is_a_noop():
    assert RegimePolicy({}).evaluate("long", _regime(-0.9, 5)).size_mult == 1.0


def test_policy_reduces_counter_trend_and_can_skip():
    reduce = RegimePolicy({"policy": {"enabled": True, "counter_trend_action": "reduce", "counter_trend_size_mult": 0.5}})
    assert reduce.evaluate("long", _regime(-0.5, 50)).size_mult == 0.5
    assert reduce.evaluate("short", _regime(-0.5, 50)).size_mult == 1.0
    skip = RegimePolicy({"policy": {"enabled": True, "counter_trend_action": "skip"}})
    assert not skip.evaluate("short", _regime(0.5, 50)).allow


def test_policy_crowded_and_missing_regime():
    p = RegimePolicy({"policy": {"enabled": True, "crowded_size_mult": 0.5}})
    assert p.evaluate("long", _regime(0.0, 90)).size_mult == 0.5
    assert p.evaluate("short", _regime(0.0, 10)).size_mult == 0.5
    assert p.evaluate("long", None).allow


def test_store_roundtrip(tmp_path):
    d, br = world(drift=0.001, close_loc=0.6)
    r = compute_regime(d, CFG, br)
    s = RegimeStore(tmp_path / "r.db")
    s.save(r)
    s.save(r)  # same as_of replaces
    assert len(s.history()) == 1
    assert s.latest()["pressure"] == pytest.approx(r.pressure)
