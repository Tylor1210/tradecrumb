"""
Unit tests for technical signal computation.

Tests use deterministic synthetic OHLCV data — no API calls.
"""

import pandas as pd
import numpy as np
import pytest

from signal_engine.signals.technical import TechnicalAnalyzer, TechnicalSignal

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "sma_fast": 20,
    "sma_mid": 50,
    "sma_slow": 200,
    "rsi_period": 14,
    "volume_window": 20,
    "breakout_window": 20,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "component_weights": {
        "ma": 0.30,
        "rsi": 0.20,
        "macd": 0.25,
        "volume": 0.10,
        "breakout": 0.15,
    },
}


def make_ohlcv(n: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")
    base = 100.0

    if trend == "up":
        closes = [base + i * 0.5 + rng.normal(0, 0.5) for i in range(n)]
    elif trend == "breakout_up":
        # Flat for most of the period, then a steady breakout in the last 40 days
        closes = [base + rng.normal(0, 0.3) for _ in range(n - 40)]
        last_close = closes[-1]
        closes += [last_close + (i + 1) * 0.8 + rng.normal(0, 0.3) for i in range(40)]
    elif trend == "down":
        closes = [max(1.0, base - i * 0.3 + rng.normal(0, 0.5)) for i in range(n)]
    elif trend == "flat":
        closes = [base + rng.normal(0, 0.3) for _ in range(n)]
    elif trend == "crash":
        # Slow rise then sudden crash at the end
        closes = [base + i * 0.4 for i in range(n - 10)]
        crash_start = closes[-1]
        closes += [crash_start * (1 - j * 0.05) for j in range(1, 11)]
    else:
        closes = [base] * n

    closes = [max(1.0, float(c)) for c in closes]

    df = pd.DataFrame(
        {
            "open": [c * 0.995 for c in closes],
            "high": [c * 1.015 for c in closes],
            "low": [c * 0.985 for c in closes],
            "close": closes,
            "volume": [1_000_000 + int(rng.integers(-200_000, 200_000)) for _ in range(n)],
        },
        index=dates,
    )
    return df


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestTechnicalAnalyzer:

    def test_returns_technical_signal_instance(self):
        df = make_ohlcv(300)
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert isinstance(result, TechnicalSignal)

    def test_uptrend_produces_positive_score(self):
        """A consistently rising price series should score > 0."""
        df = make_ohlcv(300, trend="breakout_up")
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert result is not None
        assert result.score > 0, f"Expected positive score for uptrend, got {result.score}"

    def test_downtrend_produces_negative_score(self):
        """A consistently falling price series should score < 0."""
        df = make_ohlcv(300, trend="down")
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert result is not None
        assert result.score < 0, f"Expected negative score for downtrend, got {result.score}"

    def test_score_always_in_bounds(self):
        """Score must always be in [-1.0, 1.0] regardless of market conditions."""
        for trend in ["up", "down", "flat", "crash"]:
            df = make_ohlcv(310, trend=trend)
            analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
            result = analyzer.compute("TEST", df)
            assert result is not None
            assert -1.0 <= result.score <= 1.0, (
                f"Score out of bounds for {trend} trend: {result.score}"
            )

    def test_component_scores_in_bounds(self):
        """All individual component scores must be in [-1.0, 1.0]."""
        df = make_ohlcv(300)
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert result is not None

        components = {
            "ma_score": result.ma_score,
            "rsi_score": result.rsi_score,
            "macd_score": result.macd_score,
            "volume_score": result.volume_score,
            "breakout_score": result.breakout_score,
        }
        for name, val in components.items():
            assert -1.0 <= val <= 1.0, f"{name} out of bounds: {val}"

    def test_breakdown_contains_all_expected_keys(self):
        """Breakdown dict must contain all signal component keys."""
        df = make_ohlcv(300)
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert result is not None

        expected = {"moving_averages", "rsi", "macd", "volume", "breakout", "component_weights"}
        missing = expected - set(result.breakdown.keys())
        assert not missing, f"Missing breakdown keys: {missing}"

    def test_insufficient_data_returns_none(self):
        """Should return None when data has fewer than sma_slow + 10 rows."""
        df = make_ohlcv(50)  # Way less than sma_slow=200
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert result is None, "Expected None for insufficient data"

    def test_exactly_at_minimum_data_boundary(self):
        """At exactly the minimum required rows, should not crash."""
        min_rows = DEFAULT_CONFIG["sma_slow"] + 10  # 210
        df = make_ohlcv(min_rows)
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        # Should either return None or a valid result — not raise
        result = analyzer.compute("TEST", df)
        if result is not None:
            assert -1.0 <= result.score <= 1.0

    def test_ma_breakdown_contains_ma_values(self):
        """Moving averages breakdown should have SMA values."""
        df = make_ohlcv(300)
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        ma_breakdown = result.breakdown.get("moving_averages", {})

        assert "current_price" in ma_breakdown
        assert "sma_fast" in ma_breakdown
        assert "sma_mid" in ma_breakdown
        assert "sma_slow" in ma_breakdown

    def test_rsi_breakdown_contains_rsi_value(self):
        """RSI breakdown should have the RSI numeric value."""
        df = make_ohlcv(300)
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        rsi_breakdown = result.breakdown.get("rsi", {})

        assert "rsi_value" in rsi_breakdown
        rsi_val = rsi_breakdown["rsi_value"]
        assert 0.0 <= rsi_val <= 100.0, f"RSI value out of range: {rsi_val}"

    def test_uptrend_price_above_sma200(self):
        """In a strong uptrend, price should be above SMA-200."""
        df = make_ohlcv(300, trend="breakout_up")
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        ma_breakdown = result.breakdown.get("moving_averages", {})
        assert ma_breakdown.get("above_sma200") is True

    def test_downtrend_price_below_sma200(self):
        """In a sustained downtrend, price should eventually be below SMA-200."""
        df = make_ohlcv(300, trend="down")
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        ma_breakdown = result.breakdown.get("moving_averages", {})
        assert ma_breakdown.get("above_sma200") is False

    def test_none_input_returns_none(self):
        """None input DataFrame should return None gracefully."""
        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", None)
        assert result is None

    def test_breakout_signal_on_new_high(self):
        """When price hits a new 20-day high, breakout_score should be 1.0."""
        df = make_ohlcv(250)
        # Force the last row to a new high
        df = df.copy()
        df.loc[df.index[-1], "close"] = df["high"].iloc[-21:-1].max() * 1.05
        df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] * 1.01

        analyzer = TechnicalAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("TEST", df)
        assert result is not None
        assert result.breakout_score == 1.0, f"Expected breakout_score=1.0, got {result.breakout_score}"
