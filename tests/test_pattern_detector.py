"""Tests for candlestick + chart pattern detection (signals/patterns.py)."""

import pandas as pd
import pytest

from signal_engine.signals.patterns import PatternDetector


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="B")
    return df


def _flat_bars(n: int, price: float = 100.0) -> list[dict]:
    """Boring flat bars used as padding before the pattern under test."""
    return [{"open": price, "high": price + 0.3, "low": price - 0.3, "close": price} for _ in range(n)]


class TestBullishEngulfing:
    def test_detects_bullish_engulfing(self):
        rows = _flat_bars(8)
        rows.append({"open": 100.5, "high": 100.6, "low": 99.4, "close": 99.5})   # bearish
        rows.append({"open": 99.3, "high": 101.2, "low": 99.2, "close": 101.0})    # engulfs it, bullish
        df = _make_df(rows)

        result = PatternDetector().compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Bullish Engulfing" in names
        assert result.score > 0


class TestBearishEngulfing:
    def test_detects_bearish_engulfing(self):
        rows = _flat_bars(8)
        rows.append({"open": 99.5, "high": 100.6, "low": 99.4, "close": 100.5})   # bullish
        rows.append({"open": 100.7, "high": 100.8, "low": 98.8, "close": 99.0})    # engulfs it, bearish
        df = _make_df(rows)

        result = PatternDetector().compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Bearish Engulfing" in names
        assert result.score < 0


class TestHammerShootingStar:
    def test_detects_hammer_after_downtrend(self):
        rows = _flat_bars(6, price=110)
        rows += [
            {"open": 110, "high": 110.5, "low": 108, "close": 108.2},
            {"open": 108.2, "high": 108.5, "low": 105, "close": 105.3},
            {"open": 105.3, "high": 105.6, "low": 102, "close": 102.3},
            {"open": 102.3, "high": 102.6, "low": 99, "close": 99.3},
        ]
        # Hammer: small body near the top, long lower wick, tiny upper wick
        rows.append({"open": 99.3, "high": 99.6, "low": 94.0, "close": 99.5})
        df = _make_df(rows)

        result = PatternDetector().compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Hammer" in names
        assert result.score > 0

    def test_detects_shooting_star_after_uptrend(self):
        rows = _flat_bars(6, price=90)
        rows += [
            {"open": 90, "high": 92.5, "low": 89.8, "close": 92},
            {"open": 92, "high": 94.5, "low": 91.8, "close": 94},
            {"open": 94, "high": 96.5, "low": 93.8, "close": 96},
            {"open": 96, "high": 98.5, "low": 95.8, "close": 98},
        ]
        # Shooting star: small body near the bottom, long upper wick, ~no lower wick
        rows.append({"open": 98, "high": 103.5, "low": 98.0, "close": 98.15})
        df = _make_df(rows)

        result = PatternDetector().compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Shooting Star" in names
        assert result.score < 0


class TestMorningEveningStar:
    def test_detects_morning_star(self):
        rows = _flat_bars(7)
        rows.append({"open": 100.5, "high": 100.6, "low": 96.0, "close": 96.3})    # long bearish
        rows.append({"open": 96.0, "high": 96.4, "low": 95.6, "close": 96.1})       # small indecision candle
        rows.append({"open": 96.3, "high": 99.5, "low": 96.2, "close": 99.2})       # strong bullish close > midpoint
        df = _make_df(rows)

        result = PatternDetector().compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Morning Star" in names
        assert result.score > 0

    def test_detects_evening_star(self):
        rows = _flat_bars(7)
        rows.append({"open": 96.0, "high": 100.5, "low": 95.9, "close": 100.2})    # long bullish
        rows.append({"open": 100.4, "high": 100.8, "low": 100.0, "close": 100.3})   # small indecision candle
        rows.append({"open": 100.2, "high": 100.3, "low": 97.0, "close": 97.3})     # strong bearish close < midpoint
        df = _make_df(rows)

        result = PatternDetector().compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Evening Star" in names
        assert result.score < 0


class TestDoubleTopBottom:
    def test_detects_double_bottom(self):
        # Two comparable troughs around 90, separated by a rally to ~100, then breaks above that high
        prices = [100, 98, 95, 92, 90, 92, 95, 98, 100, 98, 95, 92, 90.2, 93, 97, 101, 104]
        rows = [{"open": p, "high": p + 0.5, "low": p - 0.5, "close": p} for p in prices]
        df = _make_df(rows)

        result = PatternDetector(config={"chart_pattern_lookback": 20}).compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Double Bottom" in names

    def test_detects_double_top(self):
        prices = [90, 92, 95, 98, 100, 98, 95, 92, 90, 92, 95, 98, 99.8, 97, 93, 89, 86]
        rows = [{"open": p, "high": p + 0.5, "low": p - 0.5, "close": p} for p in prices]
        df = _make_df(rows)

        result = PatternDetector(config={"chart_pattern_lookback": 20}).compute("TEST", df)

        assert result is not None
        names = [m["name"] for m in result.matched_patterns]
        assert "Double Top" in names


class TestNoPattern:
    def test_flat_bars_produce_no_pattern(self):
        df = _make_df(_flat_bars(20))
        result = PatternDetector().compute("TEST", df)

        assert result is not None
        assert result.score == 0.0
        assert result.matched_patterns == []

    def test_insufficient_data_returns_none(self):
        df = _make_df(_flat_bars(5))
        result = PatternDetector().compute("TEST", df)
        assert result is None
