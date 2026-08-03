"""Tests for weekend_swing.scan (the shared cheap weekend watchlist builder)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from signal_engine.weekend_swing.scan import build_weekend_watchlist, next_or_current_monday


class TestNextOrCurrentMonday:
    def test_saturday_rolls_to_next_monday(self):
        result = next_or_current_monday(pd.Timestamp("2024-01-13"))  # Saturday
        assert result == pd.Timestamp("2024-01-15")

    def test_sunday_rolls_to_next_monday(self):
        result = next_or_current_monday(pd.Timestamp("2024-01-14"))  # Sunday
        assert result == pd.Timestamp("2024-01-15")

    def test_monday_stays_on_itself(self):
        result = next_or_current_monday(pd.Timestamp("2024-01-08"))  # Monday
        assert result == pd.Timestamp("2024-01-08")

    def test_midweek_rolls_to_next_monday(self):
        result = next_or_current_monday(pd.Timestamp("2024-01-10"))  # Wednesday
        assert result == pd.Timestamp("2024-01-15")


WEEK_START = pd.Timestamp("2024-01-08")

CONFIG = {
    "universe": {"top_n": 100, "by": "volume", "core_tickers": ["SPY"], "pivot_proximity_pct": 2.0},
    "strategy_1_breakout": {"volume_filter_multiplier": 1.2, "atr_period": 14, "atr_stop_multiplier": 0.3},
    "strategy_2_overnight": {"atr_range_multiplier": 1.5, "atr_period": 14, "trigger_window_minutes": 30},
    "strategy_3_gap": {"restricted_tickers": ["SPY"]},
}


def _daily_df(near_pivot: bool) -> pd.DataFrame:
    dates = pd.bdate_range(end="2024-01-05", periods=40)
    rows, price = [], 95.0
    for d in dates:
        price += 0.1
        rows.append({"open": price, "high": price + 1.0, "low": price - 1.0, "close": price})
    df = pd.DataFrame(rows, index=dates)
    df.loc["2024-01-01":"2024-01-05", "high"] = [102, 103, 104, 105, 106]
    df.loc["2024-01-01":"2024-01-05", "low"] = [99, 100, 101, 102, 103]
    # Last close near the week_high (106) if near_pivot, else far away from both levels.
    df.iloc[-1, df.columns.get_loc("close")] = 105.5 if near_pivot else 50.0
    return df


class TestBuildWeekendWatchlist:
    def test_candidate_included_when_within_proximity(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = []
        bar_store = MagicMock()
        bar_store.get_bars.return_value = _daily_df(near_pivot=True)

        result = build_weekend_watchlist(CONFIG, alpaca, finnhub_client=None, bar_store=bar_store, week_start=WEEK_START)

        tickers = {c["ticker"] for c in result["candidates"]}
        assert "SPY" in tickers

    def test_candidate_excluded_when_far_from_all_pivots(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = []
        bar_store = MagicMock()
        bar_store.get_bars.return_value = _daily_df(near_pivot=False)

        result = build_weekend_watchlist(CONFIG, alpaca, finnhub_client=None, bar_store=bar_store, week_start=WEEK_START)

        assert result["candidates"] == []

    def test_skipped_tickers_are_logged_not_silently_dropped(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = []
        bar_store = MagicMock()
        bar_store.get_bars.return_value = pd.DataFrame()  # empty -> skip

        result = build_weekend_watchlist(CONFIG, alpaca, finnhub_client=None, bar_store=bar_store, week_start=WEEK_START)

        assert result["skipped"] == [("SPY", "no daily data available")]

    def test_fetch_exception_is_logged_not_raised(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = []
        bar_store = MagicMock()
        bar_store.get_bars.side_effect = RuntimeError("rate limited")

        result = build_weekend_watchlist(CONFIG, alpaca, finnhub_client=None, bar_store=bar_store, week_start=WEEK_START)

        assert result["skipped"][0][0] == "SPY"
        assert "rate limited" in result["skipped"][0][1]

    def test_returns_week_bounds_and_universe_size(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = ["AAPL"]
        bar_store = MagicMock()
        bar_store.get_bars.return_value = pd.DataFrame()

        result = build_weekend_watchlist(CONFIG, alpaca, finnhub_client=None, bar_store=bar_store, week_start=WEEK_START)

        assert result["week_start"] == WEEK_START
        assert result["week_end"] == WEEK_START + pd.Timedelta(days=4)
        assert result["universe_size"] == 2  # core SPY + screener AAPL
