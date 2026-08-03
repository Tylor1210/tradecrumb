"""Tests for weekend_swing.weekend_swing_signals.GapStrengthStrategy (SPY/QQQ only, v1)."""

import pandas as pd
import pytest

from signal_engine.weekend_swing.weekend_swing_signals import GapStrengthStrategy

CONFIG = {"strategy_3_gap": {"restricted_tickers": ["SPY", "QQQ"], "atr_period": 14}}

WEEK_START = pd.Timestamp("2024-01-08")


def _daily_df() -> pd.DataFrame:
    dates = pd.bdate_range(end="2024-01-05", periods=40)
    rows = []
    price = 95.0
    for d in dates:
        price += 0.1
        rows.append({"open": price, "high": price + 1.0, "low": price - 1.0, "close": price})
    df = pd.DataFrame(rows, index=dates)
    df.loc["2024-01-01":"2024-01-05", "high"] = [102, 103, 104, 105, 106]  # pivot_high = 106
    df.loc["2024-01-01":"2024-01-05", "low"] = [99, 100, 101, 102, 103]     # pivot_low = 99
    return df


def _bar(open_, high, low, close, ts="2024-01-08 09:30") -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close}, name=pd.Timestamp(ts))


class TestScanScopeRestriction:
    def test_only_spy_and_qqq_are_scanned(self):
        strategy = GapStrengthStrategy(CONFIG)
        assert strategy.scan("SPY", _daily_df(), WEEK_START) is not None
        assert strategy.scan("QQQ", _daily_df(), WEEK_START) is not None
        assert strategy.scan("AAPL", _daily_df(), WEEK_START) is None

    def test_restriction_is_hard_enforced_even_if_config_tries_to_widen_it(self):
        cfg = {"strategy_3_gap": {"restricted_tickers": ["SPY", "QQQ", "AAPL", "TSLA"]}}
        strategy = GapStrengthStrategy(cfg)
        assert strategy.scan("AAPL", _daily_df(), WEEK_START) is None
        assert strategy.scan("TSLA", _daily_df(), WEEK_START) is None
        assert strategy.scan("SPY", _daily_df(), WEEK_START) is not None

    def test_pivot_levels_from_prior_week(self):
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)
        assert levels["pivot_high"] == 106.0
        assert levels["pivot_low"] == 99.0


class TestGapAndGo:
    def test_gap_up_holds_beyond_pivot(self):
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)

        signal = strategy.check_trigger(
            levels, monday_open_price=108.0,
            first_5min_bar=_bar(108.0, 108.5, 107.8, 108.3),  # closes above pivot_high (106)
        )

        assert signal is not None
        assert signal.breakdown["gap_type"] == "gap_and_go"
        assert signal.direction == "long"
        assert signal.stop_price == 106.0  # pivot level

    def test_gap_down_holds_beyond_pivot(self):
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)

        signal = strategy.check_trigger(
            levels, monday_open_price=95.0,
            first_5min_bar=_bar(95.0, 95.5, 94.0, 94.5),  # closes below pivot_low (99)
        )

        assert signal is not None
        assert signal.breakdown["gap_type"] == "gap_and_go"
        assert signal.direction == "short"
        assert signal.stop_price == 99.0


class TestGapAndFail:
    def test_gap_up_fails_back_through_pivot(self):
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)

        first_bar = _bar(108.0, 108.5, 105.5, 105.8)  # opens above pivot, closes back below it
        signal = strategy.check_trigger(levels, monday_open_price=108.0, first_5min_bar=first_bar)

        assert signal is not None
        assert signal.breakdown["gap_type"] == "gap_and_fail"
        assert signal.direction == "short"  # fading the failed gap-up
        assert signal.stop_price == 108.5  # gap-open candle's high

    def test_gap_down_fails_back_through_pivot(self):
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)

        first_bar = _bar(95.0, 99.5, 94.5, 99.3)  # opens below pivot, closes back above it
        signal = strategy.check_trigger(levels, monday_open_price=95.0, first_5min_bar=first_bar)

        assert signal is not None
        assert signal.breakdown["gap_type"] == "gap_and_fail"
        assert signal.direction == "long"  # fading the failed gap-down
        assert signal.stop_price == 94.5  # gap-open candle's low


class TestNoGap:
    def test_open_within_pivot_range_produces_no_signal(self):
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)

        signal = strategy.check_trigger(
            levels, monday_open_price=102.0,
            first_5min_bar=_bar(102.0, 102.5, 101.5, 102.2),
        )

        assert signal is None

    def test_gap_and_go_and_gap_and_fail_never_share_a_win_rate(self):
        """Sanity check on the interface: each call returns exactly one tagged
        outcome, never a merged/ambiguous result — separation happens at the
        backtest-aggregation layer by reading breakdown['gap_type']."""
        strategy = GapStrengthStrategy(CONFIG)
        levels = strategy.scan("SPY", _daily_df(), WEEK_START)

        go_signal = strategy.check_trigger(
            levels, monday_open_price=108.0, first_5min_bar=_bar(108.0, 108.5, 107.8, 108.3),
        )
        fail_signal = strategy.check_trigger(
            levels, monday_open_price=108.0, first_5min_bar=_bar(108.0, 108.5, 105.5, 105.8),
        )

        assert go_signal.breakdown["gap_type"] != fail_signal.breakdown["gap_type"]
