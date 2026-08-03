"""Tests for weekend_swing.weekend_swing_signals.OvernightRangeBreakoutStrategy."""

import pandas as pd
import pytest

from signal_engine.weekend_swing.weekend_swing_signals import OvernightRangeBreakoutStrategy

CONFIG = {
    "strategy_2_overnight": {
        "atr_range_multiplier": 1.5,
        "atr_period": 14,
        "trigger_window_minutes": 30,
    }
}

WEEK_START = pd.Timestamp("2024-01-08")
SESSION_OPEN = pd.Timestamp("2024-01-08 09:30")


def _daily_df(friday_close=105.0, monday_open=106.0) -> pd.DataFrame:
    """~40 business days ending Fri 2024-01-05, low volatility (ATR ~ 2.0), tight overnight gap."""
    dates = pd.bdate_range(end="2024-01-05", periods=40)
    rows = []
    price = 100.0
    for d in dates:
        price += 0.05
        rows.append({"open": price, "high": price + 1.0, "low": price - 1.0, "close": price})
    df = pd.DataFrame(rows, index=dates)
    df.loc["2024-01-05", ["open", "high", "low", "close"]] = [
        friday_close - 1, friday_close + 0.5, friday_close - 1.5, friday_close,
    ]

    monday_row = pd.DataFrame(
        [{"open": monday_open, "high": monday_open + 1.0, "low": monday_open - 1.0, "close": monday_open + 0.2}],
        index=[WEEK_START],
    )
    return pd.concat([df, monday_row])


class TestScan:
    def test_returns_levels_for_tight_overnight_range(self):
        # NOTE: the skip threshold is 1.5 * ATR14/24 — dividing by 24 makes it
        # tiny relative to a typical daily-scale gap, per the spec's literal
        # formula. A "tight" gap here means a few cents, not a full point.
        strategy = OvernightRangeBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(friday_close=105.0, monday_open=105.05), WEEK_START)

        assert levels is not None
        assert levels["overnight_high"] == pytest.approx(105.05)
        assert levels["overnight_low"] == 105.0
        assert levels["data_quality"] == "regular_session_proxy_no_extended_hours"

    def test_skips_ticker_when_overnight_range_too_wide(self):
        # Gap of 20 points vastly exceeds 1.5x ATR14/24 for a ~2.0 ATR series
        strategy = OvernightRangeBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(friday_close=105.0, monday_open=125.0), WEEK_START)

        assert levels is None


class TestCheckTrigger:
    def test_long_trigger_within_window(self):
        strategy = OvernightRangeBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(friday_close=105.0, monday_open=105.05), WEEK_START)

        bars = pd.DataFrame(
            [{"open": 106.0, "high": 107.5, "low": 105.9, "close": 107.0, "volume": 1000}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:35")]),
        )

        signal = strategy.check_trigger(levels, bars, SESSION_OPEN)

        assert signal is not None
        assert signal.direction == "long"
        assert signal.stop_price == levels["overnight_low"]

    def test_short_trigger_within_window(self):
        strategy = OvernightRangeBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(friday_close=105.0, monday_open=105.05), WEEK_START)

        bars = pd.DataFrame(
            [{"open": 105.0, "high": 105.2, "low": 103.5, "close": 104.0, "volume": 1000}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:35")]),
        )

        signal = strategy.check_trigger(levels, bars, SESSION_OPEN)

        assert signal is not None
        assert signal.direction == "short"
        assert signal.stop_price == levels["overnight_high"]

    def test_no_trigger_outside_window(self):
        strategy = OvernightRangeBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(friday_close=105.0, monday_open=105.05), WEEK_START)

        # 45 minutes after open — outside the 30-min trigger window
        bars = pd.DataFrame(
            [{"open": 106.0, "high": 107.5, "low": 105.9, "close": 107.0, "volume": 1000}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 10:15")]),
        )

        assert strategy.check_trigger(levels, bars, SESSION_OPEN) is None

    def test_no_trigger_when_price_stays_in_range(self):
        strategy = OvernightRangeBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(friday_close=105.0, monday_open=105.05), WEEK_START)

        bars = pd.DataFrame(
            [{"open": 105.02, "high": 105.04, "low": 105.01, "close": 105.02, "volume": 1000}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:35")]),
        )

        assert strategy.check_trigger(levels, bars, SESSION_OPEN) is None
