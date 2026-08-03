"""Tests for weekend_swing.weekend_swing_signals.PriorWeekBreakoutStrategy."""

import pandas as pd
import pytest

from signal_engine.weekend_swing.weekend_swing_signals import PriorWeekBreakoutStrategy

CONFIG = {
    "strategy_1_breakout": {
        "volume_filter_multiplier": 1.2,
        "volume_filter_max_multiplier": 3.0,
        "atr_period": 14,
        "atr_stop_multiplier": 0.3,
        "exclude_earnings_week": True,
    },
    "universe": {"max_single_day_move_pct": 50.0},
}

WEEK_START = pd.Timestamp("2024-01-08")  # Monday


def _daily_df() -> pd.DataFrame:
    """~40 business days ending Fri 2024-01-05, gentle drift, low volatility (so ATR is well-defined)."""
    dates = pd.bdate_range(end="2024-01-05", periods=40)
    rows = []
    price = 95.0
    for i, d in enumerate(dates):
        price += 0.1
        rows.append({"open": price, "high": price + 1.0, "low": price - 1.0, "close": price})
    df = pd.DataFrame(rows, index=dates)
    # Force the final week (Jan 1-5) to a known high/low for deterministic assertions.
    df.loc["2024-01-01":"2024-01-05", "high"] = [102, 103, 104, 105, 106]
    df.loc["2024-01-01":"2024-01-05", "low"] = [99, 100, 101, 102, 103]
    return df


def _historical_15min_df(baseline_volume: float = 1000.0) -> pd.DataFrame:
    """20 trading days of 09:30/09:45 15-min bars at a stable baseline volume."""
    dates = pd.bdate_range(end="2024-01-05", periods=20)
    rows = []
    idx = []
    for d in dates:
        for hm in ["09:30", "09:45"]:
            h, m = map(int, hm.split(":"))
            ts = d.replace(hour=h, minute=m)
            idx.append(ts)
            rows.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": baseline_volume})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


class TestScan:
    def test_returns_levels_with_expected_pivot_and_volume_baseline(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        assert levels is not None
        assert levels["week_high"] == 106.0
        assert levels["week_low"] == 99.0
        assert levels["avg_volume_by_slot"]["09:30"] == pytest.approx(1000.0)

    def test_excludes_earnings_week_tickers(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan(
            "AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded={"AAPL"}
        )
        assert levels is None

    def test_does_not_exclude_when_earnings_filter_disabled(self):
        cfg = {"strategy_1_breakout": {**CONFIG["strategy_1_breakout"], "exclude_earnings_week": False}}
        strategy = PriorWeekBreakoutStrategy(cfg)
        levels = strategy.scan(
            "AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded={"AAPL"}
        )
        assert levels is not None


class TestCheckTrigger:
    def test_long_trigger_on_breakout_above_week_high_with_volume(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        monday_bar = pd.DataFrame(
            [{"open": 106.5, "high": 108.0, "low": 106.2, "close": 107.0, "volume": 1500.0}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30")]),
        )

        signal = strategy.check_trigger(levels, monday_bar)

        assert signal is not None
        assert signal.direction == "long"
        assert signal.score > 0
        assert signal.entry_price == 107.0
        # stop = min(candle low, week_high - 0.3*ATR) — must be below entry
        assert signal.stop_price < signal.entry_price

    def test_short_trigger_on_breakdown_below_week_low_with_volume(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        monday_bar = pd.DataFrame(
            [{"open": 98.5, "high": 98.8, "low": 97.0, "close": 98.0, "volume": 1500.0}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30")]),
        )

        signal = strategy.check_trigger(levels, monday_bar)

        assert signal is not None
        assert signal.direction == "short"
        assert signal.score < 0
        assert signal.stop_price > signal.entry_price

    def test_no_trigger_when_volume_filter_not_met(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        # Price breaks above week_high, but volume is only at baseline (no 1.2x surge)
        monday_bar = pd.DataFrame(
            [{"open": 106.5, "high": 108.0, "low": 106.2, "close": 107.0, "volume": 1000.0}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30")]),
        )

        assert strategy.check_trigger(levels, monday_bar) is None

    def test_no_trigger_when_price_stays_within_range(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        monday_bar = pd.DataFrame(
            [{"open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 5000.0}],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30")]),
        )

        assert strategy.check_trigger(levels, monday_bar) is None

    def test_returns_first_triggering_bar_not_a_later_one(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        bars = pd.DataFrame(
            [
                {"open": 106.5, "high": 108.0, "low": 106.2, "close": 107.0, "volume": 1500.0},  # triggers here
                {"open": 107.0, "high": 109.0, "low": 106.8, "close": 108.5, "volume": 1500.0},
            ],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30"), pd.Timestamp("2024-01-08 09:45")]),
        )

        signal = strategy.check_trigger(levels, bars)

        assert signal.date == pd.Timestamp("2024-01-08 09:30")

    def test_no_trigger_when_volume_exceeds_the_ceiling(self):
        """Backtesting found >3x volume spikes correlate with WORSE outcomes
        (exhaustion, not continuation) — the filter is a band, not just a floor."""
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        monday_bar = pd.DataFrame(
            [{"open": 106.5, "high": 108.0, "low": 106.2, "close": 107.0, "volume": 5000.0}],  # 5x baseline
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30")]),
        )

        assert strategy.check_trigger(levels, monday_bar) is None

    def test_triggers_within_the_volume_band(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())

        monday_bar = pd.DataFrame(
            [{"open": 106.5, "high": 108.0, "low": 106.2, "close": 107.0, "volume": 2000.0}],  # 2x baseline
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 09:30")]),
        )

        assert strategy.check_trigger(levels, monday_bar) is not None


class TestDataSanityGuard:
    def test_rejects_ticker_with_anomalous_single_day_move(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        df = _daily_df()
        # Inject a >50% single-day jump — simulates corrupted/unadjusted screener data
        df.iloc[-5, df.columns.get_loc("close")] = df.iloc[-6]["close"] * 3.0

        levels = strategy.scan("LBGJ", df, WEEK_START, _historical_15min_df(), earnings_excluded=set())

        assert levels is None

    def test_normal_data_is_not_rejected(self):
        strategy = PriorWeekBreakoutStrategy(CONFIG)
        levels = strategy.scan("AAPL", _daily_df(), WEEK_START, _historical_15min_df(), earnings_excluded=set())
        assert levels is not None
