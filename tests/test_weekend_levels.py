"""Tests for weekend_swing.levels pivot/range/ATR math (pure functions, synthetic OHLC)."""

import pandas as pd
import pytest

from signal_engine.weekend_swing.levels import (
    atr,
    avg_volume_by_time_slot,
    is_data_sane,
    overnight_range_proxy,
    prior_week_range,
    swing_pivot,
)


def _daily_df(rows: dict) -> pd.DataFrame:
    """rows: {date_str: {open, high, low, close}}"""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(list(rows.keys()))
    return df


# Week 1: Mon 2024-01-01 .. Fri 2024-01-05
# Week 2: Mon 2024-01-08 .. Fri 2024-01-12
WEEK1_WEEK2 = _daily_df({
    "2024-01-01": {"open": 100, "high": 102, "low": 99, "close": 101},
    "2024-01-02": {"open": 101, "high": 103, "low": 100, "close": 102},
    "2024-01-03": {"open": 102, "high": 104, "low": 101, "close": 103},
    "2024-01-04": {"open": 103, "high": 105, "low": 102, "close": 104},
    "2024-01-05": {"open": 104, "high": 106, "low": 103, "close": 105},  # prior week high=106, low=99
    "2024-01-08": {"open": 110, "high": 111, "low": 109, "close": 110.5},
    "2024-01-09": {"open": 110.5, "high": 112, "low": 110, "close": 111},
    "2024-01-10": {"open": 111, "high": 113, "low": 110.5, "close": 112},
    "2024-01-11": {"open": 112, "high": 114, "low": 111, "close": 113},
    "2024-01-12": {"open": 113, "high": 115, "low": 112, "close": 114},
})

WEEK2_MONDAY = pd.Timestamp("2024-01-08")


class TestPriorWeekRange:
    def test_computes_prior_week_high_low(self):
        result = prior_week_range(WEEK1_WEEK2, WEEK2_MONDAY)
        assert result == (106.0, 99.0)

    def test_returns_none_when_no_data_for_prior_week(self):
        result = prior_week_range(WEEK1_WEEK2, pd.Timestamp("2024-03-01"))
        assert result is None


class TestSwingPivot:
    def test_matches_prior_week_range(self):
        assert swing_pivot(WEEK1_WEEK2, WEEK2_MONDAY) == prior_week_range(WEEK1_WEEK2, WEEK2_MONDAY)


class TestOvernightRangeProxy:
    def test_uses_friday_close_and_monday_open(self):
        result = overnight_range_proxy(WEEK1_WEEK2, WEEK2_MONDAY)

        assert result["friday_close"] == 105.0
        assert result["monday_open"] == 110.0
        assert result["high"] == 110.0
        assert result["low"] == 105.0

    def test_tags_data_quality_caveat(self):
        result = overnight_range_proxy(WEEK1_WEEK2, WEEK2_MONDAY)
        assert result["data_quality"] == "regular_session_proxy_no_extended_hours"

    def test_returns_none_without_both_sides(self):
        friday_only = WEEK1_WEEK2.iloc[:5]  # just week 1
        assert overnight_range_proxy(friday_only, WEEK2_MONDAY) is None


class TestATR:
    def test_atr_converges_to_constant_true_range(self):
        # Every bar has high-low = 2.0 and small close moves -> TR ~= 2.0 throughout
        rows = {}
        price = 100.0
        for i in range(40):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            rows[date.strftime("%Y-%m-%d")] = {
                "open": price, "high": price + 1.0, "low": price - 1.0, "close": price,
            }
        df = _daily_df(rows)

        result = atr(df, period=14)

        # After the warmup period, ATR should settle near the constant TR of 2.0
        assert result.iloc[-1] == pytest.approx(2.0, abs=0.05)

    def test_atr_rises_with_larger_true_range(self):
        rows = {}
        price = 100.0
        for i in range(30):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            # Range widens from 1.0 to 5.0 over time
            half_range = 1.0 + (i / 30) * 4.0
            rows[date.strftime("%Y-%m-%d")] = {
                "open": price, "high": price + half_range, "low": price - half_range, "close": price,
            }
        df = _daily_df(rows)

        result = atr(df, period=14)

        assert result.iloc[-1] > result.iloc[15]

    def test_early_values_are_nan_before_period(self):
        rows = {}
        price = 100.0
        for i in range(10):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            rows[date.strftime("%Y-%m-%d")] = {
                "open": price, "high": price + 1.0, "low": price - 1.0, "close": price,
            }
        df = _daily_df(rows)

        result = atr(df, period=14)

        assert result.isna().all()  # fewer than `period` bars available


class TestAvgVolumeByTimeSlot:
    def _intraday_df(self, num_days: int, volume_by_day_fn=None) -> pd.DataFrame:
        rows, idx = [], []
        for i in range(num_days):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            vol = volume_by_day_fn(i) if volume_by_day_fn else 1000.0
            for hm in ["09:30", "09:45"]:
                h, m = map(int, hm.split(":"))
                idx.append(date.replace(hour=h, minute=m))
                rows.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": vol})
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))

    def test_averages_volume_per_slot(self):
        df = self._intraday_df(5)
        result = avg_volume_by_time_slot(df, lookback_days=20)
        assert result["09:30"] == pytest.approx(1000.0)
        assert result["09:45"] == pytest.approx(1000.0)

    def test_restricts_to_lookback_window(self):
        # First 5 days at volume=100, last 5 days at volume=2000 -> lookback=5 should only see 2000
        df = self._intraday_df(10, volume_by_day_fn=lambda i: 100.0 if i < 5 else 2000.0)
        result = avg_volume_by_time_slot(df, lookback_days=5)
        assert result["09:30"] == pytest.approx(2000.0)

    def test_empty_input_returns_empty_dict(self):
        assert avg_volume_by_time_slot(pd.DataFrame()) == {}


class TestIsDataSane:
    def test_accepts_normal_daily_moves(self):
        df = _daily_df({
            "2024-01-01": {"open": 100, "high": 101, "low": 99, "close": 100},
            "2024-01-02": {"open": 100, "high": 103, "low": 99, "close": 102},   # +2%
            "2024-01-03": {"open": 102, "high": 103, "low": 97, "close": 98},    # -3.9%
        })
        assert is_data_sane(df, max_single_day_move_pct=50.0) is True

    def test_rejects_corrupted_split_like_jump(self):
        df = _daily_df({
            "2024-01-01": {"open": 90, "high": 93, "low": 89, "close": 92},
            "2024-01-02": {"open": 92, "high": 300, "low": 91, "close": 271},   # ~194% jump — LBGJ-style anomaly
            "2024-01-03": {"open": 271, "high": 275, "low": 268, "close": 270},
        })
        assert is_data_sane(df, max_single_day_move_pct=50.0) is False

    def test_boundary_move_at_exactly_the_threshold_is_accepted(self):
        df = _daily_df({
            "2024-01-01": {"open": 100, "high": 101, "low": 99, "close": 100},
            "2024-01-02": {"open": 100, "high": 151, "low": 99, "close": 150},  # exactly +50%
        })
        assert is_data_sane(df, max_single_day_move_pct=50.0) is True

    def test_insufficient_data_defaults_to_sane(self):
        df = _daily_df({"2024-01-01": {"open": 100, "high": 101, "low": 99, "close": 100}})
        assert is_data_sane(df) is True
        assert is_data_sane(None) is True
