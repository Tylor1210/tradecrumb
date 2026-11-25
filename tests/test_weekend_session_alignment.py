"""
Real Alpaca bars don't look like the naive midnight-stamped fixtures elsewhere:

- the index is UTC,
- a daily bar is stamped at midnight New York time (05:00 UTC in winter,
  04:00 UTC in summer), so "Friday's bar" sits *after* Friday 00:00 UTC,
- 5/15-min bars include IEX pre-market and after-hours prints.

These tests build bars in exactly that shape and check the weekend-swing
levels and harness line up with the regular session (09:30-16:00 ET).
"""

import pandas as pd
import pytest

from signal_engine.weekend_swing.backtest import WeekendSwingBacktester, _simulate_exit
from signal_engine.weekend_swing.levels import (
    monday_session_bars,
    overnight_range_proxy,
    prior_week_range,
    session_open_time,
)

ET = "America/New_York"
MONDAY = pd.Timestamp("2024-01-08", tz="UTC")       # how the CLI passes week_start
FRIDAY = pd.Timestamp("2024-01-12", tz="UTC")
SUMMER_MONDAY = pd.Timestamp("2024-07-08", tz="UTC")


def alpaca_daily(start: str, end: str, price: float = 100.0, spread: float = 3.0) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    stamps = days.tz_localize(ET).tz_convert("UTC")  # midnight ET, like Alpaca
    return pd.DataFrame(
        {"open": price, "high": price + spread, "low": price - spread, "close": price, "volume": 1e6},
        index=stamps,
    )


def alpaca_intraday(day: str, freq: str = "5min", price: float = 100.03) -> pd.DataFrame:
    stamps = pd.date_range(f"{day} 08:00", f"{day} 16:55", freq=freq, tz=ET).tz_convert("UTC")
    return pd.DataFrame(
        {"open": price, "high": price + 0.01, "low": price - 0.02, "close": price, "volume": 1000.0},
        index=stamps,
    )


class _Store:
    def __init__(self, daily, intraday):
        self.daily, self.intraday = daily, intraday

    def get_bars(self, ticker, timeframe, start, end):
        df = self.daily if timeframe == "1Day" else self.intraday
        return df.loc[(df.index >= start) & (df.index <= end)]


class TestDailyBarDates:
    def test_overnight_proxy_uses_fridays_close_not_thursdays(self):
        daily = alpaca_daily("2023-11-01", "2024-01-12")
        daily.loc[daily.index[daily.index.tz_convert(ET).day == 4][-1], "close"] = 90.0  # Thu Jan 4
        daily.loc[pd.Timestamp("2024-01-05", tz=ET).tz_convert("UTC"), "close"] = 100.0  # Fri Jan 5
        daily.loc[pd.Timestamp("2024-01-08", tz=ET).tz_convert("UTC"), "open"] = 100.05  # Mon Jan 8

        proxy = overnight_range_proxy(daily, MONDAY)

        assert proxy["friday_close"] == 100.0
        assert proxy["monday_open"] == 100.05

    def test_prior_week_range_covers_monday_to_friday(self):
        daily = alpaca_daily("2023-12-01", "2024-01-12")
        daily.loc[pd.Timestamp("2024-01-05", tz=ET).tz_convert("UTC"), "high"] = 120.0  # prior Friday
        daily.loc[pd.Timestamp("2024-01-08", tz=ET).tz_convert("UTC"), "low"] = 50.0    # this Monday

        assert prior_week_range(daily, MONDAY) == (120.0, 97.0)


class TestMondaySession:
    @pytest.mark.parametrize("monday", [MONDAY, SUMMER_MONDAY])
    def test_keeps_only_regular_session_bars(self, monday):
        bars = alpaca_intraday(monday.strftime("%Y-%m-%d"))
        rth = monday_session_bars(bars, monday)

        et_times = rth.index.tz_convert(ET)
        assert et_times.min() == pd.Timestamp(f"{monday.date()} 09:30", tz=ET)
        assert et_times.max() == pd.Timestamp(f"{monday.date()} 15:55", tz=ET)
        assert session_open_time(monday, bars.index) == et_times.min()

    def test_naive_et_bars_still_work(self):
        idx = pd.date_range("2024-01-08 08:00", "2024-01-08 16:55", freq="5min")
        bars = pd.DataFrame({"close": 1.0}, index=idx)
        rth = monday_session_bars(bars, pd.Timestamp("2024-01-08"))
        assert rth.index.min() == pd.Timestamp("2024-01-08 09:30")
        assert rth.index.max() == pd.Timestamp("2024-01-08 15:55")


class TestHarnessUsesRegularSession:
    def test_strategy_2_ignores_pre_market_print(self):
        daily = alpaca_daily("2023-11-01", "2024-01-12")
        daily.loc[pd.Timestamp("2024-01-08", tz=ET).tz_convert("UTC"), "open"] = 100.05
        bars = alpaca_intraday("2024-01-08")
        pre = pd.Timestamp("2024-01-08 08:00", tz=ET).tz_convert("UTC")
        bars.loc[pre, ["high", "close"]] = 101.0  # thin pre-market print above the range
        open_bar = pd.Timestamp("2024-01-08 09:35", tz=ET).tz_convert("UTC")
        bars.loc[open_bar, ["high", "close"]] = 100.2  # the real breakout, inside the window

        trades = WeekendSwingBacktester({}, _Store(daily, bars)).collect_strategy_2_trades(["X"], MONDAY, FRIDAY)

        assert len(trades) == 1
        assert trades[0].entry_time == open_bar
        assert trades[0].entry_price == 100.2

    def test_strategy_3_first_bar_is_the_0930_bar(self):
        daily = alpaca_daily("2023-11-01", "2024-01-12")
        daily.loc[pd.Timestamp("2024-01-08", tz=ET).tz_convert("UTC"), "open"] = 110.0  # gap above prior week
        bars = alpaca_intraday("2024-01-08", price=110.0)

        go, fail = WeekendSwingBacktester({}, _Store(daily, bars)).collect_strategy_3_trades(["SPY"], MONDAY, FRIDAY)

        trades = go + fail
        assert len(trades) == 1
        assert trades[0].entry_time == pd.Timestamp("2024-01-08 09:30", tz=ET).tz_convert("UTC")


class TestSimulateExitWithAlpacaStamps:
    def _entry(self):
        entry_time = pd.Timestamp("2024-01-08 10:00", tz=ET).tz_convert("UTC")
        later = pd.Timestamp("2024-01-08 10:05", tz=ET).tz_convert("UTC")
        intraday = pd.DataFrame(
            {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2}, index=[entry_time, later]
        )
        return entry_time, intraday

    def test_mondays_pre_entry_low_does_not_stop_the_trade(self):
        entry_time, intraday = self._entry()
        daily = alpaca_daily("2024-01-08", "2024-01-12", price=101.0, spread=0.5)
        # Monday's daily bar includes the 09:30-10:00 dip that happened before entry.
        daily.loc[pd.Timestamp("2024-01-08", tz=ET).tz_convert("UTC"), "low"] = 95.0

        price, reason, _ = _simulate_exit("long", 100.0, 99.0, entry_time, intraday, daily, MONDAY, FRIDAY)

        assert reason == "friday_close"
        assert price == 101.0

    def test_exits_on_fridays_close_not_thursdays(self):
        entry_time, intraday = self._entry()
        daily = alpaca_daily("2024-01-08", "2024-01-12", price=101.0, spread=0.5)
        friday = pd.Timestamp("2024-01-12", tz=ET).tz_convert("UTC")
        daily.loc[friday, "close"] = 104.0

        price, reason, when = _simulate_exit("long", 100.0, 99.0, entry_time, intraday, daily, MONDAY, FRIDAY)

        assert (price, reason, when) == (104.0, "friday_close", friday)
