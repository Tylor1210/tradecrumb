"""Tests for weekend_swing.backtest — trade simulation, metrics, and full harness wiring."""

import pandas as pd
import pytest

from signal_engine.weekend_swing.backtest import (
    WeekendSwingBacktester,
    WeekendSwingTrade,
    _compute_metrics,
    _mondays_between,
    _return_pct,
    _simulate_exit,
)
from signal_engine.weekend_swing.data_store import IntradayBarStore

WEEK_START = pd.Timestamp("2024-01-08")  # Monday
WEEK_END = pd.Timestamp("2024-01-12")    # Friday


def _daily_bars(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(list(rows.keys()))
    return df


class TestReturnPct:
    def test_long_profit(self):
        assert _return_pct("long", 100.0, 110.0) == pytest.approx(10.0)

    def test_long_loss(self):
        assert _return_pct("long", 100.0, 90.0) == pytest.approx(-10.0)

    def test_short_profit_on_price_drop(self):
        assert _return_pct("short", 100.0, 90.0) == pytest.approx(10.0)

    def test_short_loss_on_price_rise(self):
        assert _return_pct("short", 100.0, 110.0) == pytest.approx(-10.0)


class TestMondaysBetween:
    def test_finds_all_mondays_in_range(self):
        mondays = _mondays_between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))
        assert pd.Timestamp("2024-01-01") in mondays
        assert pd.Timestamp("2024-01-08") in mondays
        assert pd.Timestamp("2024-01-15") in mondays
        assert pd.Timestamp("2024-01-22") in mondays
        assert pd.Timestamp("2024-01-29") in mondays
        assert all(d.weekday() == 0 for d in mondays)


class TestSimulateExit:
    def test_stop_hit_same_day_intraday(self):
        entry_time = pd.Timestamp("2024-01-08 09:30")
        entry_day_bars = pd.DataFrame(
            [
                {"open": 100, "high": 101, "low": 99.5, "close": 100.5},   # entry bar
                {"open": 100.5, "high": 100.6, "low": 94.0, "close": 95.0},  # stop hit here
            ],
            index=[entry_time, pd.Timestamp("2024-01-08 09:45")],
        )
        daily_df = _daily_bars({
            "2024-01-08": {"open": 100, "high": 101, "low": 94.0, "close": 95.0},
            "2024-01-09": {"open": 95, "high": 96, "low": 93, "close": 94},
        })

        exit_price, exit_reason, exit_time = _simulate_exit(
            "long", entry_price=100.0, stop_price=95.0, entry_time=entry_time,
            entry_day_intraday_df=entry_day_bars, daily_df=daily_df,
            week_start=WEEK_START, week_end=WEEK_END,
        )

        assert exit_price == 95.0
        assert exit_reason == "stopped_out"
        assert exit_time == pd.Timestamp("2024-01-08 09:45")

    def test_stop_hit_later_in_week(self):
        entry_time = pd.Timestamp("2024-01-08 09:30")
        entry_day_bars = pd.DataFrame(
            [{"open": 100, "high": 101, "low": 99.5, "close": 100.5}], index=[entry_time],
        )
        daily_df = _daily_bars({
            "2024-01-08": {"open": 100, "high": 101, "low": 99, "close": 100.5},
            "2024-01-09": {"open": 100.5, "high": 101, "low": 98, "close": 99},
            "2024-01-10": {"open": 99, "high": 99.5, "low": 90, "close": 91},  # stop hit here
            "2024-01-11": {"open": 91, "high": 92, "low": 89, "close": 90},
        })

        exit_price, exit_reason, exit_time = _simulate_exit(
            "long", entry_price=100.0, stop_price=95.0, entry_time=entry_time,
            entry_day_intraday_df=entry_day_bars, daily_df=daily_df,
            week_start=WEEK_START, week_end=WEEK_END,
        )

        assert exit_price == 95.0
        assert exit_reason == "stopped_out"
        assert exit_time == pd.Timestamp("2024-01-10")

    def test_exits_at_friday_close_when_stop_never_hit(self):
        entry_time = pd.Timestamp("2024-01-08 09:30")
        entry_day_bars = pd.DataFrame(
            [{"open": 100, "high": 101, "low": 99.5, "close": 100.5}], index=[entry_time],
        )
        daily_df = _daily_bars({
            "2024-01-08": {"open": 100, "high": 102, "low": 99, "close": 101},
            "2024-01-09": {"open": 101, "high": 103, "low": 100, "close": 102},
            "2024-01-10": {"open": 102, "high": 104, "low": 101, "close": 103},
            "2024-01-11": {"open": 103, "high": 105, "low": 102, "close": 104},
            "2024-01-12": {"open": 104, "high": 106, "low": 103, "close": 105},  # Friday close
        })

        exit_price, exit_reason, exit_time = _simulate_exit(
            "long", entry_price=100.0, stop_price=90.0, entry_time=entry_time,
            entry_day_intraday_df=entry_day_bars, daily_df=daily_df,
            week_start=WEEK_START, week_end=WEEK_END,
        )

        assert exit_price == 105.0
        assert exit_reason == "friday_close"
        assert exit_time == pd.Timestamp("2024-01-12")

    def test_short_stop_hit_on_price_rise(self):
        entry_time = pd.Timestamp("2024-01-08 09:30")
        entry_day_bars = pd.DataFrame(
            [{"open": 100, "high": 100.5, "low": 99, "close": 99.5}], index=[entry_time],
        )
        daily_df = _daily_bars({
            "2024-01-08": {"open": 100, "high": 101, "low": 99, "close": 99.5},
            "2024-01-09": {"open": 99.5, "high": 108, "low": 99, "close": 107},  # stop hit (high >= 105)
        })

        exit_price, exit_reason, exit_time = _simulate_exit(
            "short", entry_price=100.0, stop_price=105.0, entry_time=entry_time,
            entry_day_intraday_df=entry_day_bars, daily_df=daily_df,
            week_start=WEEK_START, week_end=WEEK_END,
        )

        assert exit_price == 105.0
        assert exit_reason == "stopped_out"


class TestComputeMetrics:
    def test_empty_trades_returns_zeroed_metrics(self):
        metrics = _compute_metrics([], "prior_week_breakout")
        assert metrics.trade_count == 0
        assert metrics.win_rate_pct == 0.0

    def test_win_rate_and_avg_win_loss(self):
        trades = [
            WeekendSwingTrade(
                ticker="AAPL", strategy="prior_week_breakout", direction="long",
                entry_time=pd.Timestamp("2024-01-08"), entry_price=100, stop_price=95,
                exit_time=pd.Timestamp("2024-01-12"), exit_price=110, exit_reason="friday_close",
                return_pct=10.0,
            ),
            WeekendSwingTrade(
                ticker="MSFT", strategy="prior_week_breakout", direction="long",
                entry_time=pd.Timestamp("2024-01-08"), entry_price=100, stop_price=95,
                exit_time=pd.Timestamp("2024-01-09"), exit_price=95, exit_reason="stopped_out",
                return_pct=-5.0,
            ),
            WeekendSwingTrade(
                ticker="NVDA", strategy="prior_week_breakout", direction="long",
                entry_time=pd.Timestamp("2024-01-08"), entry_price=100, stop_price=95,
                exit_time=pd.Timestamp("2024-01-12"), exit_price=120, exit_reason="friday_close",
                return_pct=20.0,
            ),
        ]

        metrics = _compute_metrics(trades, "prior_week_breakout")

        assert metrics.trade_count == 3
        assert metrics.win_rate_pct == pytest.approx(66.67, abs=0.01)
        assert metrics.avg_win_pct == pytest.approx(15.0)
        assert metrics.avg_loss_pct == pytest.approx(-5.0)
        assert metrics.max_drawdown_pct >= 0.0

    def test_gap_type_is_carried_through(self):
        metrics = _compute_metrics([], "gap_strength", gap_type="gap_and_go")
        assert metrics.gap_type == "gap_and_go"


class FakeAlpacaClient:
    """Generates a deterministic breakout pattern so the full harness can be exercised end-to-end."""

    def get_historical_bars(self, symbols, start, end, timeframe_str="1Day"):
        ticker = symbols[0]
        if timeframe_str == "1Day":
            dates = pd.bdate_range(start.normalize(), end.normalize())
            rows, price = [], 95.0
            for d in dates:
                price += 0.1
                rows.append({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1e6})
            df = pd.DataFrame(rows, index=dates)
            # Force a known prior week (Jan 1-5) range for the Jan 8 test week
            if pd.Timestamp("2024-01-05") in df.index:
                df.loc["2024-01-01":"2024-01-05", "high"] = [102, 103, 104, 105, 106]
                df.loc["2024-01-01":"2024-01-05", "low"] = [99, 100, 101, 102, 103]
            return {ticker: df}

        # Intraday: flat baseline volume everywhere, except a breakout candle
        # at 2024-01-08 09:30 (above week_high=106, with a volume surge).
        idx = pd.date_range(start, end, freq="15min" if "15" in timeframe_str else "5min")
        idx = idx[(idx.hour >= 9) & (idx.hour < 16)]
        rows = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000.0} for _ in idx]
        df = pd.DataFrame(rows, index=idx)
        breakout_ts = pd.Timestamp("2024-01-08 09:30")
        if breakout_ts in df.index:
            df.loc[breakout_ts] = {"open": 106.2, "high": 108.0, "low": 106.0, "close": 107.5, "volume": 5000.0}
        return {ticker: df}


class TestFullHarnessWiring:
    def test_run_strategy_1_produces_a_trade_and_metrics(self, tmp_path):
        bar_store = IntradayBarStore(FakeAlpacaClient(), store_dir=tmp_path)
        backtester = WeekendSwingBacktester(
            config={"strategy_1_breakout": {"volume_filter_multiplier": 1.2, "atr_period": 14, "atr_stop_multiplier": 0.3}},
            bar_store=bar_store,
            finnhub_client=None,
        )

        metrics = backtester.run_strategy_1(["AAPL"], pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-12"))

        assert metrics.strategy == "prior_week_breakout"
        assert metrics.trade_count >= 1

    def test_run_strategy_3_splits_go_and_fail(self, tmp_path):
        bar_store = IntradayBarStore(FakeAlpacaClient(), store_dir=tmp_path)
        backtester = WeekendSwingBacktester(
            config={"strategy_3_gap": {"restricted_tickers": ["SPY", "QQQ"]}},
            bar_store=bar_store,
            finnhub_client=None,
        )

        go_metrics, fail_metrics = backtester.run_strategy_3(
            ["SPY"], pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-12")
        )

        assert go_metrics.gap_type == "gap_and_go"
        assert fail_metrics.gap_type == "gap_and_fail"

    def test_run_strategy_3_ignores_non_restricted_tickers(self, tmp_path):
        bar_store = IntradayBarStore(FakeAlpacaClient(), store_dir=tmp_path)
        backtester = WeekendSwingBacktester(
            config={"strategy_3_gap": {"restricted_tickers": ["SPY", "QQQ"]}},
            bar_store=bar_store,
            finnhub_client=None,
        )

        go_metrics, fail_metrics = backtester.run_strategy_3(
            ["AAPL"], pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-12")
        )

        assert go_metrics.trade_count == 0
        assert fail_metrics.trade_count == 0
