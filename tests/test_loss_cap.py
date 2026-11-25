"""Tests for the 10%/20% per-trade loss cap and its wiring into both backtest harnesses."""

import numpy as np
import pandas as pd
import pytest

from signal_engine.backtest.engine import BacktestConfig, BacktestEngine, walk_exits
from signal_engine.risk.loss_cap import (
    EXIT_BACKSTOP,
    EXIT_REVIEW,
    EXIT_TIMEOUT,
    EXIT_WEAK,
    LossCapRule,
    loss_pct_from_entry,
)
from signal_engine.weekend_swing.backtest import _compute_metrics, _simulate_exit


# ── The rule itself ──────────────────────────────────────────────────────


class TestLossCapRule:
    def test_under_review_level_never_exits(self):
        state = LossCapRule().new_trade()
        assert state.update(9.9, setup_alive=False) is None
        assert state.update(-5.0, setup_alive=False) is None

    def test_review_zone_holds_while_setup_alive(self):
        state = LossCapRule().new_trade()
        assert state.update(12.0, setup_alive=True) is None
        assert state.update(15.0, setup_alive=True) is None

    def test_review_zone_closes_when_setup_goes_weak(self):
        state = LossCapRule().new_trade()
        assert state.update(12.0, setup_alive=True) is None
        assert state.update(13.0, setup_alive=False) == EXIT_REVIEW

    def test_closing_zone_exits_immediately_on_weak_bar(self):
        state = LossCapRule().new_trade()
        assert state.update(21.0, setup_alive=False) == EXIT_WEAK

    def test_closing_zone_times_out_after_n_bars(self):
        state = LossCapRule(exit_within_bars=2).new_trade()
        assert state.update(20.0, setup_alive=True) is None   # enters closing zone
        assert state.update(19.0, setup_alive=True) is None   # bar 1
        assert state.update(18.0, setup_alive=True) == EXIT_TIMEOUT  # bar 2

    def test_closing_zone_stays_latched_after_a_bounce(self):
        state = LossCapRule(exit_within_bars=5).new_trade()
        assert state.update(22.0, setup_alive=True) is None
        # price bounces back to only 3% down; still working toward the exit
        assert state.update(3.0, setup_alive=False) == EXIT_WEAK

    def test_backstop_closes_no_matter_what(self):
        state = LossCapRule().new_trade()
        assert state.update(25.0, setup_alive=True) == EXIT_BACKSTOP

    def test_exit_within_zero_bars_closes_on_touch(self):
        state = LossCapRule(exit_within_bars=0).new_trade()
        assert state.update(20.0, setup_alive=True) == EXIT_TIMEOUT

    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError):
            LossCapRule(review_loss_pct=20.0, exit_loss_pct=10.0)
        with pytest.raises(ValueError):
            LossCapRule(exit_loss_pct=30.0, backstop_loss_pct=25.0)

    def test_from_config(self):
        assert LossCapRule.from_config(None) is None
        assert LossCapRule.from_config({"enabled": False}) is None
        rule = LossCapRule.from_config({"enabled": True, "exit_within_bars": 3, "replaces_fixed_stop": False})
        assert rule.review_loss_pct == 10.0
        assert rule.exit_loss_pct == 20.0
        assert rule.exit_within_bars == 3
        assert rule.replaces_fixed_stop is False

    def test_loss_pct_from_entry(self):
        assert loss_pct_from_entry("long", 100.0, 88.0) == pytest.approx(12.0)
        assert loss_pct_from_entry("long", 100.0, 110.0) == pytest.approx(-10.0)
        assert loss_pct_from_entry("short", 100.0, 112.0) == pytest.approx(12.0)


# ── Daily engine ─────────────────────────────────────────────────────────


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def _config(rule=None, hold=10, sl=5.0, tp=10.0):
    return BacktestConfig(
        start_date="2024-01-01", end_date="2024-12-31", hold_period_days=hold,
        stop_loss_pct=sl, take_profit_pct=tp, initial_capital=100_000.0,
        position_size_pct=5.0, loss_cap=rule,
    )


class TestWalkExits:
    def test_holds_through_10pct_while_setup_alive_then_exits_on_weak_bar(self):
        close = _series([100, 95, 89, 88, 87, 86])
        entries = _series([1, 0, 0, 0, 0, 0]).astype(bool)
        alive = _series([1, 1, 1, 1, 0, 0]).astype(bool)

        ent, ex, reasons = walk_exits(close, entries, alive, _config(LossCapRule()))

        assert ent.iloc[0]
        assert not ex.iloc[1]          # 5% down: old flat stop would have fired here
        assert not ex.iloc[2]          # 11% down, setup alive: hold
        assert ex.iloc[4]              # 13% down, setup weak: out
        assert reasons == [EXIT_REVIEW]

    def test_flat_stop_still_used_when_rule_does_not_replace_it(self):
        close = _series([100, 94, 90])
        entries = _series([1, 0, 0]).astype(bool)
        alive = _series([1, 1, 1]).astype(bool)

        _, ex, reasons = walk_exits(close, entries, alive, _config(LossCapRule(replaces_fixed_stop=False)))

        assert ex.iloc[1]
        assert reasons == ["stop_loss"]

    def test_take_profit_and_time_exit(self):
        close = _series([100, 104, 111, 100, 101, 102, 103])
        entries = _series([1, 0, 0, 1, 0, 0, 0]).astype(bool)
        alive = _series([1] * 7).astype(bool)

        _, ex, reasons = walk_exits(close, entries, alive, _config(LossCapRule(), hold=2))

        assert ex.iloc[2] and ex.iloc[5]
        assert reasons == ["take_profit", "time_exit"]

    def test_closing_zone_works_toward_exit(self):
        close = _series([100, 85, 79, 80, 81, 82])
        entries = _series([1, 0, 0, 0, 0, 0]).astype(bool)
        alive = _series([1] * 6).astype(bool)

        _, ex, reasons = walk_exits(close, entries, alive, _config(LossCapRule(exit_within_bars=2)))

        assert ex.iloc[4]  # hit 21% on bar 2, gets 2 more bars, closes on bar 4
        assert reasons == [EXIT_TIMEOUT]

    def test_ignores_signals_while_in_a_position(self):
        close = _series([100, 101, 102, 103, 104])
        entries = _series([1, 1, 1, 0, 0]).astype(bool)
        alive = _series([1] * 5).astype(bool)

        ent, _, _ = walk_exits(close, entries, alive, _config(LossCapRule(), hold=10))

        assert ent.tolist() == [True, False, False, False, False]


class TestEngineWithLossCap:
    def _price_data(self, closes):
        df = pd.DataFrame({"close": closes})
        return {"AAA": df}

    def test_engine_runs_and_reports_worst_trade(self):
        closes = _series([100, 95, 89, 88, 87, 86, 90, 91, 92, 93])
        signals = {"AAA": _series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(bool)}
        scores = {"AAA": _series([0.6, 0.2, 0.1, 0.05, -0.3, -0.3, 0, 0, 0, 0])}

        result = BacktestEngine(_config(LossCapRule())).run(self._price_data(closes), signals, scores)
        m = result.get_metrics()

        assert result.exit_reasons == {EXIT_REVIEW: 1}
        assert m.total_trades == 1
        assert m.winning_trades == 0
        assert m.worst_trade_pct == pytest.approx(-13.17, abs=0.05)  # 100 -> 87 plus fees
        assert m.avg_loss_pct == m.worst_trade_pct

    def test_without_scores_falls_back_to_green_bar(self):
        closes = _series([100, 95, 89, 88, 86, 85])
        opens = _series([100, 96, 87, 89, 87, 86])  # bar 2 green (buyers in), bar 3 red
        price_data = {"AAA": pd.DataFrame({"open": opens, "close": closes})}
        signals = {"AAA": _series([1, 0, 0, 0, 0, 0]).astype(bool)}

        result = BacktestEngine(_config(LossCapRule())).run(price_data, signals)

        assert result.exit_reasons == {EXIT_REVIEW: 1}
        trades = result.portfolio.trades.records_readable
        assert trades["Exit Timestamp"].iloc[0] == closes.index[3]

    def test_without_scores_or_opens_uses_prior_close(self):
        closes = _series([100, 95, 89, 90, 88, 87])
        signals = {"AAA": _series([1, 0, 0, 0, 0, 0]).astype(bool)}

        result = BacktestEngine(_config(LossCapRule())).run(self._price_data(closes), signals)

        assert result.exit_reasons == {EXIT_REVIEW: 1}
        trades = result.portfolio.trades.records_readable
        assert trades["Exit Timestamp"].iloc[0] == closes.index[2]

    def test_old_path_unchanged_and_fills_new_metric_fields(self):
        closes = _series([100, 101, 94, 90, 95, 96, 97, 98, 99, 100])
        signals = {"AAA": _series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(bool)}

        result = BacktestEngine(_config(None, hold=5)).run(self._price_data(closes), signals)
        m = result.get_metrics()

        assert result.exit_reasons == {}
        assert m.total_trades == 1
        assert m.worst_trade_pct < -5.0
        assert "worst_trade_pct" in m.to_dict()


# ── Weekend swing ────────────────────────────────────────────────────────

WEEK_START = pd.Timestamp("2024-01-08")
WEEK_END = pd.Timestamp("2024-01-12")


def _daily(rows):
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(list(rows.keys()))
    return df


class TestWeekendLossCap:
    entry_time = pd.Timestamp("2024-01-08 09:30")

    def _entry_bars(self):
        return pd.DataFrame([{"open": 100, "high": 101, "low": 99.5, "close": 100.5}], index=[self.entry_time])

    def _run(self, daily, loss_cap, stop=50.0):
        return _simulate_exit(
            "long", entry_price=100.0, stop_price=stop, entry_time=self.entry_time,
            entry_day_intraday_df=self._entry_bars(), daily_df=daily,
            week_start=WEEK_START, week_end=WEEK_END, loss_cap=loss_cap,
        )

    def test_wide_stop_review_zone_exit_on_red_day(self):
        daily = _daily({
            "2024-01-09": {"open": 99, "high": 99, "low": 88, "close": 88.0},   # 12% down, red day
            "2024-01-12": {"open": 90, "high": 95, "low": 90, "close": 95.0},
        })
        exit_price, reason, exit_time = self._run(daily, LossCapRule())
        assert (exit_price, reason, exit_time) == (88.0, EXIT_REVIEW, pd.Timestamp("2024-01-09"))

    def test_wide_stop_holds_on_green_day_in_review_zone(self):
        daily = _daily({
            "2024-01-09": {"open": 85, "high": 89, "low": 84, "close": 88.0},   # 12% down, but green
            "2024-01-12": {"open": 90, "high": 95, "low": 90, "close": 95.0},
        })
        exit_price, reason, _ = self._run(daily, LossCapRule())
        assert (exit_price, reason) == (95.0, "friday_close")

    def test_closing_zone_green_days_then_timeout(self):
        daily = _daily({
            "2024-01-09": {"open": 75, "high": 79, "low": 74, "close": 78.0},   # 22% down, green: start closing
            "2024-01-10": {"open": 78, "high": 80, "low": 77, "close": 79.0},   # bar 1
            "2024-01-11": {"open": 79, "high": 81, "low": 78, "close": 80.0},   # bar 2: out
            "2024-01-12": {"open": 80, "high": 90, "low": 80, "close": 90.0},
        })
        exit_price, reason, exit_time = self._run(daily, LossCapRule(exit_within_bars=2))
        assert (exit_price, reason, exit_time) == (80.0, EXIT_TIMEOUT, pd.Timestamp("2024-01-11"))

    def test_no_loss_cap_is_unchanged(self):
        daily = _daily({
            "2024-01-09": {"open": 99, "high": 99, "low": 88, "close": 88.0},
            "2024-01-12": {"open": 90, "high": 95, "low": 90, "close": 95.0},
        })
        exit_price, reason, _ = self._run(daily, None)
        assert (exit_price, reason) == (95.0, "friday_close")

    def test_strategy_stop_still_fires_first(self):
        daily = _daily({
            "2024-01-09": {"open": 99, "high": 99, "low": 80, "close": 85.0},
        })
        exit_price, reason, _ = self._run(daily, LossCapRule(), stop=97.0)
        assert (exit_price, reason) == (97.0, "stopped_out")

    def test_metrics_count_cap_exits_and_worst_trade(self):
        from signal_engine.weekend_swing.backtest import WeekendSwingTrade

        def trade(ret, reason):
            return WeekendSwingTrade(
                ticker="X", strategy="s", direction="long", entry_time=None, entry_price=100,
                stop_price=90, exit_time=None, exit_price=100 + ret, exit_reason=reason, return_pct=ret,
            )

        m = _compute_metrics([trade(2.0, "friday_close"), trade(-12.0, EXIT_REVIEW), trade(-1.0, "stopped_out")], "s")
        assert m.worst_trade_pct == -12.0
        assert m.loss_cap_exits == 1
