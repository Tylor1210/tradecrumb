"""Tests for the sell-half-at-+20% scale-out rule and its wiring into both backtest harnesses."""

import pandas as pd
import pytest

from signal_engine.backtest.engine import BacktestConfig, BacktestEngine, walk_exits, walk_orders
from signal_engine.risk.loss_cap import LossCapRule
from signal_engine.risk.scale_out import (
    EXIT_BREAKEVEN,
    EXIT_NO_RUNNER,
    EXIT_PARTIAL,
    EXIT_RUNNER_TIMEOUT,
    EXIT_SETUP_TURNED,
    EXIT_TRAIL,
    ScaleOutRule,
    ScaleOutStats,
    atr_series,
    bar_move_in_atr,
    gain_pct_from_entry,
)
from signal_engine.weekend_swing.backtest import (
    WeekendSwingTrade,
    _compute_metrics,
    _simulate_exit,
    _simulate_exit_detail,
    scale_out_stats,
)


def _slow(**kw):
    """A rule with the fast-move test switched off, so the plain scale-out path runs."""
    return ScaleOutRule(fast_move_bars=0, fast_move_atr_mult=0, **kw)


def _climb(state, gains, alive=True):
    """Feed several closes below the trigger; none of them should do anything."""
    for g in gains:
        assert state.update(g, setup_alive=alive) is None


# ── The rule itself ──────────────────────────────────────────────────────


class TestScaleOutRule:
    def test_below_trigger_does_nothing(self):
        state = _slow().new_trade()
        _climb(state, [5.0, 12.0, 19.9, -3.0], alive=False)
        assert not state.reached_trigger

    def test_sells_half_at_trigger_when_setup_continues(self):
        state = _slow().new_trade()
        _climb(state, [5.0, 12.0])
        action = state.update(20.0, setup_alive=True)
        assert action.reason == EXIT_PARTIAL
        assert action.sell_fraction == 0.5
        assert not action.closes_position
        assert state.partial_taken and state.managing

    def test_closes_everything_at_trigger_when_no_continuation(self):
        state = _slow().new_trade()
        _climb(state, [5.0, 12.0])
        action = state.update(21.0, setup_alive=False)
        assert action.reason == EXIT_NO_RUNNER
        assert action.closes_position
        assert state.reached_trigger and not state.partial_taken

    def test_runner_trailing_stop(self):
        state = _slow(trail_pct=8.0).new_trade()
        _climb(state, [5.0, 12.0])
        state.update(20.0, setup_alive=True)               # partial
        assert state.update(30.0, setup_alive=True) is None  # peak 130
        # 130 * 0.92 = 119.6 -> trail at +19.6%
        assert state.update(19.7, setup_alive=True) is None
        assert state.update(19.6, setup_alive=True).reason == EXIT_TRAIL

    def test_runner_never_goes_back_below_breakeven(self):
        # A very wide trail would let it fall far; the breakeven floor catches it first.
        state = _slow(trail_pct=50.0).new_trade()
        _climb(state, [5.0, 12.0])
        state.update(20.0, setup_alive=True)
        assert state.update(1.0, setup_alive=True) is None
        assert state.update(0.0, setup_alive=True).reason == EXIT_BREAKEVEN

    def test_runner_closes_when_setup_turns(self):
        state = _slow().new_trade()
        _climb(state, [5.0, 12.0])
        state.update(20.0, setup_alive=True)
        assert state.update(24.0, setup_alive=True) is None
        assert state.update(23.0, setup_alive=False).reason == EXIT_SETUP_TURNED

    def test_runner_max_bars(self):
        state = _slow(runner_max_bars=2).new_trade()
        _climb(state, [5.0, 12.0])
        state.update(20.0, setup_alive=True)
        assert state.update(22.0, setup_alive=True) is None
        assert state.update(23.0, setup_alive=True).reason == EXIT_RUNNER_TIMEOUT

    def test_fast_move_by_bars_keeps_full_position_on_trail(self):
        state = ScaleOutRule(fast_move_bars=2, fast_move_atr_mult=0).new_trade()
        assert state.update(8.0, setup_alive=True) is None
        assert state.update(25.0, setup_alive=True) is None  # bar 2: fast, nothing sold
        assert state.fast_move and state.managing and not state.partial_taken
        # setup turning does not close a fast-move position, only the trail does
        assert state.update(26.0, setup_alive=False) is None
        # peak 126 * 0.92 = 115.92 -> trail at +15.92%
        assert state.update(15.9, setup_alive=False).reason == EXIT_TRAIL

    def test_slow_climb_is_not_fast(self):
        state = ScaleOutRule(fast_move_bars=2, fast_move_atr_mult=0).new_trade()
        _climb(state, [5.0, 12.0])
        assert state.update(20.0, setup_alive=True).reason == EXIT_PARTIAL

    def test_fast_move_by_atr_thrust(self):
        state = ScaleOutRule(fast_move_bars=0, fast_move_atr_mult=3.0).new_trade()
        _climb(state, [2.0, 4.0, 6.0])
        assert state.update(21.0, setup_alive=True, bar_move_atr=5.0) is None
        assert state.fast_move

    def test_small_atr_move_is_not_fast(self):
        state = ScaleOutRule(fast_move_bars=0, fast_move_atr_mult=3.0).new_trade()
        _climb(state, [5.0, 12.0, 18.0])
        action = state.update(20.0, setup_alive=True, bar_move_atr=1.0)
        assert action.reason == EXIT_PARTIAL

    def test_config_defaults_and_disabled(self):
        assert ScaleOutRule.from_config(None) is None
        assert ScaleOutRule.from_config({"enabled": False}) is None
        rule = ScaleOutRule.from_config({"enabled": True})
        assert rule.trigger_gain_pct == 20.0
        assert rule.sell_fraction == 0.5
        assert rule.fast_move_bars == 2
        rule = ScaleOutRule.from_config_values({"enabled": False, "trigger_gain_pct": 15, "sell_fraction": 0.33})
        assert rule.trigger_gain_pct == 15.0 and rule.sell_fraction == 0.33

    def test_validation(self):
        with pytest.raises(ValueError):
            ScaleOutRule(sell_fraction=0)
        with pytest.raises(ValueError):
            ScaleOutRule(trigger_gain_pct=-1)
        with pytest.raises(ValueError):
            ScaleOutRule(trail_pct=0)
        with pytest.raises(ValueError):
            ScaleOutRule(runner_floor_pct=25.0)

    def test_helpers(self):
        assert gain_pct_from_entry("long", 100.0, 120.0) == pytest.approx(20.0)
        assert gain_pct_from_entry("short", 100.0, 80.0) == pytest.approx(20.0)
        assert bar_move_in_atr("long", 110.0, 100.0, 2.0) == pytest.approx(5.0)
        assert bar_move_in_atr("short", 90.0, 100.0, 2.0) == pytest.approx(5.0)
        assert bar_move_in_atr("long", 110.0, None, 2.0) is None
        df = pd.DataFrame({"high": [11.0] * 20, "low": [9.0] * 20, "close": [10.0] * 20})
        assert atr_series(df, 14).iloc[-1] == pytest.approx(2.0)
        assert atr_series(pd.DataFrame({"close": [1.0]})) is None

    def test_stats_reach_ladder(self):
        s = ScaleOutStats(trades=4, max_gain_pcts=[3.0, 11.0, 21.0, -2.0])
        assert s.reached_at_least(5) == 2
        assert s.reached_at_least(20) == 1


# ── Daily engine ─────────────────────────────────────────────────────────


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def _config(scale_out=None, loss_cap=None, hold=5, tp=10.0):
    return BacktestConfig(
        start_date="2024-01-01", end_date="2024-12-31", hold_period_days=hold,
        stop_loss_pct=5.0, take_profit_pct=tp, initial_capital=100_000.0,
        position_size_pct=5.0, loss_cap=loss_cap, scale_out=scale_out,
    )


CLIMB = [100, 105, 110, 115, 121, 126, 130, 128, 119, 118, 117, 116]


class TestWalkOrders:
    def test_partial_then_runner_past_hold_period(self):
        close = _series(CLIMB)
        entries = _series([1] + [0] * 11).astype(bool)
        alive = _series([1] * 12).astype(bool)

        ent, sells, reasons, stats = walk_orders(close, entries, alive, _config(_slow()))

        assert ent.iloc[0]
        assert sells.iloc[4] == 0.5                 # +21%: sell half (10% take profit switched off)
        assert pd.isna(sells.iloc[5])               # day 5 = hold period, runner ignores it
        assert sells.iloc[8] == 1.0                 # 119 <= 130 * 0.92: trailing stop
        assert reasons == [EXIT_PARTIAL, EXIT_TRAIL]
        assert stats.trades == 1 and stats.reached_trigger == 1 and stats.scaled_out == 1

    def test_take_profit_kept_when_not_replaced(self):
        close = _series(CLIMB)
        entries = _series([1] + [0] * 11).astype(bool)
        alive = _series([1] * 12).astype(bool)

        _, sells, reasons, stats = walk_orders(
            close, entries, alive, _config(_slow(replaces_take_profit=False))
        )
        assert reasons == ["take_profit"]
        assert sells.iloc[2] == 1.0
        assert stats.reached_trigger == 0

    def test_no_continuation_closes_all_at_trigger(self):
        close = _series(CLIMB)
        entries = _series([1] + [0] * 11).astype(bool)
        cont = _series([1, 1, 1, 1, 0] + [1] * 7).astype(bool)

        _, sells, reasons, stats = walk_orders(
            close, entries, cont, _config(_slow()), continuation=cont
        )
        assert sells.iloc[4] == 1.0
        assert reasons == [EXIT_NO_RUNNER]
        assert stats.closed_no_runner == 1

    def test_fast_move_skips_the_sale(self):
        close = _series([100, 112, 125, 131, 129, 118, 117])
        entries = _series([1, 0, 0, 0, 0, 0, 0]).astype(bool)
        alive = _series([1] * 7).astype(bool)

        _, sells, reasons, stats = walk_orders(
            close, entries, alive, _config(ScaleOutRule(fast_move_atr_mult=0), hold=3)
        )
        assert pd.isna(sells.iloc[2])               # +25% on bar 2: fast, kept whole
        assert sells.iloc[5] == 1.0                 # 118 <= 131 * 0.92 = 120.52
        assert reasons == [EXIT_TRAIL]
        assert stats.fast_move_held == 1

    def test_losing_trade_still_uses_loss_cap_and_time_exit(self):
        close = _series([100, 97, 89, 88, 87, 86, 85])
        entries = _series([1, 0, 0, 0, 0, 0, 0]).astype(bool)
        alive = _series([1, 1, 1, 1, 0, 0, 0]).astype(bool)

        _, sells, reasons, _ = walk_orders(
            close, entries, alive, _config(_slow(), loss_cap=LossCapRule(), hold=10)
        )
        assert reasons == ["loss_cap_review"]
        assert sells.iloc[4] == 1.0

    def test_walk_exits_unchanged_without_scale_out(self):
        close = _series([100, 101, 102, 103, 104, 105, 106])
        entries = _series([1, 0, 0, 0, 0, 0, 0]).astype(bool)
        alive = _series([1] * 7).astype(bool)
        _, ex, reasons = walk_exits(close, entries, alive, _config(None, loss_cap=LossCapRule(), hold=3))
        assert reasons == ["time_exit"]
        assert ex.iloc[3] and ex.sum() == 1


class TestEngineWithScaleOut:
    def test_partly_sold_trade_counts_once_with_blended_return(self):
        closes = _series(CLIMB)
        price_data = {"AAA": pd.DataFrame({"close": closes})}
        signals = {"AAA": _series([1] + [0] * 11).astype(bool)}
        scores = {"AAA": _series([0.6] * 12)}

        result = BacktestEngine(_config(_slow())).run(price_data, signals, scores)
        m = result.get_metrics()

        assert result.exit_reasons == {EXIT_PARTIAL: 1, EXIT_TRAIL: 1}
        assert m.total_trades == 1            # one position, not two exit fills
        assert m.winning_trades == 1
        # half sold at 121, half at 119 -> about +20% before fees
        assert m.worst_trade_pct == pytest.approx(19.8, abs=0.2)
        orders = result.portfolio.orders.records_readable
        assert len(orders) == 3
        assert orders["Size"].iloc[1] == pytest.approx(orders["Size"].iloc[2])
        assert result.scale_out_stats.scaled_out == 1

    def test_losing_runner_counts_as_one_trade(self):
        # +21% then the runner slides back to breakeven: the trade is still a winner overall
        closes = _series([100, 105, 110, 115, 121, 119, 112, 99])
        price_data = {"AAA": pd.DataFrame({"close": closes})}
        signals = {"AAA": _series([1, 0, 0, 0, 0, 0, 0, 0]).astype(bool)}
        scores = {"AAA": _series([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])}
        rule = _slow(trail_pct=20.0)

        result = BacktestEngine(_config(rule)).run(price_data, signals, scores)
        m = result.get_metrics()
        assert result.exit_reasons == {EXIT_PARTIAL: 1, EXIT_BREAKEVEN: 1}
        assert m.total_trades == 1 and m.win_rate_pct == 100.0

    def test_without_scale_out_engine_path_unchanged(self):
        closes = _series(CLIMB)
        price_data = {"AAA": pd.DataFrame({"close": closes})}
        signals = {"AAA": _series([1] + [0] * 11).astype(bool)}
        result = BacktestEngine(_config(None)).run(price_data, signals)
        assert result.scale_out_stats is None
        assert not result.count_by_position


# ── Weekend swing ────────────────────────────────────────────────────────

WEEK_START = pd.Timestamp("2024-01-08")
WEEK_END = pd.Timestamp("2024-01-12")
ENTRY_TIME = pd.Timestamp("2024-01-08 09:30")


def _daily(rows):
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(list(rows.keys()))
    return df


def _entry_bars():
    return pd.DataFrame([{"open": 100, "high": 101, "low": 99.5, "close": 100.5}], index=[ENTRY_TIME])


def _run(daily, rule, direction="long", stop=50.0):
    return _simulate_exit_detail(
        direction, 100.0, stop, ENTRY_TIME, _entry_bars(), daily, WEEK_START, WEEK_END,
        scale_out=rule,
    )


class TestWeekendScaleOut:
    rule = ScaleOutRule(fast_move_bars=1, fast_move_atr_mult=0)

    def test_partial_on_wednesday_then_friday_close_blended(self):
        daily = _daily({
            "2024-01-09": {"open": 104, "high": 110, "low": 103, "close": 109.0},  # +9%
            "2024-01-10": {"open": 110, "high": 122, "low": 109, "close": 121.0},  # +21% green: sell half
            "2024-01-11": {"open": 121, "high": 125, "low": 120, "close": 124.0},
            "2024-01-12": {"open": 120, "high": 126, "low": 120, "close": 123.0},  # Friday, green
        })
        r = _run(daily, self.rule)
        assert r.exit_reason == "friday_close"
        assert r.partial_price == 121.0 and r.partial_fraction == 0.5
        assert r.exit_price == pytest.approx(122.0)  # (121 + 123) / 2
        assert r.reached_scale_out and not r.fast_move
        assert r.max_gain_pct == pytest.approx(24.0)

    def test_three_tuple_wrapper_returns_blended_price(self):
        daily = _daily({
            "2024-01-09": {"open": 104, "high": 110, "low": 103, "close": 109.0},
            "2024-01-10": {"open": 110, "high": 122, "low": 109, "close": 121.0},
            "2024-01-12": {"open": 120, "high": 126, "low": 120, "close": 123.0},
        })
        price, reason, _ = _simulate_exit(
            "long", 100.0, 50.0, ENTRY_TIME, _entry_bars(), daily, WEEK_START, WEEK_END,
            scale_out=self.rule,
        )
        assert (price, reason) == (pytest.approx(122.0), "friday_close")

    def test_red_day_at_trigger_closes_all(self):
        daily = _daily({
            "2024-01-09": {"open": 104, "high": 110, "low": 103, "close": 109.0},
            "2024-01-10": {"open": 123, "high": 124, "low": 119, "close": 121.0},  # +21% but red
        })
        r = _run(daily, self.rule)
        assert (r.exit_price, r.exit_reason) == (121.0, EXIT_NO_RUNNER)
        assert r.partial_price is None

    def test_runner_closes_on_setup_turn(self):
        daily = _daily({
            "2024-01-09": {"open": 104, "high": 110, "low": 103, "close": 109.0},
            "2024-01-10": {"open": 110, "high": 122, "low": 109, "close": 121.0},  # sell half
            "2024-01-11": {"open": 122, "high": 122, "low": 117, "close": 118.0},  # red: runner out
            "2024-01-12": {"open": 118, "high": 130, "low": 118, "close": 130.0},
        })
        r = _run(daily, self.rule)
        assert r.exit_reason == EXIT_SETUP_TURNED
        assert r.exit_price == pytest.approx(119.5)

    def test_strategy_stop_after_partial_is_blended(self):
        daily = _daily({
            "2024-01-09": {"open": 104, "high": 110, "low": 103, "close": 109.0},
            "2024-01-10": {"open": 110, "high": 122, "low": 109, "close": 121.0},
            "2024-01-11": {"open": 121, "high": 121, "low": 90, "close": 95.0},
        })
        r = _run(daily, self.rule, stop=97.0)
        assert r.exit_reason == "stopped_out"
        assert r.exit_price == pytest.approx((121.0 + 97.0) / 2)

    def test_fast_tuesday_move_keeps_full_position(self):
        daily = _daily({
            "2024-01-09": {"open": 110, "high": 125, "low": 110, "close": 124.0},  # +24% on Tuesday
            "2024-01-10": {"open": 124, "high": 128, "low": 123, "close": 127.0},
            "2024-01-11": {"open": 127, "high": 127, "low": 115, "close": 116.0},  # 127*0.92=116.84
        })
        r = _run(daily, self.rule)
        assert (r.exit_price, r.exit_reason) == (116.0, EXIT_TRAIL)
        assert r.fast_move and r.partial_price is None

    def test_short_side(self):
        daily = _daily({
            "2024-01-09": {"open": 96, "high": 97, "low": 90, "close": 91.0},
            "2024-01-10": {"open": 90, "high": 90, "low": 78, "close": 79.0},   # +21% short, red day = sellers in
            "2024-01-12": {"open": 83, "high": 83, "low": 79, "close": 82.0},
        })
        r = _run(daily, self.rule, direction="short", stop=150.0)
        assert r.partial_price == 79.0
        assert r.exit_reason == "friday_close"
        assert r.exit_price == pytest.approx(80.5)

    def test_off_means_unchanged(self):
        daily = _daily({
            "2024-01-09": {"open": 104, "high": 110, "low": 103, "close": 109.0},
            "2024-01-10": {"open": 110, "high": 122, "low": 109, "close": 121.0},
            "2024-01-12": {"open": 124, "high": 126, "low": 120, "close": 123.0},
        })
        r = _run(daily, None)
        assert (r.exit_price, r.exit_reason) == (123.0, "friday_close")
        assert r.max_gain_pct is None and not r.reached_scale_out

    def test_metrics_and_reach_stats(self):
        def trade(ret, **kw):
            return WeekendSwingTrade(
                ticker="X", strategy="s", direction="long", entry_time=None, entry_price=100,
                stop_price=90, exit_time=None, exit_price=100 + ret, exit_reason="friday_close",
                return_pct=ret, **kw,
            )

        trades = [
            trade(22.0, partial_fraction=0.5, partial_exit_price=121.0, max_gain_pct=24.0, reached_scale_out=True),
            trade(16.0, max_gain_pct=27.0, reached_scale_out=True, scale_out_fast_move=True),
            trade(3.0, max_gain_pct=6.0),
            trade(-2.0, max_gain_pct=1.0),
            trade(-1.0),  # stopped out on Monday, no close seen
        ]
        m = _compute_metrics(trades, "s")
        assert m.win_rate_pct == 60.0
        assert m.reached_scale_out == 2 and m.scaled_out == 1
        assert m.total_return_pct == pytest.approx(38.0)

        stats = scale_out_stats(trades)
        assert stats.trades == 5
        assert stats.reached_trigger == 2 and stats.reached_pct == 40.0
        assert stats.scaled_out == 1 and stats.fast_move_held == 1
        assert stats.reached_at_least(5) == 3
