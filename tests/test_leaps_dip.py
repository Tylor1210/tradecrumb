"""Tests for leaps_dip: entry rules, point-in-time revenue check, trades, account replay, CLI wiring."""

import numpy as np
import pandas as pd
import pytest

from signal_engine.leaps_dip.data import load_daily_bars, universe_from_config
from signal_engine.leaps_dip.fundamentals import (
    parse_financials_reported,
    revenue_growing_as_of,
)
from signal_engine.leaps_dip.rules import (
    DipRules,
    add_indicators,
    bullish_pin_bar,
    divergence_add,
    drawdown_from_high,
    wilder_rsi,
)
from signal_engine.leaps_dip.simulate import (
    OptionAssumptions,
    Trade,
    TradeRules,
    build_trade,
    buy_and_hold,
    non_overlapping,
    prepare_ticker,
    replay_account,
    run_backtest,
    trade_stats,
)

RULES = DipRules(min_history_bars=50, divergence_min_bars=3)


def _bars(closes, opens=None, highs=None, lows=None, start="2020-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens if opens is not None else closes, dtype=float)
    highs = np.asarray(highs if highs is not None else np.maximum(opens, closes) * 1.005, dtype=float)
    lows = np.asarray(lows if lows is not None else np.minimum(opens, closes) * 0.995, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes), tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6}, index=idx)


def _crash_then_pin(rebound_to: float = 120.0, rebound_bars: int = 120):
    """
    Up to 200, crash to about 80 (down 60%), a bullish pin bar on the low,
    then a steady rebound. Returns (df, pin_index).
    """
    up = np.linspace(100, 200, 80)
    down = np.linspace(198, 82, 60)
    closes = list(up) + list(down)
    opens = list(closes)
    highs = [c * 1.005 for c in closes]
    lows = [c * 0.995 for c in closes]
    # Pin bar: opens 80.5, dips to 74, closes 81 near the top
    pin_i = len(closes)
    closes.append(81.0)
    opens.append(80.5)
    highs.append(81.3)
    lows.append(74.0)
    rebound = np.linspace(82, rebound_to, rebound_bars)
    for c in rebound:
        opens.append(c * 0.999)
        closes.append(c)
        highs.append(c * 1.005)
        lows.append(c * 0.995)
    return _bars(closes, opens, highs, lows), pin_i


# ─────────────────────────────────────────────────────────────
# RULES
# ─────────────────────────────────────────────────────────────

class TestRules:
    def test_rsi_low_in_a_selloff_and_high_in_a_rally(self):
        down = wilder_rsi(pd.Series(np.linspace(100, 50, 40)))
        up = wilder_rsi(pd.Series(np.linspace(50, 100, 40)))
        assert down.iloc[-1] < 5
        assert up.iloc[-1] > 95

    def test_pin_bar_shape(self):
        df = pd.DataFrame({
            "open": [80.5, 80.0, 80.0],
            "high": [81.3, 85.0, 81.0],
            "low": [74.0, 79.0, 74.0],
            "close": [81.0, 84.0, 75.0],
        })
        # Long lower wick, small body at the top / big body / close at the low
        assert bullish_pin_bar(df, RULES).tolist() == [True, False, False]

    def test_drawdown_waits_for_min_history(self):
        df = _bars(np.linspace(100, 40, 60))
        dd = drawdown_from_high(df, RULES)
        assert dd.iloc[: RULES.min_history_bars - 1].isna().all()
        assert dd.iloc[-1] < -0.55

    def test_signal_on_pin_bar_after_crash(self):
        df, pin_i = _crash_then_pin()
        ind = add_indicators(df, RULES)
        assert ind["drawdown"].iloc[pin_i] <= -0.5
        assert ind["ref_rsi"].iloc[pin_i] < 30
        assert bool(ind["signal"].iloc[pin_i])
        assert not ind["signal"].iloc[:pin_i].any()

    def test_no_signal_when_drawdown_too_small(self):
        df, pin_i = _crash_then_pin()
        ind = add_indicators(df, DipRules(min_history_bars=50, drawdown_min=0.7))
        assert not ind["signal"].any()

    def test_divergence_needs_lower_low_higher_rsi_and_green_bar(self):
        ind = pd.DataFrame({
            "open": [70.0, 72.0], "close": [71.0, 71.5], "low": [69.0, 69.0], "rsi": [35.0, 35.0],
        })
        assert divergence_add(ind, 0, ref_low=74.0, ref_rsi=20.0, rules=RULES)
        assert not divergence_add(ind, 1, ref_low=74.0, ref_rsi=20.0, rules=RULES)   # red bar
        assert not divergence_add(ind, 0, ref_low=68.0, ref_rsi=20.0, rules=RULES)   # no lower low
        assert not divergence_add(ind, 0, ref_low=74.0, ref_rsi=40.0, rules=RULES)   # RSI lower too


# ─────────────────────────────────────────────────────────────
# REVENUE CHECK
# ─────────────────────────────────────────────────────────────

def _filing(year, filed, revenue, concept="us-gaap_Revenues"):
    return {"year": year, "filedDate": filed, "report": {"ic": [
        {"concept": "us-gaap_NetIncomeLoss", "value": 5.0},
        {"concept": concept, "value": revenue},
    ]}}


class TestRevenue:
    def test_parse_picks_revenue_and_first_filing_per_year(self):
        payload = {"data": [
            _filing(2019, "2020-02-15 00:00:00", 100.0),
            _filing(2020, "2021-02-15 00:00:00", 120.0, "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"),
            _filing(2020, "2021-06-01 00:00:00", 999.0),  # amendment, later
            {"year": 2018, "filedDate": "2019-02-01", "report": {"ic": [{"concept": "x_Other", "value": 1}]}},
        ]}
        df = parse_financials_reported(payload)
        assert df["fiscal_year"].tolist() == [2019, 2020]
        assert df["revenue"].tolist() == [100.0, 120.0]

    def test_only_uses_filings_known_on_the_day(self):
        hist = parse_financials_reported({"data": [
            _filing(2019, "2020-02-15", 100.0),
            _filing(2020, "2021-02-15", 120.0),
            _filing(2021, "2022-02-15", 90.0),
        ]})
        assert revenue_growing_as_of(hist, pd.Timestamp("2020-06-01")) is None   # one year known
        assert revenue_growing_as_of(hist, pd.Timestamp("2021-06-01")) is True
        assert revenue_growing_as_of(hist, pd.Timestamp("2022-02-14")) is True   # drop not filed yet
        assert revenue_growing_as_of(hist, pd.Timestamp("2022-02-15", tz="UTC")) is False

    def test_empty_history_is_unknown(self):
        assert revenue_growing_as_of(parse_financials_reported({}), pd.Timestamp("2022-01-01")) is None


# ─────────────────────────────────────────────────────────────
# TRADES
# ─────────────────────────────────────────────────────────────

class TestTrades:
    def test_enters_next_open_and_takes_profit(self):
        df, pin_i = _crash_then_pin(rebound_to=150)
        ind = prepare_ticker(df, RULES)
        tr = TradeRules(hold_months=12, profit_take=0.3, entry_fraction=1.0)
        t = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1])
        assert t.entry_date == ind.index[pin_i + 1]
        assert t.entry_stock_price == pytest.approx(ind["open"].iloc[pin_i + 1])
        assert t.exit_reason == "profit"
        assert t.return_pct >= 30
        assert t.return_pct == pytest.approx(t.slot_return_pct)  # whole slot invested

    def test_time_exit_after_hold(self):
        df, pin_i = _crash_then_pin(rebound_to=90, rebound_bars=200)
        ind = prepare_ticker(df, RULES)
        tr = TradeRules(hold_months=6, profit_take=None)
        t = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1])
        assert t.exit_reason == "time"
        assert t.bars_held == tr.hold_bars

    def test_unused_add_cash_dilutes_slot_return(self):
        df, pin_i = _crash_then_pin(rebound_to=150)
        ind = prepare_ticker(df, RULES)
        tr = TradeRules(hold_months=12, profit_take=0.3, entry_fraction=0.5)
        t = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1])
        assert t.add_date is None
        assert t.slot_return_pct == pytest.approx(t.return_pct * 0.5)

    def test_divergence_add_fills_next_open(self):
        df, pin_i = _crash_then_pin(rebound_to=150)
        # Dip under the pin bar's low a few bars later on a green, less oversold bar
        j = pin_i + 6
        df.iloc[j, df.columns.get_loc("low")] = 72.0
        df.iloc[j, df.columns.get_loc("open")] = df["close"].iloc[j] * 0.99
        ind = prepare_ticker(df, RULES)
        assert divergence_add(ind, j, ind["ref_low"].iloc[pin_i], ind["ref_rsi"].iloc[pin_i], RULES)
        tr = TradeRules(hold_months=12, profit_take=0.3)
        t = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1])
        assert t.add_date == ind.index[j + 1]
        assert t.return_pct == pytest.approx(t.slot_return_pct)  # add used the reserve

    def test_data_ending_early_is_flagged(self):
        df, pin_i = _crash_then_pin(rebound_to=85, rebound_bars=30)
        ind = prepare_ticker(df, RULES)
        tr = TradeRules(hold_months=12, profit_take=None)
        t = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1] + pd.Timedelta(days=200))
        assert t.exit_reason == "data_ended"
        t2 = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1])
        assert t2.exit_reason == "open_at_end"

    def test_call_estimate_levers_the_move(self):
        df, pin_i = _crash_then_pin(rebound_to=110, rebound_bars=120)
        ind = prepare_ticker(df, RULES)
        stock = build_trade("XYZ", ind, pin_i, RULES, TradeRules(hold_months=12, profit_take=None, entry_fraction=1.0), ind.index[-1])
        call = build_trade("XYZ", ind, pin_i, RULES,
                           TradeRules(hold_months=12, profit_take=None, entry_fraction=1.0, instrument="call",
                                      option=OptionAssumptions(cost_pct=0.0)),
                           ind.index[-1])
        assert stock.return_pct > 0
        assert call.return_pct > stock.return_pct

    def test_call_loses_most_when_stock_goes_nowhere(self):
        df, pin_i = _crash_then_pin(rebound_to=82, rebound_bars=300)
        ind = prepare_ticker(df, RULES)
        tr = TradeRules(hold_months=12, profit_take=None, entry_fraction=1.0, instrument="call")
        t = build_trade("XYZ", ind, pin_i, RULES, tr, ind.index[-1])
        assert t.return_pct < -50


# ─────────────────────────────────────────────────────────────
# ACCOUNT
# ─────────────────────────────────────────────────────────────

def _fake_trade(ticker, entry, exit_, final, dd=-0.6):
    days = pd.bdate_range(entry, exit_)
    path = pd.Series(np.linspace(1.0, final, len(days)), index=days)
    return Trade(
        ticker=ticker, signal_date=days[0] - pd.Timedelta(days=1), entry_date=days[0], exit_date=days[-1],
        exit_reason="time", entry_stock_price=10.0, exit_stock_price=10.0 * final,
        drawdown_at_signal=dd, rsi_at_signal=25.0, return_pct=(final - 1) * 100,
        slot_return_pct=(final - 1) * 100, bars_held=len(days), path=path,
    )


class TestAccount:
    CAL = pd.bdate_range("2021-01-01", "2021-12-31")

    def test_single_trade_math(self):
        t = _fake_trade("A", "2021-02-01", "2021-03-01", 1.5)
        res = replay_account([t], self.CAL, max_positions=2, initial_capital=10_000)
        # Half the account in one slot, +50% on it = +25%
        assert res.equity.iloc[-1] == pytest.approx(12_500)
        assert res.summary.total_return_pct == pytest.approx(25.0)

    def test_position_cap_and_same_ticker(self):
        trades = [
            _fake_trade("A", "2021-02-01", "2021-06-01", 1.2, dd=-0.7),
            _fake_trade("B", "2021-02-01", "2021-06-01", 1.2, dd=-0.6),
            _fake_trade("C", "2021-02-01", "2021-06-01", 1.2, dd=-0.5),  # no slot left
            _fake_trade("A", "2021-03-01", "2021-07-01", 1.2),           # A already held
            _fake_trade("C", "2021-07-01", "2021-08-01", 1.2),           # slot free again
        ]
        res = replay_account(trades, self.CAL, max_positions=2)
        assert [(t.ticker, str(t.entry_date.date())) for t in res.taken] == [
            ("A", "2021-02-01"), ("B", "2021-02-01"), ("C", "2021-07-01"),
        ]
        assert res.skipped == 2

    def test_non_overlapping_per_ticker(self):
        trades = [
            _fake_trade("A", "2021-02-01", "2021-06-01", 1.2),
            _fake_trade("A", "2021-03-01", "2021-07-01", 1.2),
            _fake_trade("A", "2021-06-02", "2021-07-01", 1.2),
        ]
        kept = non_overlapping(trades)
        assert [str(t.entry_date.date()) for t in kept] == ["2021-02-01", "2021-06-02"]

    def test_trade_stats(self):
        trades = [_fake_trade("A", "2021-02-01", "2021-03-01", f) for f in (1.5, 0.8, 1.1)]
        s = trade_stats(trades)
        assert s.trades == 3
        assert s.win_rate_pct == pytest.approx(200 / 3)
        assert s.worst_pct == pytest.approx(-20.0)

    def test_buy_and_hold_cagr(self):
        idx = pd.bdate_range("2020-01-01", "2022-01-01", tz="UTC")
        close = pd.Series(np.linspace(100, 121, len(idx)), index=idx)
        s = buy_and_hold(close, pd.DatetimeIndex(idx.tz_convert(None).normalize()))
        assert s.cagr_pct == pytest.approx(10.0, abs=0.1)

    def test_run_backtest_end_to_end(self):
        df, pin_i = _crash_then_pin(rebound_to=150)
        prepared = {"XYZ": prepare_ticker(df, RULES)}
        cal = prepared["XYZ"].index
        res = run_backtest(prepared, cal, RULES, TradeRules(hold_months=12, profit_take=0.3), max_positions=3)
        assert res.counts.raw_signals == 1
        assert res.signal_stats.trades == 1
        assert res.account_stats.trades == 1
        assert res.account.summary.total_return_pct > 0

    def test_revenue_filter_drops_unknown_and_falling(self):
        df, pin_i = _crash_then_pin(rebound_to=150)
        prepared = {"XYZ": prepare_ticker(df, RULES)}
        cal = prepared["XYZ"].index
        falling = parse_financials_reported({"data": [
            _filing(2018, "2019-02-15", 120.0), _filing(2019, "2020-02-15", 100.0),
        ]})
        res = run_backtest(prepared, cal, RULES, TradeRules(), revenue_histories={"XYZ": falling}, revenue_filter=True)
        assert res.signal_stats.trades == 0
        assert res.counts.revenue_fail == 1
        res2 = run_backtest(prepared, cal, RULES, TradeRules(), revenue_histories={}, revenue_filter=True)
        assert res2.counts.revenue_unknown == 1 and res2.signal_stats.trades == 0


# ─────────────────────────────────────────────────────────────
# DATA + CLI
# ─────────────────────────────────────────────────────────────

class _FakeAlpaca:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def get_historical_bars(self, symbols, start, end, timeframe_str="1Day", feed="iex", asof=None):
        self.calls.append((tuple(symbols), feed, asof))
        return {s: self.frames[s] for s in symbols if s in self.frames}


def test_universe_and_loader_pass_asof():
    cfg = {"universe": {"tickers": ["aapl", "AAPL", "msft"],
                        "delisted": [{"symbol": "SIVB", "asof": "2023-03-08"}]}}
    uni = universe_from_config(cfg)
    assert uni == [("AAPL", None), ("MSFT", None), ("SIVB", "2023-03-08")]
    fake = _FakeAlpaca({"AAPL": _bars([1, 2]), "SIVB": _bars([3, 4])})
    out = load_daily_bars(fake, uni, pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"))
    assert set(out) == {"AAPL", "SIVB"}
    assert (("SIVB",), "sip", "2023-03-08") in fake.calls


def test_cli_runs_on_fake_data(monkeypatch, tmp_path):
    from click.testing import CliRunner

    import signal_engine.cli as cli
    import signal_engine.data.alpaca_client as ac
    import signal_engine.data.cache as cache_mod

    df, _ = _crash_then_pin(rebound_to=150)
    spy = _bars(np.linspace(300, 330, len(df)))
    fake = _FakeAlpaca({"XYZ": df, "SPY": spy})
    monkeypatch.setattr(ac, "AlpacaDataClient", lambda: fake)
    monkeypatch.setattr(cache_mod, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache_mod.ParquetCache.__init__, "__defaults__", (tmp_path / "cache", 24))
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "leaps_dip.yaml").write_text(
        "backtest: {start: '2020-01-01', holds_months: [6], max_positions: 3, revenue_filter: false}\n"
        "rules: {min_history_bars: 50}\n"
        "trade: {stock_profit_take: 0.3, call_profit_take: 1.0}\n"
        "universe: {tickers: [XYZ]}\n"
    )
    monkeypatch.setattr(cli, "CONFIG_DIR", cfg_dir)
    out_csv = tmp_path / "trades.csv"
    result = CliRunner().invoke(cli.main, ["leaps-dip-backtest", "--trades-csv", str(out_csv)])
    assert result.exit_code == 0, result.output
    assert "SPY buy and hold" in result.output
    trades = pd.read_csv(out_csv)
    assert set(trades["instrument"]) == {"stock", "call"}
    assert trades["ticker"].eq("XYZ").all()
