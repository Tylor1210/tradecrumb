"""Tests for the shared-account replay (backtest/account.py) and its wiring into the engine."""

import numpy as np
import pandas as pd
import pytest

from signal_engine.backtest.account import (
    AccountSizing,
    account_stats,
    buy_and_hold_stats,
    max_drawdown_pct,
    period_returns,
    simulate_account,
)
from signal_engine.backtest.engine import BacktestConfig, BacktestEngine
from signal_engine.risk.loss_cap import LossCapRule


def _trades(rows):
    return pd.DataFrame(rows, columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct"])


def _days(n, start="2024-01-01"):
    return pd.bdate_range(start, periods=n)


NO_FEES = AccountSizing(position_size_pct=10.0, max_positions=10, fees_pct=0.0)


def test_single_trade_uses_fraction_of_equity():
    idx = _days(5)
    close = pd.DataFrame({"A": [100, 100, 105, 110, 110]}, index=idx, dtype=float)
    trades = _trades([("A", idx[1], 100.0, idx[3], 110.0, 10.0)])
    res = simulate_account(trades, close, 100_000, NO_FEES)
    # 10% of 100k = 10k into A, up 10% = +1,000 on the account
    assert res.equity.iloc[-1] == pytest.approx(101_000)
    assert res.equity.iloc[2] == pytest.approx(100_500)  # marked at 105 while open
    assert res.trades["pnl"].iloc[0] == pytest.approx(1_000)


def test_fees_charged_both_sides():
    idx = _days(3)
    close = pd.DataFrame({"A": [100, 100, 100]}, index=idx, dtype=float)
    trades = _trades([("A", idx[0], 100.0, idx[2], 100.0, 0.0)])
    res = simulate_account(trades, close, 100_000, AccountSizing(10.0, 10, 0.1))
    # flat trade loses roughly 0.1% on each side of a 10k position
    assert res.equity.iloc[-1] == pytest.approx(100_000 - 20, abs=0.5)
    assert res.trades["pnl"].iloc[0] < 0


def test_max_positions_skips_extra_signals():
    idx = _days(4)
    close = pd.DataFrame({t: [10.0] * 4 for t in "ABC"}, index=idx)
    trades = _trades([(t, idx[0], 10.0, idx[2], 10.0, 0.0) for t in "ABC"])
    sizing = AccountSizing(position_size_pct=10.0, max_positions=2, fees_pct=0.0)
    res = simulate_account(trades, close, 100_000, sizing)
    assert res.skipped == 1
    assert len(res.trades) == 2
    assert list(res.trades["ticker"]) == ["A", "B"]  # stable order by ticker


def test_exit_frees_slot_same_day():
    idx = _days(4)
    close = pd.DataFrame({"A": [10.0] * 4, "B": [10.0] * 4}, index=idx)
    trades = _trades([
        ("A", idx[0], 10.0, idx[2], 10.0, 0.0),
        ("B", idx[2], 10.0, idx[3], 10.0, 0.0),
    ])
    res = simulate_account(trades, close, 100_000, AccountSizing(10.0, 1, 0.0))
    assert res.skipped == 0
    assert len(res.trades) == 2


def test_sizing_compounds_with_equity():
    idx = _days(6)
    close = pd.DataFrame({"A": [100, 200, 200, 200, 200, 200]}, index=idx, dtype=float)
    trades = _trades([
        ("A", idx[0], 100.0, idx[1], 200.0, 100.0),   # +10k on a 10k position
        ("A", idx[2], 200.0, idx[4], 200.0, 0.0),
    ])
    res = simulate_account(trades, close, 100_000, NO_FEES)
    # second trade is 10% of 110k, not 10% of the starting 100k
    assert res.invested.iloc[3] == pytest.approx(11_000)


def test_open_trade_marked_to_market_and_counted():
    idx = _days(4)
    close = pd.DataFrame({"A": [50, 50, 40, 45]}, index=idx, dtype=float)
    trades = _trades([("A", idx[1], 50.0, pd.NaT, np.nan, np.nan)])
    res = simulate_account(trades, close, 100_000, NO_FEES)
    assert res.equity.iloc[-1] == pytest.approx(99_000)
    assert len(res.trades) == 1


def test_period_returns_first_period_from_initial_capital():
    idx = pd.bdate_range("2024-01-01", "2024-01-19")  # three Mon-Fri weeks
    eq = pd.Series(100_000.0, index=idx)
    eq.loc["2024-01-05"] = 99_000   # week 1 ends down 1%
    eq.loc["2024-01-08":] = 99_000
    eq.loc["2024-01-12"] = 101_970  # week 2 ends up 3%
    eq.loc["2024-01-15":] = 101_970
    weekly = period_returns(eq, 100_000, "W")
    assert len(weekly) == 3
    assert weekly.iloc[0] == pytest.approx(-1.0)
    assert weekly.iloc[1] == pytest.approx(3.0)
    assert weekly.iloc[2] == pytest.approx(0.0)


def test_max_drawdown_and_stats():
    idx = pd.bdate_range("2024-01-01", periods=60)
    eq = pd.Series(np.linspace(100_000, 120_000, 60), index=idx)
    eq.iloc[30:40] = eq.iloc[29] * 0.9  # 10% below the peak
    assert max_drawdown_pct(eq) == pytest.approx(10.0, abs=0.01)

    res = simulate_account(_trades([]), pd.DataFrame({"A": 1.0}, index=idx), 100_000, NO_FEES)
    res.equity = eq
    stats = account_stats(res)
    assert stats.total_return_pct == pytest.approx(20.0)
    assert stats.monthly.periods == 3
    assert stats.cagr_pct > 20.0  # 20% in under a year annualizes higher
    assert 0 < stats.weekly.positive_pct <= 100


def test_buy_and_hold_benchmark():
    idx = pd.bdate_range("2024-01-01", periods=10)
    stats = buy_and_hold_stats(pd.Series(np.linspace(100, 110, 10), index=idx))
    assert stats.total_return_pct == pytest.approx(10.0)
    assert stats.max_drawdown_pct == 0.0


# ── engine wiring: the multi-ticker 0.00% bug ────────────────────────────────

def _price_frame(prices, idx):
    s = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"open": s, "high": s * 1.01, "low": s * 0.99, "close": s, "volume": 1e6})


@pytest.mark.parametrize("loss_cap", [None, LossCapRule()])
def test_multi_ticker_return_and_drawdown_not_zero(loss_cap):
    pytest.importorskip("vectorbt")
    idx = pd.bdate_range("2023-01-02", periods=80)
    up = np.linspace(100, 140, 80)
    wave = 100 + 8 * np.sin(np.arange(80) / 4)
    price_data = {"UP": _price_frame(up, idx), "WAVE": _price_frame(wave, idx)}
    sig = pd.Series(False, index=idx)
    sig.iloc[::10] = True
    signals = {"UP": sig, "WAVE": sig}

    cfg = BacktestConfig(
        start_date="2023-01-02", end_date="2023-04-21", hold_period_days=5,
        stop_loss_pct=5.0, take_profit_pct=50.0, initial_capital=100_000,
        position_size_pct=10.0, max_positions=10, loss_cap=loss_cap,
    )
    m = BacktestEngine(cfg).run(price_data, signals).get_metrics()

    assert m.total_trades > 0
    assert m.account is not None
    assert m.total_return_pct != 0.0
    assert m.max_drawdown_pct > 0.0
    assert m.account.weekly.periods > 0
    assert m.account.monthly.periods >= 3
    assert m.account.trades_taken == m.total_trades  # two tickers never fill 10 slots
