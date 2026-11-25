"""
Account-level replay of backtest trades: one shared cash balance, explicit
position sizing, a cap on open positions, and weekly/monthly return stats.

Why this exists: the vectorbt portfolios in engine.py run every ticker as its
own column with its own starting cash. With more than one ticker,
pf.total_return() and pf.max_drawdown() return a Series (one value per
ticker), float() on that raises, and the metrics fell back to 0.00%. Even when
they did compute, a per-ticker number is not what the account would have done.

This module takes the trades vectorbt already decided (ticker, entry date and
price, exit date and price) and replays them through one account:

  - each day, exits are processed first (cash comes back), then new entries
  - a new position gets position_size_pct of current account equity
    (cash + open positions marked at that day's close), capped by free cash
  - if max_positions are already open, the new trade is skipped
  - fees_pct is charged on each side, like the vectorbt runs (0.1%)
  - open positions are marked to market at each day's close

Signal-level stats (win rate, avg loss, worst trade) still come from every
trade the signals produced. The account stats here show what one account
following those signals with this sizing would have earned week to week.

Depends on: pandas/numpy only. Pure functions, no I/O.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AccountSizing:
    """How the account sizes trades. Defaults live in config/backtest.yaml."""

    position_size_pct: float = 10.0   # % of current equity put into each new trade
    max_positions: int = 10           # most trades open at once; extra signals are skipped
    fees_pct: float = 0.1             # % charged per side (buy and sell)


@dataclass
class PeriodStats:
    """Return stats for one bucket size (weeks or months)."""

    periods: int = 0
    avg_pct: float = 0.0
    median_pct: float = 0.0
    positive_pct: float = 0.0      # share of periods that ended up
    best_pct: float = 0.0
    worst_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "periods": self.periods,
            "avg_pct": self.avg_pct,
            "median_pct": self.median_pct,
            "positive_pct": self.positive_pct,
            "best_pct": self.best_pct,
            "worst_pct": self.worst_pct,
        }


@dataclass
class AccountStats:
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0   # positive number, e.g. 12.5 = fell 12.5% from a peak
    sharpe_ratio: float = 0.0       # annualized from daily returns, no risk-free rate
    trades_taken: int = 0
    trades_skipped: int = 0         # signals skipped because max_positions were open
    win_rate_pct: float = 0.0       # of trades the account actually took
    avg_invested_pct: float = 0.0   # average share of equity in open positions
    weekly: PeriodStats = field(default_factory=PeriodStats)
    monthly: PeriodStats = field(default_factory=PeriodStats)

    def to_dict(self) -> dict:
        return {
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "trades_taken": self.trades_taken,
            "trades_skipped": self.trades_skipped,
            "win_rate_pct": self.win_rate_pct,
            "avg_invested_pct": self.avg_invested_pct,
            "weekly": self.weekly.to_dict(),
            "monthly": self.monthly.to_dict(),
        }


@dataclass
class AccountResult:
    equity: pd.Series               # account value at each day's close
    invested: pd.Series             # value of open positions at each day's close
    trades: pd.DataFrame            # trades the account took, with dollar PnL
    skipped: int
    initial_capital: float

    def stats(self) -> AccountStats:
        return account_stats(self)


def trades_from_portfolio(pf) -> pd.DataFrame:
    """
    Pull a plain trade list out of a vectorbt Portfolio.

    Columns: ticker, entry_date, entry_price, exit_date (NaT if still open),
    exit_price (NaN if still open), return_pct (signal-level, after fees).
    """
    rec = pf.trades.records_readable
    if rec is None or len(rec) == 0:
        return pd.DataFrame(
            columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "return_pct"]
        )
    out = pd.DataFrame({
        "ticker": rec["Column"].astype(str).to_numpy(),
        "entry_date": pd.to_datetime(rec["Entry Timestamp"]).to_numpy(),
        "entry_price": rec["Avg Entry Price"].astype(float).to_numpy(),
        "exit_date": pd.to_datetime(rec["Exit Timestamp"]).to_numpy(),
        "exit_price": rec["Avg Exit Price"].astype(float).to_numpy(),
        "return_pct": (rec["Return"].astype(float) * 100).to_numpy(),
    })
    is_open = (rec["Status"].astype(str) == "Open").to_numpy()
    out.loc[is_open, "exit_date"] = pd.NaT
    out.loc[is_open, "exit_price"] = np.nan
    return out


def simulate_account(
    trades: pd.DataFrame,
    close_df: pd.DataFrame,
    initial_capital: float,
    sizing: AccountSizing,
) -> AccountResult:
    """
    Replay trades through one shared account.

    trades:   output of trades_from_portfolio (or the same columns built by hand)
    close_df: daily closes, one column per ticker, used to mark open positions
    """
    close_df = close_df.sort_index().ffill()
    dates = close_df.index
    fee = sizing.fees_pct / 100.0
    frac = sizing.position_size_pct / 100.0

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    # Entries grouped by day, in a stable order (ticker name) so runs repeat exactly.
    by_entry: dict = {}
    for i, row in trades.sort_values(["entry_date", "ticker"]).iterrows():
        by_entry.setdefault(row["entry_date"], []).append(i)

    cash = float(initial_capital)
    open_pos: dict = {}   # trade index -> (ticker, shares, exit_date, exit_price, cost)
    taken_rows = []
    skipped = 0
    equity_vals = np.empty(len(dates))
    invested_vals = np.empty(len(dates))

    for d_i, day in enumerate(dates):
        prices = close_df.iloc[d_i]

        # 1) exits that happen today
        for t_idx in [k for k, p in open_pos.items() if not pd.isna(p[2]) and p[2] <= day]:
            ticker, shares, _exit_date, exit_price, cost = open_pos.pop(t_idx)
            proceeds = shares * exit_price * (1 - fee)
            cash += proceeds
            taken_rows.append((t_idx, proceeds - cost))

        # 2) equity before today's entries, marked at today's close
        def _mark() -> float:
            total = 0.0
            for ticker, shares, *_ in open_pos.values():
                px = prices.get(ticker, np.nan)
                if not np.isnan(px):
                    total += shares * float(px)
            return total

        invested = _mark()
        equity = cash + invested

        # 3) new entries
        for t_idx in by_entry.get(day, []):
            row = trades.loc[t_idx]
            if len(open_pos) >= sizing.max_positions:
                skipped += 1
                continue
            budget = min(frac * equity, cash)
            if budget <= 0 or not row["entry_price"] or np.isnan(row["entry_price"]):
                skipped += 1
                continue
            shares = budget / (row["entry_price"] * (1 + fee))
            cash -= budget
            exit_date = row["exit_date"] if not pd.isna(row["exit_date"]) else pd.NaT
            open_pos[t_idx] = (row["ticker"], shares, exit_date, float(row["exit_price"]), budget)

        invested = _mark()
        equity_vals[d_i] = cash + invested
        invested_vals[d_i] = invested

    # Trades still open at the end: count their unrealized PnL at the last close.
    if len(dates):
        last = close_df.iloc[-1]
        for t_idx, (ticker, shares, _e, _p, cost) in open_pos.items():
            px = float(last.get(ticker, np.nan))
            taken_rows.append((t_idx, shares * px * (1 - fee) - cost if not np.isnan(px) else 0.0))

    taken_idx = [t for t, _ in taken_rows]
    taken = trades.loc[taken_idx].copy() if taken_idx else trades.iloc[0:0].copy()
    taken["pnl"] = [p for _, p in taken_rows] if taken_rows else []

    return AccountResult(
        equity=pd.Series(equity_vals, index=dates, name="equity"),
        invested=pd.Series(invested_vals, index=dates, name="invested"),
        trades=taken,
        skipped=skipped,
        initial_capital=float(initial_capital),
    )


def period_returns(equity: pd.Series, initial_capital: float, rule: str) -> pd.Series:
    """
    Percent return of each calendar bucket (rule "W" = weeks ending Friday,
    "M" = calendar months). The first bucket is measured from the starting
    capital, so a loss in the first week still counts.
    """
    if equity.empty:
        return pd.Series(dtype=float)
    freq = {"W": "W-FRI", "M": "ME"}[rule]
    ends = equity.resample(freq).last().dropna()
    starts = ends.shift(1)
    starts.iloc[0] = initial_capital
    return (ends / starts - 1.0) * 100.0


def _period_stats(rets: pd.Series) -> PeriodStats:
    if rets.empty:
        return PeriodStats()
    return PeriodStats(
        periods=int(len(rets)),
        avg_pct=round(float(rets.mean()), 2),
        median_pct=round(float(rets.median()), 2),
        positive_pct=round(float((rets > 0).mean() * 100), 1),
        best_pct=round(float(rets.max()), 2),
        worst_pct=round(float(rets.min()), 2),
    )


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max() * 100)


def account_stats(result: AccountResult) -> AccountStats:
    eq = result.equity
    if eq.empty:
        return AccountStats()
    start = result.initial_capital
    end = float(eq.iloc[-1])
    total = (end / start - 1.0) * 100.0

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
    cagr = ((end / start) ** (1 / years) - 1.0) * 100.0 if end > 0 else -100.0

    daily = eq.pct_change().dropna()
    sharpe = 0.0
    if len(daily) > 1 and daily.std() > 0:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))

    trades = result.trades
    win_rate = float((trades["pnl"] > 0).mean() * 100) if len(trades) else 0.0
    invested_share = (result.invested / eq.replace(0, np.nan)).fillna(0.0)

    return AccountStats(
        total_return_pct=round(total, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_drawdown_pct(eq), 2),
        sharpe_ratio=round(sharpe, 3),
        trades_taken=int(len(trades)),
        trades_skipped=int(result.skipped),
        win_rate_pct=round(win_rate, 1),
        avg_invested_pct=round(float(invested_share.mean() * 100), 1),
        weekly=_period_stats(period_returns(eq, start, "W")),
        monthly=_period_stats(period_returns(eq, start, "M")),
    )


def buy_and_hold_stats(close: pd.Series, initial_capital: float = 100_000.0) -> Optional[AccountStats]:
    """Same stats for simply holding one series (e.g. SPY) the whole time, as a yardstick."""
    close = close.dropna().astype(float)
    if len(close) < 2:
        return None
    equity = close / float(close.iloc[0]) * initial_capital
    res = AccountResult(
        equity=equity,
        invested=equity,
        trades=pd.DataFrame({"pnl": [float(equity.iloc[-1] - initial_capital)]}),
        skipped=0,
        initial_capital=initial_capital,
    )
    return account_stats(res)
