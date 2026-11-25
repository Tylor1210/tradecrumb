"""
Trade and account simulation for the dip strategy.

How a trade works
  - Signal at the close of day t (see rules.py), entry at the open of t+1.
  - Each trade owns a "slot" of the account. entry_fraction of the slot is
    bought on entry; the rest is held back in cash for one add on bullish
    RSI divergence. If no add happens, that cash just sits in the slot.
  - Exits are checked at each close, in this order: holding period reached,
    profit take hit, optional stop loss, option expiry. Exits fill at that
    close. If the ticker's data stops early (delisted, bought out) the trade
    closes at its last close and is flagged "data_ended".
  - instrument "stock" trades the shares. instrument "call" is an ESTIMATE:
    one long call priced with Black-Scholes from trailing realized
    volatility, with a spread cost on each side. It is not an options
    backtest; real option quotes would be needed for that.

Account replay
  One cash balance. A new trade gets equity / max_positions as its slot,
  capped by free cash. When max_positions trades are open, or the ticker is
  already held, the new signal is skipped. Same-day signals are taken
  deepest drawdown first.

Depends on: pandas/numpy, leaps.candidate_screener's Black-Scholes helper.
"""

from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd

from signal_engine.leaps.candidate_screener import _black_scholes_call
from signal_engine.leaps_dip.fundamentals import revenue_growing_as_of
from signal_engine.leaps_dip.rules import DipRules, add_indicators, divergence_add

TRADING_DAYS_PER_MONTH = 21


@dataclass
class OptionAssumptions:
    """Rough LEAPS call assumptions. Every number here is a guess, not a quote."""

    min_years: float = 1.0          # LEAPS: at least a year to expiry
    extra_years: float = 0.25       # expiry = max(min_years, hold + extra_years)
    strike_pct: float = 1.0         # strike / stock price at entry (1.0 = at the money)
    vol_window: int = 60            # trailing days of realized volatility used as IV
    vol_floor: float = 0.25
    vol_multiplier: float = 1.0
    risk_free: float = 0.045
    cost_pct: float = 3.0           # % lost to the bid/ask spread on each side

    @classmethod
    def from_config(cls, cfg: dict) -> "OptionAssumptions":
        known = {k: v for k, v in (cfg or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TradeRules:
    hold_months: int = 12
    profit_take: Optional[float] = 0.5   # 0.5 = take profit at +50% on the instrument
    stop_loss: Optional[float] = None    # None = no stop (the video uses none)
    entry_fraction: float = 2 / 3        # share of the slot bought on entry
    fees_pct: float = 0.1                # stock fee/slippage per side, %
    instrument: str = "stock"            # "stock" or "call"
    option: OptionAssumptions = field(default_factory=OptionAssumptions)

    @property
    def hold_bars(self) -> int:
        return int(self.hold_months * TRADING_DAYS_PER_MONTH)


@dataclass
class Trade:
    ticker: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    exit_reason: str                 # time, profit, stop, expiry, data_ended, open_at_end
    entry_stock_price: float
    exit_stock_price: float
    drawdown_at_signal: float        # e.g. -0.62
    rsi_at_signal: float
    return_pct: float                # on money actually put in (entry + add), after costs
    slot_return_pct: float           # on the whole slot, idle add cash included
    bars_held: int
    add_date: Optional[pd.Timestamp] = None
    revenue_growing: Optional[bool] = None
    path: pd.Series = field(default_factory=pd.Series, repr=False)  # slot value at each close, starts at 1.0

    @property
    def stock_return_pct(self) -> float:
        return (self.exit_stock_price / self.entry_stock_price - 1.0) * 100

    def value_on(self, day: pd.Timestamp) -> float:
        v = self.path.asof(day)
        return 1.0 if pd.isna(v) else float(v)


def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    rets = np.log(close).diff()
    return rets.rolling(window, min_periods=max(5, window // 3)).std() * sqrt(252)


class _Instrument:
    """Prices one unit of what the trade holds, for buys and for marks."""

    def __init__(self, ind: pd.DataFrame, entry_i: int, signal_i: int, tr: TradeRules) -> None:
        self.tr = tr
        self.ind = ind
        self.entry_date = ind.index[entry_i]
        if tr.instrument == "call":
            o = tr.option
            years = max(o.min_years, tr.hold_months / 12 + o.extra_years)
            self.expiry = self.entry_date + pd.Timedelta(days=round(years * 365.25))
            self.strike = float(ind["open"].iloc[entry_i]) * o.strike_pct

    def _sigma(self, i: int) -> float:
        o = self.tr.option
        v = self.ind["vol"].iloc[i]
        v = o.vol_floor if pd.isna(v) else float(v) * o.vol_multiplier
        return max(o.vol_floor, v)

    def _years_left(self, i: int) -> float:
        return max(0.0, (self.expiry - self.ind.index[i]).days / 365.25)

    def buy_price(self, i: int, at: str) -> float:
        px = float(self.ind[at].iloc[i])
        if self.tr.instrument == "call":
            # Price with yesterday's vol at the open (today's isn't known yet)
            sig = self._sigma(max(0, i - 1)) if at == "open" else self._sigma(i)
            prem = _black_scholes_call(px, self.strike, self._years_left(i), self.tr.option.risk_free, sig)
            return prem * (1 + self.tr.option.cost_pct / 100)
        return px * (1 + self.tr.fees_pct / 100)

    def sell_price(self, i: int) -> float:
        px = float(self.ind["close"].iloc[i])
        if self.tr.instrument == "call":
            prem = _black_scholes_call(px, self.strike, self._years_left(i), self.tr.option.risk_free, self._sigma(i))
            return prem * (1 - self.tr.option.cost_pct / 100)
        return px * (1 - self.tr.fees_pct / 100)

    def expired(self, i: int) -> bool:
        return self.tr.instrument == "call" and self._years_left(i) <= 0


def build_trade(
    ticker: str,
    ind: pd.DataFrame,
    signal_i: int,
    rules: DipRules,
    tr: TradeRules,
    data_end: pd.Timestamp,
    revenue_growing: Optional[bool] = None,
) -> Optional[Trade]:
    """Walk one signal forward to its exit. ind comes from prepare_ticker()."""
    entry_i = signal_i + 1
    if entry_i >= len(ind):
        return None
    inst = _Instrument(ind, entry_i, signal_i, tr)
    entry_cost = inst.buy_price(entry_i, "open")
    if not entry_cost > 0:
        return None

    units = tr.entry_fraction / entry_cost
    invested = tr.entry_fraction
    reserve = 1.0 - tr.entry_fraction
    ref_low = float(ind["ref_low"].iloc[signal_i])
    ref_rsi = float(ind["ref_rsi"].iloc[signal_i])
    adds = 0
    add_pending = False
    add_date = None
    dates, values = [], []
    exit_reason = None
    last_i = len(ind) - 1
    i = entry_i

    for i in range(entry_i, len(ind)):
        if add_pending:
            price = inst.buy_price(i, "open")
            if price > 0:
                units += reserve / price
                invested += reserve
                reserve = 0.0
                adds += 1
                add_date = ind.index[i]
            add_pending = False

        unit_value = inst.sell_price(i)
        slot_value = units * unit_value + reserve
        dates.append(ind.index[i])
        values.append(slot_value)
        inst_return = (units * unit_value - invested) / invested

        held = i - entry_i + 1
        if held >= tr.hold_bars:
            exit_reason = "time"
        elif tr.profit_take is not None and inst_return >= tr.profit_take:
            exit_reason = "profit"
        elif tr.stop_loss is not None and inst_return <= -tr.stop_loss:
            exit_reason = "stop"
        elif inst.expired(i):
            exit_reason = "expiry"
        elif i == last_i:
            ended_early = ind.index[i] < data_end - pd.Timedelta(days=10)
            exit_reason = "data_ended" if ended_early else "open_at_end"
        if exit_reason:
            break

        if (
            adds < rules.max_adds
            and reserve > 0
            and i - signal_i >= rules.divergence_min_bars
            and divergence_add(ind, i, ref_low, ref_rsi, rules)
        ):
            add_pending = True

    final_value = values[-1]
    return Trade(
        ticker=ticker,
        signal_date=ind.index[signal_i],
        entry_date=ind.index[entry_i],
        exit_date=ind.index[i],
        exit_reason=exit_reason or "open_at_end",
        entry_stock_price=float(ind["open"].iloc[entry_i]),
        exit_stock_price=float(ind["close"].iloc[i]),
        drawdown_at_signal=float(ind["drawdown"].iloc[signal_i]),
        rsi_at_signal=float(ind["ref_rsi"].iloc[signal_i]),
        return_pct=(final_value - 1.0) / invested * 100,
        slot_return_pct=(final_value - 1.0) * 100,
        bars_held=i - entry_i + 1,
        add_date=add_date,
        revenue_growing=revenue_growing,
        path=pd.Series(values, index=pd.DatetimeIndex(dates)),
    )


def prepare_ticker(df: pd.DataFrame, rules: DipRules, vol_window: int = 60) -> pd.DataFrame:
    ind = add_indicators(_naive_index(df), rules)
    ind["vol"] = realized_vol(ind["close"], vol_window)
    return ind


@dataclass
class SignalCounts:
    raw_signals: int = 0            # pin bar + RSI + drawdown, before the revenue check
    revenue_pass: int = 0
    revenue_fail: int = 0
    revenue_unknown: int = 0


def candidate_trades(
    ticker: str,
    ind: pd.DataFrame,
    rules: DipRules,
    tr: TradeRules,
    data_end: pd.Timestamp,
    revenue_history: Optional[pd.DataFrame] = None,
    revenue_filter: bool = False,
    counts: Optional[SignalCounts] = None,
) -> list[Trade]:
    """One trade per signal (overlapping allowed; the account decides which run)."""
    trades = []
    for signal_i in np.flatnonzero(ind["signal"].to_numpy()):
        growing = None
        if revenue_history is not None:
            growing = revenue_growing_as_of(revenue_history, ind.index[signal_i])
        if counts is not None:
            counts.raw_signals += 1
            if growing is True:
                counts.revenue_pass += 1
            elif growing is False:
                counts.revenue_fail += 1
            else:
                counts.revenue_unknown += 1
        if revenue_filter and growing is not True:
            continue
        t = build_trade(ticker, ind, int(signal_i), rules, tr, data_end, growing)
        if t is not None:
            trades.append(t)
    return trades


def non_overlapping(trades: list[Trade]) -> list[Trade]:
    """Per ticker, earliest trade first; skip any that enters before the last one exited."""
    out = []
    last_exit: dict[str, pd.Timestamp] = {}
    for t in sorted(trades, key=lambda t: (t.entry_date, t.ticker)):
        prev = last_exit.get(t.ticker)
        if prev is not None and t.entry_date <= prev:
            continue
        out.append(t)
        last_exit[t.ticker] = t.exit_date
    return out


# ─────────────────────────────────────────────────────────────
# ACCOUNT
# ─────────────────────────────────────────────────────────────

@dataclass
class AccountSummary:
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0     # positive, 30 = fell 30% from a peak
    sharpe_ratio: float = 0.0
    avg_invested_pct: float = 0.0     # average share of equity in open trades (slots)
    years: float = 0.0


@dataclass
class AccountResult:
    equity: pd.Series
    taken: list[Trade]
    skipped: int
    summary: AccountSummary


def equity_summary(equity: pd.Series, invested: Optional[pd.Series] = None) -> AccountSummary:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return AccountSummary()
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else -1.0
    dd = (equity / equity.cummax() - 1).min()
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * sqrt(252)) if rets.std() > 0 else 0.0
    avg_inv = float((invested / equity).mean() * 100) if invested is not None else 0.0
    return AccountSummary(
        total_return_pct=float(total * 100),
        cagr_pct=float(cagr * 100),
        max_drawdown_pct=float(-dd * 100),
        sharpe_ratio=round(sharpe, 2),
        avg_invested_pct=avg_inv,
        years=round(years, 2),
    )


def replay_account(
    trades: list[Trade],
    calendar: pd.DatetimeIndex,
    max_positions: int = 5,
    initial_capital: float = 10_000.0,
) -> AccountResult:
    pending = sorted(trades, key=lambda t: (t.entry_date, t.drawdown_at_signal, t.ticker))
    p = 0
    cash = initial_capital
    equity_prev = initial_capital
    active: list[tuple[Trade, float]] = []
    taken: list[Trade] = []
    skipped = 0
    eq_vals, inv_vals = [], []

    for day in calendar:
        while p < len(pending) and pending[p].entry_date <= day:
            t = pending[p]
            p += 1
            held = {a.ticker for a, _ in active}
            if len(active) >= max_positions or t.ticker in held:
                skipped += 1
                continue
            slot = min(equity_prev / max_positions, cash)
            if slot <= 0:
                skipped += 1
                continue
            cash -= slot
            active.append((t, slot))
            taken.append(t)

        open_value = 0.0
        still = []
        for t, slot in active:
            if day >= t.exit_date:
                cash += slot * float(t.path.iloc[-1])
            else:
                open_value += slot * t.value_on(day)
                still.append((t, slot))
        active = still
        equity_prev = cash + open_value
        eq_vals.append(equity_prev)
        inv_vals.append(open_value)

    equity = pd.Series(eq_vals, index=calendar)
    invested = pd.Series(inv_vals, index=calendar)
    return AccountResult(equity, taken, skipped, equity_summary(equity, invested))


# ─────────────────────────────────────────────────────────────
# TRADE STATS
# ─────────────────────────────────────────────────────────────

@dataclass
class TradeStats:
    trades: int = 0
    win_rate_pct: float = 0.0
    avg_pct: float = 0.0
    median_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_pct: float = 0.0
    worst_pct: float = 0.0
    profit_takes: int = 0
    adds: int = 0
    data_ended: int = 0
    open_at_end: int = 0
    avg_months_held: float = 0.0


def trade_stats(trades: list[Trade]) -> TradeStats:
    if not trades:
        return TradeStats()
    r = np.array([t.return_pct for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    return TradeStats(
        trades=len(trades),
        win_rate_pct=float((r > 0).mean() * 100),
        avg_pct=float(r.mean()),
        median_pct=float(np.median(r)),
        avg_win_pct=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_pct=float(losses.mean()) if len(losses) else 0.0,
        best_pct=float(r.max()),
        worst_pct=float(r.min()),
        profit_takes=sum(t.exit_reason == "profit" for t in trades),
        adds=sum(t.add_date is not None for t in trades),
        data_ended=sum(t.exit_reason == "data_ended" for t in trades),
        open_at_end=sum(t.exit_reason == "open_at_end" for t in trades),
        avg_months_held=float(np.mean([t.bars_held for t in trades]) / TRADING_DAYS_PER_MONTH),
    )


# ─────────────────────────────────────────────────────────────
# FULL RUN
# ─────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    tr: TradeRules
    signal_stats: TradeStats         # every signal, one at a time per ticker
    account: AccountResult
    account_stats: TradeStats        # trades the account actually took
    counts: SignalCounts


def run_backtest(
    prepared: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    rules: DipRules,
    tr: TradeRules,
    max_positions: int = 5,
    initial_capital: float = 10_000.0,
    revenue_histories: Optional[dict[str, pd.DataFrame]] = None,
    revenue_filter: bool = False,
) -> RunResult:
    """prepared maps ticker -> prepare_ticker() output; calendar is the SPY trading days."""
    data_end = calendar[-1]
    counts = SignalCounts()
    candidates: list[Trade] = []
    for ticker, ind in prepared.items():
        hist = revenue_histories.get(ticker) if revenue_histories is not None else None
        candidates += candidate_trades(
            ticker, ind, rules, tr, data_end,
            revenue_history=hist if revenue_histories is not None else None,
            revenue_filter=revenue_filter, counts=counts,
        )
    sequential = non_overlapping(candidates)
    account = replay_account(candidates, calendar, max_positions, initial_capital)
    return RunResult(
        tr=tr,
        signal_stats=trade_stats(sequential),
        account=account,
        account_stats=trade_stats(account.taken),
        counts=counts,
    )


def buy_and_hold(close: pd.Series, calendar: pd.DatetimeIndex) -> AccountSummary:
    s = _naive_index(close.to_frame("close"))["close"].reindex(calendar).ffill().dropna()
    return equity_summary(s)
