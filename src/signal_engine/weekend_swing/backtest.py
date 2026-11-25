"""
Weekend Swing Signals — backtest harness.

Event-driven, per-ticker-per-week simulation — not a reuse of
backtest/engine.py, which is daily-bar/hold-N-days oriented and doesn't fit
weekly-trigger + absolute-level-stop + Friday-forced-exit mechanics. Each
strategy's real scan()/check_trigger() code is replayed against real
historical bars (via IntradayBarStore), so backtest results reflect exactly
the same code path that would run live.

Trade simulation (entry -> stop-or-Friday-close exit) is a direct, testable
event walk — deliberately not forced through vectorbt's from_signals(),
which is built around a single continuous price series and doesn't map
cleanly onto "one trade a week, entry from an intraday trigger, exit checked
across two different bar resolutions." Instead, vectorbt is used where it
fits naturally: its `.vbt.returns` pandas accessor computes portfolio-level
stats (max drawdown) from the resulting trade-return series.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import vectorbt as vbt  # noqa: F401 (registers the .vbt accessor)

from signal_engine.risk.loss_cap import LossCapRule, loss_pct_from_entry
from signal_engine.weekend_swing.earnings import get_earnings_exclusions
from signal_engine.weekend_swing.levels import (
    bar_dates,
    calendar_day,
    monday_session_bars,
    session_open_time,
)

logger = logging.getLogger(__name__)


@dataclass
class WeekendSwingTrade:
    ticker: str
    strategy: str
    direction: str
    entry_time: object
    entry_price: float
    stop_price: float
    exit_time: object
    exit_price: float
    exit_reason: str  # "stopped_out" | "friday_close" | "loss_cap_*" (see risk/loss_cap.py)
    return_pct: float
    gap_type: Optional[str] = None  # strategy 3 only
    breakdown: dict = field(default_factory=dict)  # the triggering signal's breakdown (volume ratio, ATR, etc.)
    regime: Optional[dict] = None  # MarketRegime.to_dict() as of the trade's Monday (None if no provider set)


@dataclass
class WeekendSwingBacktestMetrics:
    strategy: str
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    trade_count: int
    gap_type: Optional[str] = None  # set only for strategy 3's split reporting
    worst_trade_pct: float = 0.0    # single worst trade return (negative)
    loss_cap_exits: int = 0         # trades closed by the 10%/20% loss cap

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "gap_type": self.gap_type,
            "win_rate_pct": self.win_rate_pct,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "worst_trade_pct": self.worst_trade_pct,
            "loss_cap_exits": self.loss_cap_exits,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trade_count": self.trade_count,
        }


def _compute_metrics(
    trades: list[WeekendSwingTrade], strategy_name: str, gap_type: Optional[str] = None
) -> WeekendSwingBacktestMetrics:
    if not trades:
        return WeekendSwingBacktestMetrics(
            strategy=strategy_name, win_rate_pct=0.0, avg_win_pct=0.0,
            avg_loss_pct=0.0, max_drawdown_pct=0.0, trade_count=0, gap_type=gap_type,
        )

    returns_pct = [t.return_pct for t in trades]
    wins = [r for r in returns_pct if r > 0]
    losses = [r for r in returns_pct if r <= 0]

    fractional_returns = pd.Series([r / 100 for r in returns_pct])
    try:
        max_dd = abs(float(fractional_returns.vbt.returns.max_drawdown())) * 100
    except Exception as exc:
        logger.debug("vectorbt max_drawdown accessor failed (%s), falling back to manual calc", exc)
        cum = (1 + fractional_returns).cumprod()
        running_max = cum.cummax()
        max_dd = float(((running_max - cum) / running_max).max()) * 100 if len(cum) else 0.0

    return WeekendSwingBacktestMetrics(
        strategy=strategy_name,
        win_rate_pct=round(len(wins) / len(returns_pct) * 100, 2),
        avg_win_pct=round(sum(wins) / len(wins), 2) if wins else 0.0,
        avg_loss_pct=round(sum(losses) / len(losses), 2) if losses else 0.0,
        max_drawdown_pct=round(max_dd, 2),
        trade_count=len(trades),
        gap_type=gap_type,
        worst_trade_pct=round(min(returns_pct), 2),
        loss_cap_exits=sum(1 for t in trades if t.exit_reason.startswith("loss_cap")),
    )


def _simulate_exit(
    direction: str,
    entry_price: float,
    stop_price: float,
    entry_time,
    entry_day_intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    loss_cap: Optional[LossCapRule] = None,
) -> tuple:
    """
    Walks forward from entry: checks the rest of the entry day's intraday
    bars for a stop touch, then daily bars through week_end (Friday).
    Returns (exit_price, exit_reason, exit_time).

    When `loss_cap` is set it runs on each daily close after the stop check
    (it never loosens the strategy's own stop). "Setup still shows a chance
    of gain" here means buyers stepped in that day: the daily bar closed
    above its open for a long (below its open for a short). With today's tight
    ATR-based stops the stop almost always fires long before a 10% loss, so
    in practice the cap only matters if stops are widened or removed.
    """
    remaining_today = entry_day_intraday_df.loc[entry_day_intraday_df.index > entry_time]
    for idx, row in remaining_today.iterrows():
        if direction == "long" and float(row["low"]) <= stop_price:
            return stop_price, "stopped_out", idx
        if direction == "short" and float(row["high"]) >= stop_price:
            return stop_price, "stopped_out", idx

    # Monday after entry is covered by the intraday walk above. Monday's daily bar is
    # skipped: its low/high includes prices from before the entry.
    dates = bar_dates(daily_df)
    rest_of_week = daily_df.loc[(dates > calendar_day(week_start)) & (dates <= calendar_day(week_end))]
    exit_price, exit_time = entry_price, entry_time
    cap_state = loss_cap.new_trade() if loss_cap is not None else None
    for idx, row in rest_of_week.iterrows():
        if direction == "long" and float(row["low"]) <= stop_price:
            return stop_price, "stopped_out", idx
        if direction == "short" and float(row["high"]) >= stop_price:
            return stop_price, "stopped_out", idx
        close = float(row["close"])
        if cap_state is not None:
            day_open = float(row["open"])
            buyers_in = close > day_open if direction == "long" else close < day_open
            reason = cap_state.update(loss_pct_from_entry(direction, entry_price, close), buyers_in)
            if reason is not None:
                return close, reason, idx
        exit_price, exit_time = close, idx

    return exit_price, "friday_close", exit_time


def _return_pct(direction: str, entry_price: float, exit_price: float) -> float:
    raw = (exit_price - entry_price) / entry_price * 100
    return raw if direction == "long" else -raw


def _mondays_between(start: pd.Timestamp, end: pd.Timestamp) -> list:
    all_days = pd.date_range(start, end, freq="D")
    return [d for d in all_days if d.weekday() == 0]


class WeekendSwingBacktester:
    """Runs each strategy's real scan()/check_trigger() logic against cached historical bars."""

    def __init__(self, config: dict, bar_store, finnhub_client=None, regime_provider=None) -> None:
        self.config = config
        self.bar_store = bar_store
        self.finnhub_client = finnhub_client
        # Optional market_regime.RegimeProvider. When set, every trade is tagged with the
        # regime of its Monday so results can be sliced by fear/greed and pressure.
        self.regime_provider = regime_provider
        self._earnings_cache: dict = {}
        self.loss_cap = LossCapRule.from_config(config.get("loss_cap"))

    def _regime_for(self, week_start: pd.Timestamp) -> Optional[dict]:
        if self.regime_provider is None:
            return None
        regime = self.regime_provider.as_of(week_start)
        return regime.to_dict() if regime else None

    def _earnings_excluded_for_week(self, week_start: pd.Timestamp, week_end: pd.Timestamp) -> set:
        key = week_start.date()
        if key not in self._earnings_cache:
            self._earnings_cache[key] = get_earnings_exclusions(self.finnhub_client, week_start, week_end)
        return self._earnings_cache[key]

    # ── Strategy 1 ────────────────────────────────────────────────────────

    def run_strategy_1(self, tickers: list, start: pd.Timestamp, end: pd.Timestamp) -> WeekendSwingBacktestMetrics:
        return _compute_metrics(self.collect_strategy_1_trades(tickers, start, end), "prior_week_breakout")

    def collect_strategy_1_trades(self, tickers: list, start: pd.Timestamp, end: pd.Timestamp) -> list:
        """Raw per-trade results (for diagnostics/analysis) — run_strategy_1() aggregates these into metrics."""
        from signal_engine.weekend_swing.weekend_swing_signals import PriorWeekBreakoutStrategy

        strategy = PriorWeekBreakoutStrategy(self.config)
        mondays = _mondays_between(start, end)
        trades = []

        for ticker in tickers:
            daily_df = self.bar_store.get_bars(ticker, "1Day", start - pd.Timedelta(days=60), end)
            intraday_15 = self.bar_store.get_bars(ticker, "15Min", start - pd.Timedelta(days=40), end)
            if daily_df.empty or intraday_15.empty:
                logger.info("Strategy 1: skipping %s — insufficient cached data", ticker)
                continue

            for week_start in mondays:
                week_end = week_start + pd.Timedelta(days=4)
                historical_slice = intraday_15.loc[intraday_15.index < week_start]
                earnings_excluded = self._earnings_excluded_for_week(week_start, week_end)

                levels = strategy.scan(ticker, daily_df, week_start, historical_slice, earnings_excluded)
                if levels is None:
                    continue

                monday_bars = monday_session_bars(intraday_15, week_start)
                if monday_bars.empty:
                    continue

                signal = strategy.check_trigger(levels, monday_bars)
                if signal is None:
                    continue

                exit_price, exit_reason, exit_time = _simulate_exit(
                    signal.direction, signal.entry_price, signal.stop_price, signal.date,
                    monday_bars, daily_df, week_start, week_end, loss_cap=self.loss_cap,
                )
                trades.append(WeekendSwingTrade(
                    ticker=ticker, strategy="prior_week_breakout", direction=signal.direction,
                    entry_time=signal.date, entry_price=signal.entry_price, stop_price=signal.stop_price,
                    exit_time=exit_time, exit_price=exit_price, exit_reason=exit_reason,
                    return_pct=round(_return_pct(signal.direction, signal.entry_price, exit_price), 4),
                    breakdown=signal.breakdown, regime=self._regime_for(week_start),
                ))

        return trades

    # ── Strategy 2 ────────────────────────────────────────────────────────

    def run_strategy_2(self, tickers: list, start: pd.Timestamp, end: pd.Timestamp) -> WeekendSwingBacktestMetrics:
        return _compute_metrics(self.collect_strategy_2_trades(tickers, start, end), "overnight_range_breakout")

    def collect_strategy_2_trades(self, tickers: list, start: pd.Timestamp, end: pd.Timestamp) -> list:
        """Raw per-trade results (for diagnostics/analysis) — run_strategy_2() aggregates these into metrics."""
        from signal_engine.weekend_swing.weekend_swing_signals import OvernightRangeBreakoutStrategy

        strategy = OvernightRangeBreakoutStrategy(self.config)
        mondays = _mondays_between(start, end)
        trades = []

        for ticker in tickers:
            daily_df = self.bar_store.get_bars(ticker, "1Day", start - pd.Timedelta(days=60), end)
            intraday_5 = self.bar_store.get_bars(ticker, "5Min", start - pd.Timedelta(days=5), end)
            if daily_df.empty or intraday_5.empty:
                logger.info("Strategy 2: skipping %s — insufficient cached data", ticker)
                continue

            for week_start in mondays:
                week_end = week_start + pd.Timedelta(days=4)
                levels = strategy.scan(ticker, daily_df, week_start)
                if levels is None:
                    continue

                monday_bars = monday_session_bars(intraday_5, week_start)
                if monday_bars.empty:
                    continue
                session_open = session_open_time(week_start, monday_bars.index)

                signal = strategy.check_trigger(levels, monday_bars, session_open)
                if signal is None:
                    continue

                exit_price, exit_reason, exit_time = _simulate_exit(
                    signal.direction, signal.entry_price, signal.stop_price, signal.date,
                    monday_bars, daily_df, week_start, week_end, loss_cap=self.loss_cap,
                )
                trades.append(WeekendSwingTrade(
                    ticker=ticker, strategy="overnight_range_breakout", direction=signal.direction,
                    entry_time=signal.date, entry_price=signal.entry_price, stop_price=signal.stop_price,
                    exit_time=exit_time, exit_price=exit_price, exit_reason=exit_reason,
                    return_pct=round(_return_pct(signal.direction, signal.entry_price, exit_price), 4),
                    breakdown=signal.breakdown, regime=self._regime_for(week_start),
                ))

        return trades

    # ── Strategy 3 ────────────────────────────────────────────────────────

    def run_strategy_3(
        self, tickers: list, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple:
        """Returns (gap_and_go_metrics, gap_and_fail_metrics) — never combined."""
        go_trades, fail_trades = self.collect_strategy_3_trades(tickers, start, end)
        return (
            _compute_metrics(go_trades, "gap_strength", gap_type="gap_and_go"),
            _compute_metrics(fail_trades, "gap_strength", gap_type="gap_and_fail"),
        )

    def collect_strategy_3_trades(self, tickers: list, start: pd.Timestamp, end: pd.Timestamp) -> tuple:
        """Raw (go_trades, fail_trades) — run_strategy_3() aggregates these into metrics."""
        from signal_engine.weekend_swing.weekend_swing_signals import GapStrengthStrategy

        strategy = GapStrengthStrategy(self.config)
        mondays = _mondays_between(start, end)
        go_trades, fail_trades = [], []

        for ticker in tickers:
            if ticker.upper() not in strategy.allowed_tickers:
                continue

            daily_df = self.bar_store.get_bars(ticker, "1Day", start - pd.Timedelta(days=60), end)
            intraday_5 = self.bar_store.get_bars(ticker, "5Min", start - pd.Timedelta(days=5), end)
            if daily_df.empty or intraday_5.empty:
                logger.info("Strategy 3: skipping %s — insufficient cached data", ticker)
                continue

            for week_start in mondays:
                week_end = week_start + pd.Timedelta(days=4)
                levels = strategy.scan(ticker, daily_df, week_start)
                if levels is None:
                    continue

                monday_bars = monday_session_bars(intraday_5, week_start)
                if monday_bars.empty:
                    continue
                monday_daily_row = daily_df.loc[bar_dates(daily_df) >= calendar_day(week_start)]
                if monday_daily_row.empty:
                    continue
                monday_open = float(monday_daily_row.iloc[0]["open"])
                first_bar = monday_bars.iloc[0]

                signal = strategy.check_trigger(levels, monday_open, first_bar)
                if signal is None:
                    continue

                exit_price, exit_reason, exit_time = _simulate_exit(
                    signal.direction, signal.entry_price, signal.stop_price, signal.date,
                    monday_bars, daily_df, week_start, week_end, loss_cap=self.loss_cap,
                )
                trade = WeekendSwingTrade(
                    ticker=ticker, strategy="gap_strength", direction=signal.direction,
                    entry_time=signal.date, entry_price=signal.entry_price, stop_price=signal.stop_price,
                    exit_time=exit_time, exit_price=exit_price, exit_reason=exit_reason,
                    return_pct=round(_return_pct(signal.direction, signal.entry_price, exit_price), 4),
                    gap_type=signal.breakdown.get("gap_type"),
                    breakdown=signal.breakdown, regime=self._regime_for(week_start),
                )
                if trade.gap_type == "gap_and_go":
                    go_trades.append(trade)
                else:
                    fail_trades.append(trade)

        return go_trades, fail_trades
