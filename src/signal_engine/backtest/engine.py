"""
vectorbt-based backtesting engine.

Converts signal engine output into vectorbt Portfolio simulations.
Reports win rate (profitable trades / total), drawdown, and Sharpe ratio.

Win rate target: ≥70% before activating paper trading.
Win = any profitable trade, even +0.01%.

vectorbt note: requires numba/LLVM for JIT compilation.
If install issues occur: pip install llvmlite --pre before uv sync.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from signal_engine.backtest.metrics import BacktestMetrics, compute_directional_accuracy
from signal_engine.risk.loss_cap import LossCapRule
from signal_engine.risk.scale_out import ScaleOutRule, ScaleOutStats, atr_series, bar_move_in_atr

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    start_date: str         # "YYYY-MM-DD"
    end_date: str           # "YYYY-MM-DD"
    hold_period_days: int   # 1=intraday, 5=swing, 21=monthly
    stop_loss_pct: float    # e.g. 5.0 = 5%
    take_profit_pct: float  # e.g. 10.0 = 10%
    initial_capital: float  # e.g. 100000
    position_size_pct: float = 5.0  # % of portfolio per trade
    # Optional 10%/20% per-trade loss cap (see risk/loss_cap.py). None keeps
    # the original vectorbt sl_stop/tp_stop path exactly as it was.
    loss_cap: Optional[LossCapRule] = None
    # Loss cap "setup still shows a chance of gain" test for this engine:
    # the same technical score used for entries is at or above this floor on
    # that bar (0.0 = not bearish). Used only when a trade is 10%+ down.
    setup_score_floor: float = 0.0
    # Optional "sell half at +20%, let the rest run" rule (see risk/scale_out.py).
    # None keeps every existing path exactly as it was.
    scale_out: Optional[ScaleOutRule] = None
    # Scale-out "setup still points to continuation" test: technical score at
    # or above this floor on that bar. Used only once a trade is at the trigger.
    continuation_score_floor: float = 0.0


def walk_exits(
    close: pd.Series,
    entries: pd.Series,
    setup_alive: pd.Series,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """
    Event walk for one ticker, used when a loss cap is active.

    Same position model as the vectorbt path (one position per ticker, enter
    on a signal bar's close, no pyramiding, fills on closes) but with exits
    decided bar by bar so the stateful loss cap can run:

      take profit -> flat stop (unless the loss cap replaces it)
      -> loss cap -> time exit after hold_period_days bars

    Returns (entries_taken, exits, exit_reasons). entries_taken only marks
    signals that actually opened a position, so vectorbt replays exactly the
    trades this walk made. exits only marks full closes; use walk_orders()
    when a scale-out rule can sell part of a position.
    """
    entries_taken, sells, reasons, _stats = walk_orders(close, entries, setup_alive, config)
    exits = (sells >= 1.0).fillna(False).astype(bool)
    return entries_taken, exits, reasons


def walk_orders(
    close: pd.Series,
    entries: pd.Series,
    setup_alive: pd.Series,
    config: BacktestConfig,
    continuation: Optional[pd.Series] = None,
    atr: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series, list[str], ScaleOutStats]:
    """
    Event walk for one ticker that can also sell part of a position.

    Order of checks on each close while a position is open:

      runner/fast-move position (scale-out owns the exit once triggered)
      otherwise: take profit (unless scale-out replaces it) -> flat stop
      (unless the loss cap replaces it) -> scale-out trigger -> loss cap
      -> time exit after hold_period_days bars (a runner ignores it when
      runner_ignores_time_exit is on; runner_max_bars caps it instead)

    Returns (entries_taken, sell_fractions, reasons, scale_out_stats).
    sell_fractions is NaN on bars with no sale, otherwise the fraction of the
    position still held that is sold on that close (1.0 = close it all).
    """
    rule = config.loss_cap
    so_rule = config.scale_out
    tp = config.take_profit_pct
    if so_rule is not None and so_rule.replaces_take_profit:
        tp = None
    sl = None if (rule is not None and rule.replaces_fixed_stop) else config.stop_loss_pct
    hold = config.hold_period_days

    entries_taken = pd.Series(False, index=close.index)
    sells = pd.Series(np.nan, index=close.index)
    reasons: list[str] = []
    stats = ScaleOutStats()

    in_pos = False
    entry_price = 0.0
    entry_i = 0
    state = None
    so_state = None
    prev_price = None

    values = close.to_numpy(dtype=float)
    ent = entries.reindex(close.index).fillna(False).astype(bool).to_numpy()
    alive = setup_alive.reindex(close.index).fillna(False).astype(bool).to_numpy()
    cont = (
        continuation.reindex(close.index).fillna(False).astype(bool).to_numpy()
        if continuation is not None else alive
    )
    atr_vals = atr.reindex(close.index).to_numpy(dtype=float) if atr is not None else None

    for i, price in enumerate(values):
        if np.isnan(price):
            continue
        if not in_pos:
            if ent[i]:
                in_pos = True
                entry_price = price
                entry_i = i
                state = rule.new_trade() if rule is not None else None
                so_state = so_rule.new_trade() if so_rule is not None else None
                entries_taken.iloc[i] = True
            prev_price = price
            continue

        gain_pct = (price - entry_price) / entry_price * 100
        loss_pct = -gain_pct
        prev_atr = atr_vals[i - 1] if atr_vals is not None and i > 0 else None
        move_atr = bar_move_in_atr("long", price, prev_price, prev_atr)
        prev_price = price

        sell_fraction = None
        reason = None

        if so_state is not None and so_state.managing:
            action = so_state.update(gain_pct, bool(cont[i]), move_atr)
            if action is not None:
                sell_fraction, reason = action.sell_fraction, action.reason
            elif not so_rule.runner_ignores_time_exit and i - entry_i >= hold:
                sell_fraction, reason = 1.0, "time_exit"
        else:
            if tp and gain_pct >= tp:
                reason = "take_profit"
            elif sl and loss_pct >= sl:
                reason = "stop_loss"
            else:
                if so_state is not None:
                    action = so_state.update(gain_pct, bool(cont[i]), move_atr)
                    if action is not None:
                        sell_fraction, reason = action.sell_fraction, action.reason
                if reason is None and so_state is not None and so_state.managing:
                    pass  # fast move just started: the rule owns the exit from here
                elif reason is None and state is not None:
                    reason = state.update(loss_pct, bool(alive[i]))
            if reason is not None and sell_fraction is None:
                sell_fraction = 1.0
            if reason is None and i - entry_i >= hold and not (
                so_state is not None and so_state.managing and so_rule.runner_ignores_time_exit
            ):
                sell_fraction, reason = 1.0, "time_exit"

        if reason is not None:
            sells.iloc[i] = sell_fraction
            reasons.append(reason)
            if sell_fraction >= 1.0:
                in_pos = False
                stats.add(so_state)

    if in_pos:
        reasons.append("open_at_end")
        stats.add(so_state)

    return entries_taken, sells, reasons, stats


class BacktestEngine:
    """
    Runs portfolio backtests using vectorbt.

    Input:
      - Price DataFrames (OHLCV per ticker, daily)
      - Signal Series (boolean, True = buy signal on that date)

    Output:
      - BacktestResult with equity curve + metrics
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._check_vectorbt()

    def _check_vectorbt(self) -> None:
        try:
            import vectorbt as vbt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "vectorbt is required for backtesting.\n"
                "Install with: uv add vectorbt\n"
                "If LLVM errors occur: pip install llvmlite --pre first."
            ) from exc

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],  # ticker → bool Series (True = buy)
        setup_scores: Optional[dict[str, pd.Series]] = None,
    ) -> "BacktestResult":
        """
        Run a portfolio backtest.

        Args:
            price_data:   Dict of ticker → OHLCV DataFrame (DatetimeIndex)
            signals:      Dict of ticker → boolean buy signal Series
            setup_scores: Optional dict of ticker → per-bar technical score.
                          Only used by the loss cap to judge whether a losing
                          trade's setup is still alive. Without it the loss
                          cap falls back to "the bar closed above its open"
                          (buyers stepped in), or above the prior close if
                          there's no open column.

        Returns:
            BacktestResult with portfolio and metrics
        """
        if self.config.scale_out is not None:
            return self._run_with_scale_out(price_data, signals, setup_scores or {})
        if self.config.loss_cap is not None:
            return self._run_with_loss_cap(price_data, signals, setup_scores or {})

        import vectorbt as vbt

        # Build aligned close price DataFrame and signal matrices
        closes: dict[str, pd.Series] = {}
        entries: dict[str, pd.Series] = {}
        exits: dict[str, pd.Series] = {}

        for ticker, df in price_data.items():
            if ticker not in signals:
                continue

            close_series = df["close"].astype(float)
            sig = signals[ticker].reindex(close_series.index).fillna(False).astype(bool)

            closes[ticker] = close_series
            entries[ticker] = sig

            # Exit signal: hold for N days after entry
            # vectorbt handles exits via sl_stop/tp_stop; we also set a time exit
            exit_sig = sig.shift(self.config.hold_period_days).fillna(False).astype(bool)
            exits[ticker] = exit_sig

        if not closes:
            raise ValueError("No overlapping tickers between price_data and signals")

        # Align all series to common date index
        close_df = pd.DataFrame(closes).sort_index()
        close_df = close_df.ffill()

        entries_df = pd.DataFrame(entries, index=close_df.index).fillna(False)
        exits_df = pd.DataFrame(exits, index=close_df.index).fillna(False)

        logger.info(
            "Running backtest: %d tickers, %d bars, hold=%dd",
            len(closes),
            len(close_df),
            self.config.hold_period_days,
        )

        # Size per position = fixed % of initial capital
        size = self.config.initial_capital * (self.config.position_size_pct / 100)

        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            sl_stop=self.config.stop_loss_pct / 100,
            tp_stop=self.config.take_profit_pct / 100,
            size=size,
            size_type="value",
            init_cash=self.config.initial_capital,
            fees=0.001,          # 0.1% per-side commission approximation
            freq="D",
        )

        return BacktestResult(portfolio=pf, config=self.config, close_df=close_df)

    def _run_with_loss_cap(
        self,
        price_data: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],
        setup_scores: dict[str, pd.Series],
    ) -> "BacktestResult":
        import vectorbt as vbt

        closes: dict[str, pd.Series] = {}
        raw_entries: dict[str, pd.Series] = {}
        for ticker, df in price_data.items():
            if ticker not in signals:
                continue
            close_series = df["close"].astype(float)
            closes[ticker] = close_series
            raw_entries[ticker] = signals[ticker].reindex(close_series.index).fillna(False).astype(bool)

        if not closes:
            raise ValueError("No overlapping tickers between price_data and signals")

        close_df = pd.DataFrame(closes).sort_index().ffill()

        entries: dict[str, pd.Series] = {}
        exits: dict[str, pd.Series] = {}
        exit_reasons: dict[str, int] = {}
        for ticker in close_df.columns:
            close = close_df[ticker]
            alive = _setup_alive(
                close, price_data[ticker], setup_scores.get(ticker), self.config.setup_score_floor
            )
            ent, ex, reasons = walk_exits(close, raw_entries[ticker], alive, self.config)
            entries[ticker] = ent
            exits[ticker] = ex
            for r in reasons:
                exit_reasons[r] = exit_reasons.get(r, 0) + 1

        entries_df = pd.DataFrame(entries, index=close_df.index).fillna(False)
        exits_df = pd.DataFrame(exits, index=close_df.index).fillna(False)

        logger.info(
            "Running loss-cap backtest: %d tickers, %d bars, hold=%dd, exits=%s",
            len(closes), len(close_df), self.config.hold_period_days, exit_reasons,
        )

        size = self.config.initial_capital * (self.config.position_size_pct / 100)
        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            size=size,
            size_type="value",
            init_cash=self.config.initial_capital,
            fees=0.001,
            freq="D",
        )

        return BacktestResult(
            portfolio=pf, config=self.config, close_df=close_df, exit_reasons=exit_reasons
        )


    def _run_with_scale_out(
        self,
        price_data: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],
        setup_scores: dict[str, pd.Series],
    ) -> "BacktestResult":
        """
        Same position model as the loss-cap path, but a position can be sold
        in parts, so orders go through vectorbt's from_orders and results are
        counted per position (a trade sold in two parts is still one trade,
        with its blended return) rather than per exit fill.
        """
        import vectorbt as vbt
        from vectorbt.portfolio.enums import SizeType

        closes: dict[str, pd.Series] = {}
        raw_entries: dict[str, pd.Series] = {}
        for ticker, df in price_data.items():
            if ticker not in signals:
                continue
            close_series = df["close"].astype(float)
            closes[ticker] = close_series
            raw_entries[ticker] = signals[ticker].reindex(close_series.index).fillna(False).astype(bool)

        if not closes:
            raise ValueError("No overlapping tickers between price_data and signals")

        close_df = pd.DataFrame(closes).sort_index().ffill()
        position_value = self.config.initial_capital * (self.config.position_size_pct / 100)

        sizes: dict[str, pd.Series] = {}
        size_types: dict[str, pd.Series] = {}
        exit_reasons: dict[str, int] = {}
        stats = ScaleOutStats()
        for ticker in close_df.columns:
            close = close_df[ticker]
            df = price_data[ticker]
            scores = setup_scores.get(ticker)
            alive = _setup_alive(close, df, scores, self.config.setup_score_floor)
            continuation = _setup_alive(close, df, scores, self.config.continuation_score_floor)
            atr = atr_series(df)
            atr = atr.reindex(close.index) if atr is not None else None
            ent, sells, reasons, t_stats = walk_orders(
                close, raw_entries[ticker], alive, self.config, continuation=continuation, atr=atr
            )
            size = pd.Series(np.nan, index=close.index)
            stype = pd.Series(int(SizeType.Amount), index=close.index)
            size[ent] = position_value
            stype[ent] = int(SizeType.Value)
            sold = sells.notna()
            # Percent on a sell = that fraction of the shares currently held.
            size[sold] = -sells[sold]
            stype[sold] = int(SizeType.Percent)
            sizes[ticker] = size
            size_types[ticker] = stype
            stats = stats.merge(t_stats)
            for r in reasons:
                exit_reasons[r] = exit_reasons.get(r, 0) + 1

        size_df = pd.DataFrame(sizes, index=close_df.index)
        size_type_df = pd.DataFrame(size_types, index=close_df.index)

        logger.info(
            "Running scale-out backtest: %d tickers, %d bars, exits=%s, scale-out=%s",
            len(closes), len(close_df), exit_reasons, stats.to_dict(),
        )

        pf = vbt.Portfolio.from_orders(
            close=close_df,
            size=size_df,
            size_type=size_type_df,
            direction="longonly",
            init_cash=self.config.initial_capital,
            fees=0.001,
            freq="D",
        )

        return BacktestResult(
            portfolio=pf, config=self.config, close_df=close_df, exit_reasons=exit_reasons,
            scale_out_stats=stats, count_by_position=True,
        )


def _setup_alive(
    close: pd.Series, df: pd.DataFrame, scores: Optional[pd.Series], floor: float
) -> pd.Series:
    """Per-bar yes/no: technical score at or above floor, else the bar closed above its open."""
    if scores is not None:
        return (scores.reindex(close.index) >= floor).fillna(False)
    if "open" in df.columns:
        # Fallback: buyers stepped in that day (bar closed above its open).
        day_open = df["open"].astype(float).reindex(close.index)
        return (close > day_open).fillna(False)
    return (close > close.shift(1)).fillna(False)


class BacktestResult:
    """
    Result of a backtest run.
    Wraps a vectorbt Portfolio with convenience accessors.
    """

    def __init__(
        self,
        portfolio,           # vbt.Portfolio
        config: BacktestConfig,
        close_df: pd.DataFrame,
        exit_reasons: Optional[dict[str, int]] = None,
        scale_out_stats: Optional[ScaleOutStats] = None,
        count_by_position: bool = False,
    ) -> None:
        self.portfolio = portfolio
        self.config = config
        self.close_df = close_df
        # Only filled on the event-walk paths: how many closes/sales happened for each reason.
        self.exit_reasons = exit_reasons or {}
        # Only filled on the scale-out path: how many trades reached the trigger, and what happened.
        self.scale_out_stats = scale_out_stats
        # Scale-out sells in parts. vectorbt's `trades` would count each partial
        # sale as its own trade, so win rate is counted per position instead.
        self.count_by_position = count_by_position

    def get_metrics(self) -> BacktestMetrics:
        """Compute and return BacktestMetrics from the vectorbt portfolio."""
        pf = self.portfolio

        try:
            total_return = float(pf.total_return())
        except Exception:
            total_return = 0.0

        try:
            trades = pf.positions if self.count_by_position else pf.trades
            trade_records = trades.records_readable
            total_trades = len(trade_records)
            if total_trades > 0:
                winning_trades = int((trade_records["PnL"] > 0).sum())
                win_rate = winning_trades / total_trades * 100
                trade_returns = trade_records["Return"].astype(float) * 100
                losers = trade_returns[trade_records["PnL"] <= 0]
                avg_loss = float(losers.mean()) if len(losers) else 0.0
                worst_trade = float(trade_returns.min())
                avg_trade = float(trade_returns.mean())
            else:
                winning_trades = 0
                win_rate = 0.0
                avg_loss = 0.0
                worst_trade = 0.0
                avg_trade = 0.0
        except Exception as exc:
            logger.warning("Could not compute trade stats: %s", exc)
            total_trades = 0
            winning_trades = 0
            win_rate = 0.0
            avg_loss = 0.0
            worst_trade = 0.0
            avg_trade = 0.0

        try:
            max_dd = float(pf.max_drawdown() * 100)
        except Exception:
            max_dd = 0.0

        try:
            sharpe = float(pf.sharpe_ratio())
        except Exception:
            sharpe = 0.0

        return BacktestMetrics(
            total_return_pct=round(total_return * 100, 2),
            win_rate_pct=round(win_rate, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            directional_accuracy_pct=0.0,  # computed by grid_search
            avg_loss_pct=round(avg_loss, 2),
            worst_trade_pct=round(worst_trade, 2),
            avg_trade_pct=round(avg_trade, 2),
        )

    def save_equity_curve(self, output_path: Path) -> None:
        """Save an interactive HTML equity curve to output_path."""
        try:
            fig = self.portfolio.plot()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info("Equity curve saved to %s", output_path)
        except Exception as exc:
            logger.warning("Could not save equity curve: %s", exc)
