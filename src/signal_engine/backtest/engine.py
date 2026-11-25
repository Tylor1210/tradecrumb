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

from signal_engine.backtest.account import (
    AccountResult,
    AccountSizing,
    simulate_account,
    trades_from_portfolio,
)
from signal_engine.backtest.metrics import BacktestMetrics, compute_directional_accuracy
from signal_engine.risk.loss_cap import LossCapRule

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
    # Account sizing (see backtest/account.py): each new trade gets this % of
    # current account equity, at most max_positions open at once.
    position_size_pct: float = 10.0
    max_positions: int = 10
    # Optional 10%/20% per-trade loss cap (see risk/loss_cap.py). None keeps
    # the original vectorbt sl_stop/tp_stop path exactly as it was.
    loss_cap: Optional[LossCapRule] = None
    # Loss cap "setup still shows a chance of gain" test for this engine:
    # the same technical score used for entries is at or above this floor on
    # that bar (0.0 = not bearish). Used only when a trade is 10%+ down.
    setup_score_floor: float = 0.0


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
    trades this walk made.
    """
    rule = config.loss_cap
    tp = config.take_profit_pct
    sl = None if (rule is not None and rule.replaces_fixed_stop) else config.stop_loss_pct
    hold = config.hold_period_days

    entries_taken = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    reasons: list[str] = []

    in_pos = False
    entry_price = 0.0
    entry_i = 0
    state = None

    values = close.to_numpy(dtype=float)
    ent = entries.reindex(close.index).fillna(False).astype(bool).to_numpy()
    alive = setup_alive.reindex(close.index).fillna(False).astype(bool).to_numpy()

    for i, price in enumerate(values):
        if np.isnan(price):
            continue
        if not in_pos:
            if ent[i]:
                in_pos = True
                entry_price = price
                entry_i = i
                state = rule.new_trade() if rule is not None else None
                entries_taken.iloc[i] = True
            continue

        loss_pct = (entry_price - price) / entry_price * 100
        reason = None
        if tp and -loss_pct >= tp:
            reason = "take_profit"
        elif sl and loss_pct >= sl:
            reason = "stop_loss"
        elif state is not None:
            reason = state.update(loss_pct, bool(alive[i]))
        if reason is None and i - entry_i >= hold:
            reason = "time_exit"

        if reason is not None:
            exits.iloc[i] = True
            reasons.append(reason)
            in_pos = False

    if in_pos:
        reasons.append("open_at_end")

    return entries_taken, exits, reasons


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
            if ticker in setup_scores:
                scores = setup_scores[ticker].reindex(close.index)
                alive = (scores >= self.config.setup_score_floor).fillna(False)
            elif "open" in price_data[ticker].columns:
                # Fallback: buyers stepped in that day (bar closed above its open).
                day_open = price_data[ticker]["open"].astype(float).reindex(close.index)
                alive = (close > day_open).fillna(False)
            else:
                alive = (close > close.shift(1)).fillna(False)
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
    ) -> None:
        self.portfolio = portfolio
        self.config = config
        self.close_df = close_df
        # Only filled on the loss-cap path: how many trades closed for each reason.
        self.exit_reasons = exit_reasons or {}
        self._account: Optional[AccountResult] = None

    def account(self) -> AccountResult:
        """
        Replay the trades through one shared account with the configured sizing.

        The vectorbt portfolio keeps separate cash per ticker, so its
        total_return/max_drawdown are per-ticker Series on multi-ticker runs.
        Account return, drawdown, Sharpe and weekly/monthly stats come from
        here instead.
        """
        if self._account is None:
            sizing = AccountSizing(
                position_size_pct=self.config.position_size_pct,
                max_positions=self.config.max_positions,
                fees_pct=0.1,  # same 0.1% per side as the vectorbt runs
            )
            self._account = simulate_account(
                trades_from_portfolio(self.portfolio),
                self.close_df,
                self.config.initial_capital,
                sizing,
            )
        return self._account

    def get_metrics(self) -> BacktestMetrics:
        """Compute BacktestMetrics: trade stats from every signal, account stats from one account."""
        pf = self.portfolio

        try:
            acct = self.account().stats()
        except Exception as exc:
            logger.warning("Could not compute account stats: %s", exc)
            acct = None

        try:
            trades = pf.trades
            trade_records = trades.records_readable
            total_trades = len(trade_records)
            if total_trades > 0:
                winning_trades = int((trade_records["PnL"] > 0).sum())
                win_rate = winning_trades / total_trades * 100
                trade_returns = trade_records["Return"].astype(float) * 100
                losers = trade_returns[trade_records["PnL"] <= 0]
                avg_loss = float(losers.mean()) if len(losers) else 0.0
                worst_trade = float(trade_returns.min())
            else:
                winning_trades = 0
                win_rate = 0.0
                avg_loss = 0.0
                worst_trade = 0.0
        except Exception as exc:
            logger.warning("Could not compute trade stats: %s", exc)
            total_trades = 0
            winning_trades = 0
            win_rate = 0.0
            avg_loss = 0.0
            worst_trade = 0.0

        total_return = acct.total_return_pct if acct else 0.0
        max_dd = acct.max_drawdown_pct if acct else 0.0
        sharpe = acct.sharpe_ratio if acct else 0.0

        return BacktestMetrics(
            total_return_pct=round(total_return, 2),
            win_rate_pct=round(win_rate, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            directional_accuracy_pct=0.0,  # computed by grid_search
            avg_loss_pct=round(avg_loss, 2),
            worst_trade_pct=round(worst_trade, 2),
            account=acct,
        )

    def save_equity_curve(self, output_path: Path) -> None:
        """Save an interactive HTML chart of the shared account's equity to output_path."""
        try:
            import plotly.graph_objects as go

            acct = self.account()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=acct.equity.index, y=acct.equity.values, name="Account equity"))
            fig.add_trace(go.Scatter(x=acct.invested.index, y=acct.invested.values, name="Invested"))
            fig.update_layout(title="Backtest account equity", yaxis_title="$")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info("Equity curve saved to %s", output_path)
        except Exception as exc:
            logger.warning("Could not save equity curve: %s", exc)
