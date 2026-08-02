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
    ) -> "BacktestResult":
        """
        Run a portfolio backtest.

        Args:
            price_data: Dict of ticker → OHLCV DataFrame (DatetimeIndex)
            signals:    Dict of ticker → boolean buy signal Series

        Returns:
            BacktestResult with portfolio and metrics
        """
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
    ) -> None:
        self.portfolio = portfolio
        self.config = config
        self.close_df = close_df

    def get_metrics(self) -> BacktestMetrics:
        """Compute and return BacktestMetrics from the vectorbt portfolio."""
        pf = self.portfolio

        try:
            total_return = float(pf.total_return())
        except Exception:
            total_return = 0.0

        try:
            trades = pf.trades
            trade_records = trades.records_readable
            total_trades = len(trade_records)
            if total_trades > 0:
                winning_trades = int((trade_records["PnL"] > 0).sum())
                win_rate = winning_trades / total_trades * 100
            else:
                winning_trades = 0
                win_rate = 0.0
        except Exception as exc:
            logger.warning("Could not compute trade stats: %s", exc)
            total_trades = 0
            winning_trades = 0
            win_rate = 0.0

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
