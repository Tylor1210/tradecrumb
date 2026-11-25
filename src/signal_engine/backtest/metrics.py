"""
Backtest metrics computation and reporting.

Win rate definition (per user requirement):
  win_rate = profitable_trades / total_trades
  A trade is profitable if exit_price > entry_price (even +$0.01 counts).

Target: ≥70% win rate before signals are trusted for paper trading.
The output clearly shows ✅ or ❌ against this bar.
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class BacktestMetrics:
    """
    Full metrics report for a single backtest run.

    Win rate = profitable trades / total trades.
    Even +0.01% profit = a winning trade.
    """

    total_return_pct: float         # total portfolio return %
    win_rate_pct: float             # profitable trades / total trades * 100
    total_trades: int
    winning_trades: int
    max_drawdown_pct: float         # worst peak-to-trough on equity curve
    sharpe_ratio: float             # annualized Sharpe
    directional_accuracy_pct: float # % signals where price moved in predicted direction
    avg_loss_pct: float = 0.0       # mean return of losing trades (negative)
    worst_trade_pct: float = 0.0    # single worst trade return (negative)
    avg_trade_pct: float = 0.0      # mean return per trade (a partly sold trade counts once, blended)

    WIN_RATE_TARGET = 70.0          # class constant: the 70% bar

    @property
    def meets_win_rate_target(self) -> bool:
        """True if win rate ≥ 70% (ready to consider paper trading)."""
        return self.win_rate_pct >= self.WIN_RATE_TARGET

    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.winning_trades

    def to_dict(self) -> dict:
        return {
            "total_return_pct": self.total_return_pct,
            "win_rate_pct": self.win_rate_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "directional_accuracy_pct": self.directional_accuracy_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "worst_trade_pct": self.worst_trade_pct,
            "avg_trade_pct": self.avg_trade_pct,
            "meets_70pct_target": self.meets_win_rate_target,
        }

    def __str__(self) -> str:
        target_icon = "✅" if self.meets_win_rate_target else "❌"
        dir_acc = f"{self.directional_accuracy_pct:.1f}%" if self.directional_accuracy_pct > 0 else "N/A"

        return (
            f"\n{'═' * 52}\n"
            f"  BACKTEST RESULTS\n"
            f"{'═' * 52}\n"
            f"  Total Return:        {self.total_return_pct:+.2f}%\n"
            f"  Win Rate:            {self.win_rate_pct:.1f}%  {target_icon}  (target: ≥{self.WIN_RATE_TARGET:.0f}%)\n"
            f"  Total Trades:        {self.total_trades}\n"
            f"  Winning Trades:      {self.winning_trades}\n"
            f"  Losing Trades:       {self.losing_trades}\n"
            f"  Avg Losing Trade:    {self.avg_loss_pct:+.2f}%\n"
            f"  Worst Trade:         {self.worst_trade_pct:+.2f}%\n"
            f"  Max Drawdown:        {self.max_drawdown_pct:.2f}%\n"
            f"  Sharpe Ratio:        {self.sharpe_ratio:.3f}\n"
            f"  Dir. Accuracy:       {dir_acc}\n"
            f"{'═' * 52}\n"
            f"  {'✅ READY for paper trading consideration' if self.meets_win_rate_target else '❌ NOT ready — win rate below 70% target'}\n"
            f"{'═' * 52}\n"
        )


def compute_directional_accuracy(
    price_data: pd.DataFrame,
    signals: pd.Series,
    hold_period_days: int,
) -> float:
    """
    Compute directional accuracy of a signal series.

    Definition: % of buy signals where price rose within hold_period_days.
    A signal of True = BUY (expecting price to go up).

    Args:
        price_data:       DataFrame with 'close' column
        signals:          Boolean Series (True = buy signal on that date)
        hold_period_days: Lookahead window in trading days

    Returns:
        Directional accuracy as a percentage (0-100).
    """
    buy_dates = signals[signals].index
    if len(buy_dates) == 0:
        return 0.0

    closes = price_data["close"]
    correct = 0
    total = 0

    for date in buy_dates:
        try:
            pos = closes.index.get_loc(date)
            future_pos = pos + hold_period_days
            if future_pos >= len(closes):
                continue  # not enough future data
            entry = float(closes.iloc[pos])
            exit_price = float(closes.iloc[future_pos])
            if exit_price > entry:  # even +$0.01 = correct
                correct += 1
            total += 1
        except (KeyError, IndexError):
            continue

    return round(correct / total * 100, 2) if total > 0 else 0.0
