"""
Paper trading daily report generator.

Produces Rich console output with:
  - Account summary (portfolio value, buying power)
  - Open positions table
  - Closed trade performance summary (win rate vs 70% target)
  - Top performers and worst trades

All output is to stdout (Rich console). 
Optionally saves to reports/paper/ directory.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from signal_engine.paper_trading.tracker import TradeTracker

logger = logging.getLogger(__name__)
console = Console()


class PaperTradingReporter:
    """
    Generates daily paper trading summary reports using Rich.
    """

    def __init__(self, tracker: TradeTracker) -> None:
        self.tracker = tracker

    def print_daily_summary(
        self,
        account_info: Optional[dict] = None,
        positions: Optional[list[dict]] = None,
    ) -> None:
        """
        Print the full daily paper trading summary to the console.
        """
        console.print()
        console.rule("[bold cyan]📊 Paper Trading Daily Summary[/bold cyan]")
        console.print(f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Paper trading only — no live money[/dim]\n")

        # ── Account Info ──────────────────────────────────────────────────
        if account_info:
            self._print_account_info(account_info)

        # ── Open Positions ────────────────────────────────────────────────
        open_trades = self.tracker.get_open_trades()
        if positions:
            self._print_live_positions(positions)
        elif open_trades:
            self._print_logged_open_trades(open_trades)
        else:
            console.print("[dim]No open positions.[/dim]\n")

        # ── Performance Summary ───────────────────────────────────────────
        summary = self.tracker.get_performance_summary()
        self._print_performance_summary(summary)

        # ── Recent Closed Trades ──────────────────────────────────────────
        closed = self.tracker.get_closed_trades()
        if closed:
            self._print_recent_closed_trades(closed[:10])

        console.print()

    def _print_account_info(self, account: dict) -> None:
        pv = account.get("portfolio_value", 0)
        bp = account.get("buying_power", 0)
        cash = account.get("cash", 0)
        status = account.get("status", "UNKNOWN")

        console.print(Panel(
            f"[bold]Portfolio Value:[/bold]  ${pv:>12,.2f}\n"
            f"[bold]Buying Power:[/bold]     ${bp:>12,.2f}\n"
            f"[bold]Cash:[/bold]             ${cash:>12,.2f}\n"
            f"[bold]Account Status:[/bold]   {status}\n"
            f"[bold]Mode:[/bold]             [yellow]PAPER TRADING[/yellow]",
            title="Account",
            expand=False,
        ))
        console.print()

    def _print_live_positions(self, positions: list[dict]) -> None:
        if not positions:
            console.print("[dim]No open positions in broker.[/dim]\n")
            return

        table = Table(title="Live Open Positions (from Alpaca Paper)", box=box.ROUNDED)
        table.add_column("Ticker", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Entry Price", justify="right")
        table.add_column("Current Price", justify="right")
        table.add_column("Unr. P&L", justify="right")
        table.add_column("Unr. P&L %", justify="right")
        table.add_column("Market Value", justify="right")

        for pos in positions:
            unr_pl = pos.get("unrealized_pl") or 0
            unr_plpc = (pos.get("unrealized_plpc") or 0) * 100
            pl_style = "green" if unr_pl >= 0 else "red"

            table.add_row(
                pos["symbol"],
                f"{pos['qty']:.4f}",
                f"${pos['avg_entry_price']:.2f}",
                f"${pos.get('current_price', 0):.2f}" if pos.get("current_price") else "N/A",
                f"[{pl_style}]${unr_pl:+,.2f}[/{pl_style}]",
                f"[{pl_style}]{unr_plpc:+.2f}%[/{pl_style}]",
                f"${pos.get('market_value', 0):,.2f}" if pos.get("market_value") else "N/A",
            )

        console.print(table)
        console.print()

    def _print_logged_open_trades(self, trades: list[dict]) -> None:
        table = Table(title="Open Trades (from local log)", box=box.ROUNDED)
        table.add_column("Ticker", style="bold")
        table.add_column("Entry Date")
        table.add_column("Entry Price", justify="right")
        table.add_column("Notional", justify="right")
        table.add_column("Signal Score", justify="right")
        table.add_column("Hold Period")
        table.add_column("Recommendation")

        for trade in trades:
            score = trade.get("composite_score") or 0
            score_style = "green" if score > 0 else "red"
            entry_date = (trade.get("entry_date") or "")[:10]

            table.add_row(
                trade["ticker"],
                entry_date,
                f"${trade.get('entry_price', 0):.2f}" if trade.get("entry_price") else "N/A",
                f"${trade.get('notional', 0):,.2f}" if trade.get("notional") else "N/A",
                f"[{score_style}]{score:+.3f}[/{score_style}]",
                trade.get("hold_period", "N/A"),
                trade.get("recommendation", "N/A"),
            )

        console.print(table)
        console.print()

    def _print_performance_summary(self, summary: dict) -> None:
        if summary.get("total_closed_trades", 0) == 0:
            console.print("[dim]No closed trades yet.[/dim]\n")
            return

        win_rate = summary.get("win_rate_pct", 0)
        target_met = win_rate >= 70.0
        target_icon = "✅" if target_met else "❌"
        win_style = "green" if target_met else "red"
        avg = summary.get("avg_outcome_pct", 0)
        avg_style = "green" if avg >= 0 else "red"

        console.print(Panel(
            f"[bold]Closed Trades:[/bold]     {summary['total_closed_trades']}  "
            f"({summary.get('profitable_trades', 0)} wins, {summary.get('losing_trades', 0)} losses)\n"
            f"[bold]Win Rate:[/bold]          [{win_style}]{win_rate:.1f}%[/{win_style}]  "
            f"{target_icon}  (target: ≥70%)\n"
            f"[bold]Avg Return/Trade:[/bold]  [{avg_style}]{avg:+.2f}%[/{avg_style}]\n"
            f"[bold]Best Trade:[/bold]        [green]{summary.get('best_trade_pct', 0):+.2f}%[/green]\n"
            f"[bold]Worst Trade:[/bold]       [red]{summary.get('worst_trade_pct', 0):+.2f}%[/red]\n"
            f"[bold]Cumulative:[/bold]        {summary.get('cumulative_return_pct', 0):+.2f}%",
            title=f"Performance Summary  {target_icon}",
            expand=False,
        ))
        console.print()

    def _print_recent_closed_trades(self, trades: list[dict]) -> None:
        table = Table(title="Recent Closed Trades", box=box.SIMPLE)
        table.add_column("Ticker", style="bold")
        table.add_column("Entry")
        table.add_column("Exit")
        table.add_column("Entry $", justify="right")
        table.add_column("Exit $", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Hold Period")

        for trade in trades:
            outcome = trade.get("outcome_pct") or 0
            outcome_style = "green" if outcome > 0 else "red"
            result_icon = "✅" if outcome > 0 else "❌"

            table.add_row(
                trade["ticker"],
                (trade.get("entry_date") or "")[:10],
                (trade.get("exit_date") or "")[:10],
                f"${trade.get('entry_price', 0):.2f}" if trade.get("entry_price") else "N/A",
                f"${trade.get('exit_price', 0):.2f}" if trade.get("exit_price") else "N/A",
                f"[{outcome_style}]{result_icon} {outcome:+.2f}%[/{outcome_style}]",
                trade.get("hold_period", "N/A"),
            )

        console.print(table)
