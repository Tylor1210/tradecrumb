"""
signal-engine CLI entry points.

Commands:
  scan             — Run daily signal scan and print ranked results
  backtest         — Run backtest against historical data
  paper            — Run paper trading session
  live             — Live intraday day-trading trigger (fixed-notional paper trades)
  leaps            — Run LEAPS candidate screener
  weekend-scan     — Weekend Swing Signals: Sat/Sun watchlist scan (output only)
  weekend-monitor  — Weekend Swing Signals: Monday-morning trigger checker (logs only)
  weekend-backtest — Weekend Swing Signals: backtest all 3 strategies

Usage:
  uv run signal-engine scan
  uv run signal-engine scan --date 2024-06-01
  uv run signal-engine backtest
  uv run signal-engine backtest --grid-search
  uv run signal-engine paper --dry-run
  uv run signal-engine live --once
  uv run signal-engine live --poll-minutes 15
  uv run signal-engine leaps
  uv run signal-engine weekend-scan
  uv run signal-engine weekend-monitor --once
  uv run signal-engine weekend-backtest
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()

console = Console()
logging.basicConfig(level=logging.WARNING)  # suppress noisy debug logs

CONFIG_DIR = Path("config")


def _load_config(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        console.print("[dim]Run from the project root directory.[/dim]")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
# CLI GROUP
# ─────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="signal-engine")
def main():
    """
    signal-engine — Market sentiment & signal research system.

    \b
    ⚠️  PAPER TRADING ONLY — no live order execution implemented.
    """
    pass


# ─────────────────────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--date", default=None, help="Scan date YYYY-MM-DD (default: today)")
@click.option("--signals", "signals_config", default="signals.yaml", help="Signal config file")
@click.option("--watchlist", "watchlist_config", default="watchlist.yaml", help="Watchlist file")
@click.option("--top", default=None, type=int, help="Show only top N tickers")
def scan(date, signals_config, watchlist_config, top):
    """Run daily signal scan and print ranked ticker results."""
    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient, NewsAPIClient
    from signal_engine.data.cache import ParquetCache
    from signal_engine.signals.technical import TechnicalAnalyzer
    from signal_engine.signals.sentiment import SentimentAnalyzer
    from signal_engine.signals.political import PoliticalTriggerDetector
    from signal_engine.signals.patterns import PatternDetector
    from signal_engine.signals.composite import CompositeScorer

    sig_cfg = _load_config(signals_config)
    wl_cfg = _load_config(watchlist_config)

    scan_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    all_tickers = wl_cfg.get("tickers", [])
    public_tickers = [t for t in all_tickers if t.get("public", True)]
    symbols = [t["symbol"] for t in public_tickers]

    console.print()
    console.rule(f"[bold cyan]signal-engine scan — {scan_date.strftime('%Y-%m-%d')}[/bold cyan]")
    console.print(f"[dim]Scanning {len(symbols)} publicly traded tickers[/dim]\n")

    # ── Clients ──────────────────────────────────────────────────────────
    cache = ParquetCache()

    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")
        finnhub = None

    newsapi = None
    try:
        newsapi = NewsAPIClient()
    except EnvironmentError:
        console.print("[dim]NewsAPI not configured — political sweep limited to Finnhub[/dim]")

    # ── Fetch price data ──────────────────────────────────────────────────
    price_start = scan_date - timedelta(days=400)  # enough for SMA-200

    cache_key = cache.cache_key_for_bars(symbols, price_start, scan_date, "1Day")
    price_data = cache.get(cache_key, ttl_hours=24)

    if price_data is None:
        with console.status("[bold green]Fetching price data from Alpaca..."):
            raw_prices = alpaca.get_historical_bars(symbols, price_start, scan_date)
        # Cache is per-ticker; we cache each ticker's data separately
        price_data = raw_prices
    else:
        # Cache returned a combined DataFrame — split back to dict
        # (for simplicity, re-fetch if structure doesn't match)
        if isinstance(price_data, dict):
            raw_prices = price_data
            price_data = raw_prices
        else:
            raw_prices = alpaca.get_historical_bars(symbols, price_start, scan_date)
            price_data = raw_prices

    console.print(f"[dim]✓ Price data loaded for {len(price_data)} tickers[/dim]")

    # ── Fetch news ────────────────────────────────────────────────────────
    from_date_str = (scan_date - timedelta(days=3)).strftime("%Y-%m-%d")
    to_date_str = scan_date.strftime("%Y-%m-%d")

    all_news = []
    if finnhub:
        with console.status("[bold green]Fetching news from Finnhub..."):
            for sym in symbols[:20]:  # rate limit protection
                try:
                    news = finnhub.get_company_news(sym, from_date_str, to_date_str)
                    all_news.extend(news)
                except Exception:
                    pass
            try:
                general = finnhub.get_market_news("general")
                all_news.extend(general)
            except Exception:
                pass

    # Political sweep via NewsAPI
    if newsapi:
        pol_config = sig_cfg.get("political", {})
        pol_keywords = pol_config.get("political_keywords", {})
        flat_keywords = [kw for kws in pol_keywords.values() for kw in kws]
        # Take a representative sample to stay under rate limit
        sample_keywords = flat_keywords[:8]
        if sample_keywords:
            with console.status("[bold green]Fetching political news from NewsAPI..."):
                try:
                    pol_news = newsapi.search_political_news(
                        sample_keywords, from_date_str, to_date_str
                    )
                    all_news.extend(pol_news)
                except Exception as e:
                    console.print(f"[yellow]NewsAPI sweep: {e}[/yellow]")

    console.print(f"[dim]✓ {len(all_news)} news articles loaded[/dim]")

    # ── Signal computation ────────────────────────────────────────────────
    tech_config = sig_cfg.get("technical", {})
    sent_config = sig_cfg.get("sentiment", {})
    pol_config = sig_cfg.get("political", {})
    pattern_config = sig_cfg.get("pattern", {})
    weights = sig_cfg.get("weights", {"technical": 0.30, "sentiment": 0.25, "political": 0.15, "pattern": 0.30})
    thresholds = sig_cfg.get("thresholds", {"buy_signal": 0.55, "sell_signal": -0.40})

    tech_analyzer = TechnicalAnalyzer(tech_config)
    sent_analyzer = SentimentAnalyzer(sent_config)
    watchlist_map = {t["symbol"]: t for t in all_tickers}
    pol_detector = PoliticalTriggerDetector(pol_config, watchlist_map)
    pattern_detector = PatternDetector(pattern_config)
    scorer = CompositeScorer(weights, thresholds)

    reports = []
    with console.status("[bold green]Computing signals..."):
        for ticker in symbols:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            technical = tech_analyzer.compute(ticker, df)
            sentiment = sent_analyzer.compute(ticker, all_news, scan_date)
            political = pol_detector.compute(ticker, all_news, scan_date)
            pattern = pattern_detector.compute(ticker, df)
            if technical is None and abs(sentiment.score) < 0.01 and abs(political.score) < 0.01:
                continue  # skip truly data-less tickers
            report = scorer.compute(ticker, technical, sentiment, political, pattern)
            reports.append(report)

    if not reports:
        console.print("[red]No signals computed — check API keys and watchlist.[/red]")
        sys.exit(1)

    ranked = scorer.rank_signals(reports)
    if top:
        ranked = ranked.head(top)

    # ── Display results ───────────────────────────────────────────────────
    table = Table(
        title=f"Signal Scan Results — {scan_date.strftime('%Y-%m-%d')}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Ticker", style="bold", width=7)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Rec", justify="center", width=7)
    table.add_column("Conf", justify="center", width=6)
    table.add_column("Tech", justify="right", width=7)
    table.add_column("Sent", justify="right", width=7)
    table.add_column("Pol", justify="right", width=7)
    table.add_column("Pattern", justify="right", width=7)
    table.add_column("Triggered Rules")

    for i, row in ranked.iterrows():
        rec = row["recommendation"]
        rec_color = "green" if rec == "BUY" else "red" if rec == "SHORT" else "yellow"
        score = row["composite_score"]
        score_color = "green" if score > 0.3 else "red" if score < -0.3 else "white"
        rules = row["triggered_rules"]
        if len(rules) > 70:
            rules = rules[:70] + "…"

        table.add_row(
            str(i + 1),
            row["ticker"],
            f"[{score_color}]{score:+.3f}[/{score_color}]",
            f"[{rec_color}]{rec}[/{rec_color}]",
            row["confidence"],
            f"{row['technical_score']:+.3f}",
            f"{row['sentiment_score']:+.3f}",
            f"{row['political_score']:+.3f}",
            f"{row['pattern_score']:+.3f}",
            rules,
        )

    console.print(table)
    buy_count = len(ranked[ranked["recommendation"] == "BUY"])
    short_count = len(ranked[ranked["recommendation"] == "SHORT"])
    console.print(
        f"\n[dim]  {len(ranked)} tickers scanned  |  "
        f"[green]{buy_count} BUY[/green]  "
        f"[red]{short_count} SHORT[/red]  "
        f"{len(ranked) - buy_count - short_count} HOLD[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--config", "bt_config", default="backtest.yaml", help="Backtest config file")
@click.option("--signals", "sig_config", default="signals.yaml", help="Signal config file")
@click.option("--watchlist", "wl_config", default="watchlist.yaml", help="Watchlist file")
@click.option("--grid-search", is_flag=True, help="Run full weight grid search")
@click.option("--compare-exits", is_flag=True, help="Run old exits vs the 10%/20% loss cap side by side")
@click.option("--output-dir", default="reports", help="Output directory for results")
def backtest(bt_config, sig_config, wl_config, grid_search, compare_exits, output_dir):
    """Run backtest against historical data. Use --grid-search to sweep weight combos."""
    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.backtest.engine import BacktestConfig, BacktestEngine
    from signal_engine.backtest.metrics import compute_directional_accuracy
    from signal_engine.backtest.grid_search import WeightGridSearch, GridSearchConfig

    bt_cfg = _load_config(bt_config)
    sig_cfg = _load_config(sig_config)
    wl_cfg = _load_config(wl_config)

    console.print()
    console.rule("[bold cyan]signal-engine backtest[/bold cyan]")

    symbols = [
        t["symbol"] for t in wl_cfg.get("tickers", []) if t.get("public", True)
    ]

    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    start_dt = datetime.strptime(bt_cfg["start_date"], "%Y-%m-%d")
    end_dt = datetime.strptime(bt_cfg["end_date"], "%Y-%m-%d")

    console.print(
        f"Period: {bt_cfg['start_date']} → {bt_cfg['end_date']}  |  "
        f"{len(symbols)} tickers"
    )

    with console.status("[bold green]Fetching historical price data..."):
        price_data = alpaca.get_historical_bars(symbols, start_dt, end_dt)

    console.print(f"[dim]✓ Price data loaded for {len(price_data)} tickers[/dim]\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if grid_search:
        console.print("[bold]Running weight grid search...[/bold]")
        console.print("[dim]This generates pre-computed signals from price data only (technical).[/dim]")
        console.print("[dim]For full signal grid search, run 'scan' for each date first.[/dim]\n")

        gs_cfg = GridSearchConfig.from_yaml(bt_cfg)

        # For grid search, use technical-only signals (price data available, no news API calls)
        from signal_engine.signals.technical import TechnicalAnalyzer
        from signal_engine.signals.composite import CompositeScorer

        tech_cfg = sig_cfg.get("technical", {})
        tech_analyzer = TechnicalAnalyzer(tech_cfg)

        # Pre-compute technical signals for all tickers and dates
        console.print("[dim]Pre-computing technical signals...[/dim]")
        all_tech_signals: dict[str, dict] = {}  # ticker → {date: score}
        for ticker, df in price_data.items():
            ticker_signals = {}
            for i in range(200, len(df)):
                sub_df = df.iloc[:i + 1]
                sig = tech_analyzer.compute(ticker, sub_df)
                if sig:
                    ticker_signals[df.index[i]] = sig.score
            all_tech_signals[ticker] = ticker_signals
        console.print(f"[dim]✓ Technical signals computed for {len(all_tech_signals)} tickers[/dim]\n")

        def signal_gen(weights: dict, hold_days: int, threshold: float) -> dict:
            """Generate buy signal Series for a given weight/threshold combo."""
            import pandas as pd
            result = {}
            w_tech = weights.get("weight_technical", 1.0)
            for ticker, score_dict in all_tech_signals.items():
                dates = sorted(score_dict.keys())
                signals = pd.Series(
                    [score_dict[d] * w_tech >= threshold for d in dates],
                    index=dates,
                )
                result[ticker] = signals
            return result

        gs = WeightGridSearch(gs_cfg)
        csv_path = output_path / "grid_search_results.csv"
        gs.run(price_data, signal_gen, output_csv=csv_path)

    else:
        # Single backtest run
        weights = sig_cfg.get("weights", {"technical": 0.40, "sentiment": 0.35, "political": 0.25})
        thresholds = sig_cfg.get("thresholds", {"buy_signal": 0.55, "sell_signal": -0.40})

        from signal_engine.signals.technical import TechnicalAnalyzer
        from signal_engine.backtest.engine import BacktestConfig, BacktestEngine

        from signal_engine.risk.loss_cap import LossCapRule

        loss_cap_cfg = bt_cfg.get("loss_cap") or {}
        loss_cap_rule = LossCapRule.from_config(loss_cap_cfg)

        console.print(f"Hold period: {bt_cfg.get('hold_period_days', 5)} days")
        console.print(f"Stop loss: {bt_cfg.get('stop_loss_pct', 5.0)}% | Take profit: {bt_cfg.get('take_profit_pct', 10.0)}%")
        if loss_cap_rule is not None:
            console.print(
                f"Loss cap: review at {loss_cap_rule.review_loss_pct:g}% down, start closing at "
                f"{loss_cap_rule.exit_loss_pct:g}% (within {loss_cap_rule.exit_within_bars} bars), "
                f"backstop {loss_cap_rule.backstop_loss_pct:g}%"
                + (" — replaces the flat stop" if loss_cap_rule.replaces_fixed_stop else "")
            )
        console.print()

        tech_analyzer = TechnicalAnalyzer(sig_cfg.get("technical", {}))
        buy_threshold = thresholds.get("buy_signal", 0.55)

        # Generate technical signals for each ticker
        import pandas as pd
        signals: dict = {}
        setup_scores: dict = {}  # per-bar technical score, used by the loss cap
        for ticker, df in price_data.items():
            sig_dates = []
            sig_vals = []
            score_vals = []
            for i in range(200, len(df)):
                sub = df.iloc[:i + 1]
                ts = tech_analyzer.compute(ticker, sub)
                sig_dates.append(df.index[i])
                sig_vals.append(ts.score >= buy_threshold if ts else False)
                score_vals.append(ts.score if ts else float("nan"))
            if sig_dates:
                signals[ticker] = pd.Series(sig_vals, index=sig_dates)
                setup_scores[ticker] = pd.Series(score_vals, index=sig_dates)

        def _make_config(rule):
            return BacktestConfig(
                start_date=bt_cfg["start_date"],
                end_date=bt_cfg["end_date"],
                hold_period_days=bt_cfg.get("hold_period_days", 5),
                stop_loss_pct=bt_cfg.get("stop_loss_pct", 5.0),
                take_profit_pct=bt_cfg.get("take_profit_pct", 10.0),
                initial_capital=bt_cfg.get("initial_capital", 100000.0),
                position_size_pct=bt_cfg.get("position_size_pct", 5.0),
                loss_cap=rule,
                setup_score_floor=float(loss_cap_cfg.get("setup_score_floor", 0.0)),
            )

        if compare_exits:
            rule_for_compare = loss_cap_rule or LossCapRule()
            rows = []
            for label, rule in [("Old exits (flat stop)", None), ("10%/20% loss cap", rule_for_compare)]:
                res = BacktestEngine(_make_config(rule)).run(price_data, signals, setup_scores)
                rows.append((label, res.get_metrics(), res.exit_reasons))
            _print_exit_comparison(rows)
            return

        bt_config_obj = _make_config(loss_cap_rule)

        try:
            engine = BacktestEngine(bt_config_obj)
            result = engine.run(price_data, signals, setup_scores)
            metrics = result.get_metrics()

            # Compute directional accuracy per ticker
            dir_acc_values = []
            for ticker, df in price_data.items():
                if ticker in signals:
                    da = compute_directional_accuracy(df, signals[ticker], bt_config_obj.hold_period_days)
                    dir_acc_values.append(da)
            if dir_acc_values:
                metrics.directional_accuracy_pct = round(sum(dir_acc_values) / len(dir_acc_values), 2)

            console.print(metrics)
            if result.exit_reasons:
                console.print(f"[dim]Exit reasons: {result.exit_reasons}[/dim]")

            if bt_cfg.get("save_equity_curve", True):
                eq_path = output_path / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                result.save_equity_curve(eq_path)

        except ImportError as e:
            console.print(f"[red]vectorbt not available: {e}[/red]")
            console.print("[dim]Install with: uv add vectorbt[/dim]")


def _bar_check(win_rate: float, trades: int) -> str:
    """Ty's bar for going past paper: 60-70%+ of trades in profit."""
    if trades == 0:
        return "n/a"
    if win_rate >= 70:
        return "✅ ≥70%"
    if win_rate >= 60:
        return "🟡 60-70%"
    return "❌ <60%"


def _print_exit_comparison(rows: list) -> None:
    table = Table(title="Exit rules: before vs after", box=box.ROUNDED)
    for col in ["Exits", "Trades", "Win Rate", "Bar", "Avg Loss", "Worst Trade", "Max DD", "Return"]:
        table.add_column(col, justify="left" if col == "Exits" else "right")
    for label, m, reasons in rows:
        table.add_row(
            label, str(m.total_trades), f"{m.win_rate_pct:.1f}%", _bar_check(m.win_rate_pct, m.total_trades),
            f"{m.avg_loss_pct:+.2f}%", f"{m.worst_trade_pct:+.2f}%",
            f"{m.max_drawdown_pct:.2f}%", f"{m.total_return_pct:+.2f}%",
        )
    console.print(table)
    for label, _m, reasons in rows:
        if reasons:
            console.print(f"[dim]{label} exit reasons: {reasons}[/dim]")


# ─────────────────────────────────────────────────────────────
# PAPER TRADING
# ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--config", "pt_config", default="paper_trading.yaml", help="Paper trading config")
@click.option("--signals", "sig_config", default="signals.yaml", help="Signal config")
@click.option("--watchlist", "wl_config", default="watchlist.yaml", help="Watchlist")
@click.option("--dry-run", is_flag=True, help="Compute signals but don't submit orders")
def paper(pt_config, sig_config, wl_config, dry_run):
    """Run paper trading session. All trades are simulated via Alpaca paper API."""
    from signal_engine.paper_trading.tracker import TradeTracker
    from signal_engine.paper_trading.reporter import PaperTradingReporter

    console.print()
    console.rule("[bold cyan]signal-engine paper trading[/bold cyan]")

    if dry_run:
        console.print("[yellow]DRY RUN — signals computed but no orders submitted[/yellow]")

    tracker = TradeTracker()
    reporter = PaperTradingReporter(tracker)

    account_info = None
    positions = None

    try:
        from signal_engine.paper_trading.broker import PaperTradingBroker
        broker = PaperTradingBroker()
        account_info = broker.get_account()
        positions = broker.get_positions()
        console.print("[dim]✓ Connected to Alpaca paper account[/dim]")
    except EnvironmentError as e:
        console.print(f"[yellow]{e}[/yellow]")
        console.print("[dim]Running in log-only mode (no broker connection)[/dim]")
    except Exception as e:
        console.print(f"[yellow]Broker connection failed: {e}[/yellow]")

    reporter.print_daily_summary(account_info=account_info, positions=positions)

    if not dry_run and account_info:
        console.print("[dim]Signal-driven paper order execution: see documentation.[/dim]")
        console.print("[dim]Run 'signal-engine scan' then pass high-confidence BUY signals to broker.[/dim]")

    console.print("[yellow]⚠️  Paper trading only — no live funds involved[/yellow]\n")


# ─────────────────────────────────────────────────────────────
# LIVE (intraday day-trading trigger)
# ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--config", "pt_config", default="paper_trading.yaml", help="Paper trading config")
@click.option("--signals", "sig_config", default="signals.yaml", help="Signal config")
@click.option("--watchlist", "wl_config", default="watchlist.yaml", help="Watchlist")
@click.option("--once", is_flag=True, help="Run a single check then exit (use this from cron)")
@click.option("--poll-minutes", default=15, type=int, help="Minutes between checks in loop mode")
@click.option("--force", is_flag=True, help="Ignore market-hours gating (for testing outside market hours)")
def live(pt_config, sig_config, wl_config, once, poll_minutes, force):
    """
    Live intraday day-trading trigger — paper only.

    Re-scans the watchlist and submits fixed-notional ($1,000 by default,
    see paper_trading.yaml) paper day trades for signals crossing the
    execution threshold, auto-closing them near the close. Use --once from
    cron to fire every N minutes during market hours; omit it to run a
    foreground polling loop.
    """
    import time as _time

    from signal_engine.paper_trading.live_trigger import LiveDayTradeTrigger, is_market_hours

    pt_cfg = _load_config(pt_config)
    sig_cfg = _load_config(sig_config)
    wl_cfg = _load_config(wl_config)

    console.print()
    console.rule("[bold cyan]signal-engine live day-trade trigger[/bold cyan]")
    console.print(
        f"[dim]Fixed trade size: ${pt_cfg.get('fixed_trade_size', 1000):.0f}"
        f" | execution threshold: {pt_cfg.get('execution_threshold', 0.55)}"
        f" | max positions: {pt_cfg.get('max_positions', 10)}[/dim]"
    )
    console.print("[yellow]⚠️  Paper trading only — no live funds involved[/yellow]\n")

    try:
        trigger = LiveDayTradeTrigger(pt_cfg, sig_cfg, wl_cfg)
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    def tick():
        stamp = datetime.now().isoformat(timespec="seconds")
        if not force and not is_market_hours():
            console.print(f"[dim]{stamp} — market closed, skipping[/dim]")
            return
        result = trigger.run_once(force=force)
        for e in result.get("entered", []):
            side = e.get("side", "BUY")
            color = "green" if side == "BUY" else "red"
            console.print(f"[{color}]{stamp} {side} {e['ticker']} — score {e['score']:.3f} — trade {e['trade_id'][:8]}[/{color}]")
        for x in result.get("exited", []):
            pct = x.get("outcome_pct")
            pct_str = f"{pct:+.2f}%" if pct is not None else "n/a"
            console.print(f"[cyan]{stamp} EXIT {x['ticker']} — {pct_str} — trade {x['trade_id'][:8]}[/cyan]")
        if not result.get("entered") and not result.get("exited"):
            console.print(f"[dim]{stamp} — no action[/dim]")

    if once:
        tick()
        return

    console.print(f"[dim]Polling every {poll_minutes} min during market hours. Ctrl+C to stop.[/dim]\n")
    try:
        while True:
            tick()
            _time.sleep(poll_minutes * 60)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


# ─────────────────────────────────────────────────────────────
# LEAPS
# ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--signals", "sig_config", default="signals.yaml", help="Signal config")
@click.option("--watchlist", "wl_config", default="watchlist.yaml", help="Watchlist")
@click.option("--top", default=15, type=int, help="Show top N candidates")
def leaps(sig_config, wl_config, top):
    """Run LEAPS options candidate screener."""
    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient
    from signal_engine.leaps.candidate_screener import LEAPSScreener
    from signal_engine.leaps.options_stub import OptionsChainClient

    sig_cfg = _load_config(sig_config)
    wl_cfg = _load_config(wl_config)
    leaps_cfg = sig_cfg.get("leaps", {})

    console.print()
    console.rule("[bold cyan]signal-engine LEAPS screener[/bold cyan]")

    symbols = [
        t["symbol"] for t in wl_cfg.get("tickers", []) if t.get("public", True)
    ]
    watchlist_map = {t["symbol"]: t for t in wl_cfg.get("tickers", [])}

    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    finnhub = None
    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError:
        console.print("[dim]Finnhub not configured — earnings trend scoring disabled[/dim]")

    # Need 400 days to get SMA-200 + RS window
    start_dt = datetime.now() - timedelta(days=400)
    end_dt = datetime.now()

    # Always fetch SPY for relative strength benchmark
    fetch_symbols = list(set(symbols + ["SPY"]))
    with console.status("[bold green]Fetching price data..."):
        price_data = alpaca.get_historical_bars(fetch_symbols, start_dt, end_dt)

    spy_df = price_data.get("SPY")
    if spy_df is None or spy_df.empty:
        console.print("[red]Could not fetch SPY data — required for relative strength[/red]")
        sys.exit(1)

    console.print(f"[dim]✓ Price data for {len(price_data)} tickers[/dim]\n")

    screener = LEAPSScreener(leaps_cfg, finnhub_client=finnhub)
    options_client = OptionsChainClient()

    candidates = []
    with console.status("[bold green]Screening LEAPS candidates..."):
        for ticker in symbols:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            candidate = screener.screen(ticker, df, spy_df, watchlist_map.get(ticker, {}))
            candidates.append(candidate)

    candidates.sort(key=lambda c: c.leaps_score, reverse=True)

    table = Table(
        title=f"LEAPS Long Call Candidates (Top {min(top, len(candidates))}) — Bullish / Price UP",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Ticker", style="bold", width=7)
    table.add_column("Direction", justify="center", width=16)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Now", justify="right", width=9)
    table.add_column("Strike", justify="right", width=9)
    table.add_column("1yr Target", justify="right", width=11)
    table.add_column("Upside", justify="right", width=8)
    table.add_column("Vol", justify="right", width=6)
    table.add_column("Prem/share", justify="right", width=11)
    table.add_column("Est $1k → (hit target)", justify="right", width=22)
    table.add_column("Catalysts")

    for i, c in enumerate(candidates[:top]):
        score_color = "green" if c.leaps_score >= 0.6 else "yellow" if c.leaps_score >= 0.3 else "red"
        catalysts_str = " | ".join(c.catalysts[:2]) or "[dim]None[/dim]"

        # 1yr target & upside
        target_str = f"${c.target_price_1yr:.2f}" if c.target_price_1yr else "N/A"
        upside_str = f"[green]+{c.target_upside_pct:.1f}%[/green]" if c.target_upside_pct else "N/A"

        # Volatility
        vol_str = f"{c.hist_volatility * 100:.0f}%" if c.hist_volatility else "N/A"

        # Premium per share (Black-Scholes estimate)
        prem_str = f"${c.est_call_premium:.2f}" if c.est_call_premium else "N/A"

        # Estimated return on $1,000
        if c.est_return_on_1k is not None and c.est_return_pct_on_1k is not None:
            ret_color = "green" if c.est_return_on_1k >= 0 else "red"
            sign = "+" if c.est_return_on_1k >= 0 else ""
            ret_str = (
                f"[{ret_color}]{sign}${c.est_return_on_1k:,.0f} ({sign}{c.est_return_pct_on_1k:.0f}%)[/{ret_color}]"
            )
        else:
            ret_str = "N/A"

        table.add_row(
            str(i + 1),
            c.ticker,
            "[green]📈 CALL (Bullish)[/green]",
            f"[{score_color}]{c.leaps_score:.3f}[/{score_color}]",
            f"${c.current_price:.2f}" if c.current_price else "N/A",
            f"${c.suggested_strike:.2f}" if c.suggested_strike else "N/A",
            target_str,
            upside_str,
            vol_str,
            prem_str,
            ret_str,
            catalysts_str,
        )

    console.print(table)
    console.print(
        "\n[bold]📖 How to read this table:[/bold]\n"
        "  [green]Direction = CALL (Bullish)[/green] — ALL candidates are long call setups. You profit if the stock goes [green]UP[/green].\n"
        "  [bold]Now[/bold] = current stock price  |  [bold]Strike[/bold] = suggested call strike (~5% OTM)  |  [bold]1yr Target[/bold] = conservative price projection\n"
        "  [bold]Upside[/bold] = stock % gain needed to hit target  |  [bold]Vol[/bold] = annualized historical volatility\n"
        "  [bold]Prem/share[/bold] = Black-Scholes call premium estimate (× 100 = cost of 1 contract)\n"
        "  [bold]Est $1k → (hit target)[/bold] = P&L if you put $1,000 into calls [italic]and[/italic] the stock exactly hits the 1yr target at expiry\n"
        "  [yellow]⚠️  These are model estimates — options can expire worthless. Past momentum ≠ future returns.[/yellow]\n"
    )

    if not options_client.is_connected:
        console.print(
            "\n[dim]💡 Options chain data is stubbed — connect Polygon.io for live IV & Greeks (sharper premium estimates).[/dim]"
        )
        console.print("[dim]   See: src/signal_engine/leaps/README_LEAPS.md[/dim]\n")




# ─────────────────────────────────────────────────────────────
# WEEKEND SWING SIGNALS
# ─────────────────────────────────────────────────────────────
# Independent weekly strategy lane: weekend scan -> Monday trigger -> stop ->
# Friday-close-or-stopped-out exit. NOT wired into the daily scan/live-trigger
# flow — see src/signal_engine/weekend_swing/weekend_swing_signals.py.
# Scan results and triggered signals persist to data/weekend_swing.db and
# surface in the dashboard's Weekend Swing tab. Paper-trading submission
# only happens when explicitly triggered from that tab (see
# dashboard/server.py's /api/weekend-swing/submit) — these CLI commands
# never submit an order on their own.

@main.command(name="weekend-scan")
@click.option("--config", "ws_config", default="weekend_swing.yaml", help="Weekend swing config")
def weekend_scan(ws_config):
    """
    Weekend Swing Signals — Saturday/Sunday scan.

    Builds the top-100-most-active + core universe, runs the cheap
    (daily-bar-only) pivot-proximity filter across all 3 strategies, prints
    the resulting watchlist, and saves it (data/weekend_swing.db) so the
    dashboard's Weekend Swing tab can show it. Still no order submission —
    output/persistence only.
    """
    import pandas as pd

    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient
    from signal_engine.weekend_swing.data_store import IntradayBarStore
    from signal_engine.weekend_swing.scan import build_weekend_watchlist, next_or_current_monday
    from signal_engine.weekend_swing.tracker import WeekendSwingTracker

    ws_cfg = _load_config(ws_config)

    console.print()
    console.rule("[bold cyan]signal-engine weekend swing scan[/bold cyan]")

    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    finnhub = None
    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError:
        console.print("[dim]Finnhub not configured — earnings exclusion disabled[/dim]")

    week_start = next_or_current_monday(pd.Timestamp.now(tz="UTC"))
    bar_store = IntradayBarStore(alpaca)

    with console.status("[bold green]Building universe + running cheap pivot-proximity scan..."):
        result = build_weekend_watchlist(ws_cfg, alpaca, finnhub, bar_store, week_start)

    console.print(f"[dim]✓ Universe: {result['universe_size']} tickers | Week of {result['week_start'].date()}[/dim]")

    if result["skipped"]:
        console.print(f"[yellow]⚠ {len(result['skipped'])} tickers skipped[/yellow]")
        for ticker, reason in result["skipped"][:20]:
            console.print(f"[dim]  - {ticker}: {reason}[/dim]")
        if len(result["skipped"]) > 20:
            console.print(f"[dim]  ... and {len(result['skipped']) - 20} more (see logs)[/dim]")

    candidates = result["candidates"]

    tracker = WeekendSwingTracker()
    tracker.save_scan_results(result["week_start"].date().isoformat(), candidates)
    console.print(f"[dim]✓ Saved to data/weekend_swing.db — visible in the dashboard's Weekend Swing tab[/dim]")

    if not candidates:
        console.print("\n[yellow]No candidates within the pivot-proximity filter this week.[/yellow]\n")
        return

    table = Table(
        title=f"Weekend Swing Watchlist — week of {result['week_start'].date()}",
        box=box.ROUNDED, show_lines=False,
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Strategy")
    table.add_column("Pivot High", justify="right")
    table.add_column("Pivot Low", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Dist %", justify="right")
    table.add_column("Earnings", justify="center")
    table.add_column("Data Note")

    for c in candidates:
        table.add_row(
            c["ticker"], c["strategy"],
            f"${c['pivot_high']:.2f}", f"${c['pivot_low']:.2f}",
            f"${c['current_price']:.2f}", f"{c['distance_pct']:.2f}%",
            "[red]⚠ YES[/red]" if c["earnings_excluded"] else "",
            c["data_quality"] or "",
        )

    console.print(table)
    console.print(
        f"\n[dim]{len(candidates)} candidates ready for Monday's trigger check "
        f"(run: signal-engine weekend-monitor)[/dim]\n"
    )


@main.command(name="weekend-monitor")
@click.option("--config", "ws_config", default="weekend_swing.yaml", help="Weekend swing config")
@click.option("--once", is_flag=True, help="Run a single check then exit (use this from cron)")
@click.option("--poll-minutes", default=5, type=int, help="Minutes between checks in loop mode")
def weekend_monitor(ws_config, once, poll_minutes):
    """
    Weekend Swing Signals — Monday-morning trigger checker.

    Rebuilds the same cheap shortlist as weekend-scan, then pulls intraday
    5/15-min data ONLY for shortlisted tickers and checks each strategy's
    real trigger condition. Triggered signals are logged to the console AND
    saved (data/weekend_swing.db) so they show up in the dashboard's Monday
    Trigger Feed — still no automatic paper-trading submission; that happens
    only when you explicitly submit a signal from the dashboard.
    """
    import time as _time

    import pandas as pd

    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient
    from signal_engine.weekend_swing.data_store import IntradayBarStore
    from signal_engine.weekend_swing.scan import build_weekend_watchlist, next_or_current_monday
    from signal_engine.weekend_swing.tracker import WeekendSwingTracker
    from signal_engine.weekend_swing.weekend_swing_signals import (
        GapStrengthStrategy,
        OvernightRangeBreakoutStrategy,
        PriorWeekBreakoutStrategy,
    )

    ws_cfg = _load_config(ws_config)

    console.print()
    console.rule("[bold cyan]signal-engine weekend swing — Monday trigger monitor[/bold cyan]")
    console.print("[yellow]⚠️  Signal logging only — no paper-trading submission in this pass[/yellow]\n")

    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    finnhub = None
    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError:
        pass

    bar_store = IntradayBarStore(alpaca)
    week_start = next_or_current_monday(pd.Timestamp.now(tz="UTC"))

    with console.status("[bold green]Rebuilding shortlist (cheap daily-bar pass)..."):
        watchlist = build_weekend_watchlist(ws_cfg, alpaca, finnhub, bar_store, week_start)

    candidates = watchlist["candidates"]
    console.print(f"[dim]✓ {len(candidates)} shortlisted ticker/strategy pairs for this Monday[/dim]\n")

    if not candidates:
        console.print("[yellow]Nothing shortlisted — nothing to monitor.[/yellow]")
        return

    strategies = {
        "prior_week_breakout": PriorWeekBreakoutStrategy(ws_cfg),
        "overnight_range_breakout": OvernightRangeBreakoutStrategy(ws_cfg),
        "gap_strength": GapStrengthStrategy(ws_cfg),
    }
    tracker = WeekendSwingTracker()
    week_start_str = week_start.date().isoformat()
    already_triggered = set()  # (ticker, strategy) pairs — don't re-log the same trigger

    def tick():
        stamp = datetime.now().isoformat(timespec="seconds")
        fired_any = False
        for c in candidates:
            key = (c["ticker"], c["strategy"])
            if key in already_triggered:
                continue

            ticker, strategy_name = c["ticker"], c["strategy"]
            strategy = strategies[strategy_name]

            try:
                if strategy_name == "prior_week_breakout":
                    daily_df = bar_store.get_bars(ticker, "1Day", week_start - pd.Timedelta(days=90), week_start)
                    hist_15 = bar_store.get_bars(ticker, "15Min", week_start - pd.Timedelta(days=40), week_start)
                    levels = strategy.scan(ticker, daily_df, week_start, hist_15, watchlist["earnings_excluded"])
                    if levels is None:
                        continue
                    monday_bars = bar_store.get_bars(ticker, "15Min", week_start, pd.Timestamp.now(tz="UTC"))
                    monday_bars = monday_bars.loc[monday_bars.index < week_start + pd.Timedelta(days=1)]
                    signal = strategy.check_trigger(levels, monday_bars)

                elif strategy_name == "overnight_range_breakout":
                    daily_df = bar_store.get_bars(ticker, "1Day", week_start - pd.Timedelta(days=60), week_start)
                    levels = strategy.scan(ticker, daily_df, week_start)
                    if levels is None:
                        continue
                    monday_bars = bar_store.get_bars(ticker, "5Min", week_start, pd.Timestamp.now(tz="UTC"))
                    monday_bars = monday_bars.loc[monday_bars.index < week_start + pd.Timedelta(days=1)]
                    if monday_bars.empty:
                        continue
                    signal = strategy.check_trigger(levels, monday_bars, monday_bars.index.min())

                else:  # gap_strength
                    daily_df = bar_store.get_bars(ticker, "1Day", week_start - pd.Timedelta(days=60), week_start)
                    levels = strategy.scan(ticker, daily_df, week_start)
                    if levels is None:
                        continue
                    monday_daily = daily_df.loc[daily_df.index >= week_start]
                    if monday_daily.empty:
                        continue
                    monday_open = float(monday_daily.iloc[0]["open"])
                    monday_bars = bar_store.get_bars(ticker, "5Min", week_start, pd.Timestamp.now(tz="UTC"))
                    monday_bars = monday_bars.loc[monday_bars.index < week_start + pd.Timedelta(days=1)]
                    if monday_bars.empty:
                        continue
                    signal = strategy.check_trigger(levels, monday_open, monday_bars.iloc[0])

            except Exception as e:
                console.print(f"[dim]{stamp} {ticker}/{strategy_name} check failed: {e}[/dim]")
                continue

            if signal is not None:
                already_triggered.add(key)
                fired_any = True
                signal_id = tracker.log_triggered_signal(signal, week_start_str)
                color = "green" if signal.direction == "long" else "red"
                gap_note = f" [{signal.breakdown.get('gap_type')}]" if signal.breakdown.get("gap_type") else ""
                console.print(
                    f"[{color}]{stamp} TRIGGER {ticker} ({strategy_name}{gap_note}) — "
                    f"{signal.direction.upper()} @ ${signal.entry_price:.2f} stop ${signal.stop_price:.2f} "
                    f"— saved as {signal_id[:8]}[/{color}]"
                )

        if not fired_any:
            console.print(
                f"[dim]{stamp} — no new triggers ({len(already_triggered)}/{len(candidates)} fired so far)[/dim]"
            )

    if once:
        tick()
        return

    console.print(f"[dim]Polling every {poll_minutes} min. Ctrl+C to stop.[/dim]\n")
    try:
        while True:
            tick()
            _time.sleep(poll_minutes * 60)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@main.command(name="weekend-backtest")
@click.option("--config", "ws_config", default="weekend_swing.yaml", help="Weekend swing config")
@click.option("--tickers", default=None, help="Comma-separated ticker override (default: core + N from screener, per config)")
@click.option("--months", default=None, type=int, help="Lookback months override (default: from config)")
@click.option("--compare-exits", is_flag=True, help="Run without and with the 10%/20% loss cap side by side")
def weekend_backtest(ws_config, tickers, months, compare_exits):
    """
    Weekend Swing Signals — backtest all 3 strategies against cached
    historical data.

    First run fetches and permanently caches intraday bars
    (data/weekend_swing_bars/) — subsequent runs over the same range make
    no new API calls.
    """
    import pandas as pd

    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient
    from signal_engine.weekend_swing.backtest import WeekendSwingBacktester
    from signal_engine.weekend_swing.data_store import IntradayBarStore

    ws_cfg = _load_config(ws_config)
    bt_cfg = ws_cfg.get("backtest", {})

    console.print()
    console.rule("[bold cyan]signal-engine weekend swing backtest[/bold cyan]")

    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    finnhub = None
    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError:
        console.print("[dim]Finnhub not configured — earnings exclusion disabled for this backtest[/dim]")

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        core = ws_cfg.get("universe", {}).get("core_tickers", [])
        initial_size = bt_cfg.get("initial_universe_size", 20)
        extra_needed = max(0, initial_size - len(core))
        with console.status("[bold green]Fetching most-active tickers for backtest universe..."):
            most_active = alpaca.get_most_active_symbols(top=max(extra_needed * 3, 20))
        extra = [s for s in most_active if s not in core][:extra_needed]
        ticker_list = core + extra

    lookback_months = months or bt_cfg.get("lookback_months", 12)
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.DateOffset(months=lookback_months)

    console.print(f"[dim]Universe ({len(ticker_list)}): {', '.join(ticker_list)}[/dim]")
    console.print(f"[dim]Lookback: {start.date()} -> {end.date()} ({lookback_months} months)[/dim]")
    console.print("[dim]First run fetches + permanently caches intraday bars — this can take a while.[/dim]\n")

    bar_store = IntradayBarStore(alpaca)
    backtester = WeekendSwingBacktester(ws_cfg, bar_store, finnhub)

    if compare_exits:
        from signal_engine.risk.loss_cap import LossCapRule

        rule = backtester.loss_cap or LossCapRule()
        table = Table(title="Weekend Swing: exits before vs after", box=box.ROUNDED)
        for col in ["Strategy", "Exits", "Trades", "Win Rate", "Bar", "Avg Loss", "Worst Trade", "Max DD", "Cap exits"]:
            table.add_column(col, justify="left" if col in ("Strategy", "Exits") else "right")
        for exits_label, cap in [("old", None), ("loss cap", rule)]:
            backtester.loss_cap = cap
            with console.status(f"[bold green]Backtesting all strategies ({exits_label} exits)..."):
                s1 = backtester.run_strategy_1(ticker_list, start, end)
                s2 = backtester.run_strategy_2(ticker_list, start, end)
                s3_go, s3_fail = backtester.run_strategy_3(ticker_list, start, end)
            for label, m in [
                ("1: Prior-Week Breakout", s1), ("2: Overnight Range Breakout", s2),
                ("3: Gap-and-Go (SPY/QQQ)", s3_go), ("3: Gap-and-Fail (SPY/QQQ)", s3_fail),
            ]:
                has = m.trade_count > 0
                table.add_row(
                    label, exits_label, str(m.trade_count),
                    f"{m.win_rate_pct:.1f}%" if has else "n/a", _bar_check(m.win_rate_pct, m.trade_count),
                    f"{m.avg_loss_pct:.2f}%" if has else "n/a", f"{m.worst_trade_pct:.2f}%" if has else "n/a",
                    f"{m.max_drawdown_pct:.2f}%" if has else "n/a", str(m.loss_cap_exits),
                )
        console.print(table)
        console.print("[dim]Paper trading only — no live funds involved. Output only, nothing submitted.[/dim]\n")
        return

    with console.status("[bold green]Backtesting Strategy 1 (Prior-Week Breakout Continuation)..."):
        s1_metrics = backtester.run_strategy_1(ticker_list, start, end)
    with console.status("[bold green]Backtesting Strategy 2 (Overnight Range Breakout)..."):
        s2_metrics = backtester.run_strategy_2(ticker_list, start, end)
    with console.status("[bold green]Backtesting Strategy 3 (Gap Strength/Fade — SPY/QQQ only)..."):
        s3_go, s3_fail = backtester.run_strategy_3(ticker_list, start, end)

    table = Table(title="Weekend Swing Backtest Results", box=box.ROUNDED)
    table.add_column("Strategy")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Win", justify="right")
    table.add_column("Avg Loss", justify="right")
    table.add_column("Worst Trade", justify="right")
    table.add_column("Max DD", justify="right")

    for label, m in [
        ("1: Prior-Week Breakout", s1_metrics),
        ("2: Overnight Range Breakout", s2_metrics),
        ("3: Gap-and-Go (SPY/QQQ)", s3_go),
        ("3: Gap-and-Fail (SPY/QQQ)", s3_fail),
    ]:
        table.add_row(
            label, str(m.trade_count),
            f"{m.win_rate_pct:.1f}%" if m.trade_count else "n/a",
            f"+{m.avg_win_pct:.2f}%" if m.trade_count else "n/a",
            f"{m.avg_loss_pct:.2f}%" if m.trade_count else "n/a",
            f"{m.worst_trade_pct:.2f}%" if m.trade_count else "n/a",
            f"{m.max_drawdown_pct:.2f}%" if m.trade_count else "n/a",
        )

    console.print(table)
    console.print(
        "\n[yellow]⚠️  Strategy 2 uses a regular-session proxy for overnight range "
        "(no true after-hours data on the free IEX feed) — treat its numbers accordingly.[/yellow]"
    )
    console.print("[dim]Paper trading only — no live funds involved. Output only, nothing submitted.[/dim]\n")


# ─────────────────────────────────────────────────────────────
# MARKET REGIME (fear/greed + buying/selling pressure)
# ─────────────────────────────────────────────────────────────

def _cached_universe(ws_cfg: dict) -> list:
    """Core tickers + every ticker with cached daily AND 15-min bars (no API calls needed to use them)."""
    from pathlib import Path

    store = Path("data/weekend_swing_bars")
    cached = {p.name.split("_")[0] for p in store.glob("*_15Min.parquet")} if store.exists() else set()
    core = [t.upper() for t in ws_cfg.get("universe", {}).get("core_tickers", [])]
    return list(dict.fromkeys(core + sorted(cached)))


@main.command()
@click.option("--config", "rg_config", default="market_regime.yaml", help="Regime config file")
@click.option("--weekend-config", "ws_config", default="weekend_swing.yaml", help="Weekend swing config (breadth universe)")
@click.option("--no-save", is_flag=True, help="Don't write the snapshot to data/market_regime.db")
def regime(rg_config, ws_config, no_save):
    """Fear/Greed index + which way buying/selling pressure is coming from."""
    import pandas as pd

    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.market_regime.provider import RegimeProvider
    from signal_engine.market_regime.store import RegimeStore
    from signal_engine.weekend_swing.data_store import IntradayBarStore

    rg_cfg, ws_cfg = _load_config(rg_config), _load_config(ws_config)
    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    provider = RegimeProvider(rg_cfg, IntradayBarStore(alpaca), _cached_universe(ws_cfg))
    with console.status("[bold green]Computing market regime..."):
        r = provider.as_of(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1))  # include today's bar
    if r is None:
        console.print("[red]No SPY data available — cannot compute regime.[/red]")
        sys.exit(1)

    console.print()
    console.rule(f"[bold cyan]Market regime as of {r.as_of}[/bold cyan]")
    fg = f"{r.fear_greed:.0f}" if r.fear_greed is not None else "n/a"
    color = "green" if r.pressure_label == "BULLISH" else "red" if r.pressure_label == "BEARISH" else "yellow"
    console.print(f"Fear/Greed: [bold]{fg}[/bold] ({r.fear_greed_label})   "
                  f"Pressure: [{color}]{r.pressure:+.2f} {r.pressure_label}[/{color}]\n")

    t = Table(box=box.SIMPLE, title="Components")
    t.add_column("Fear/Greed input"); t.add_column("Score", justify="right"); t.add_column("Raw", justify="right")
    for k, v in r.components.items():
        t.add_row(k, f"{v['score']:.0f}", str(v["raw"]))
    console.print(t)
    t = Table(box=box.SIMPLE)
    t.add_column("Pressure input"); t.add_column("Value", justify="right")
    for k, v in r.pressure_components.items():
        t.add_row(k, f"{v:+.2f}")
    console.print(t)
    for line in r.summary:
        console.print(f"  - {line}")
    console.print()

    if not no_save:
        RegimeStore().save(r)


@main.command(name="weekend-regime-study")
@click.option("--config", "ws_config", default="weekend_swing.yaml", help="Weekend swing config")
@click.option("--regime-config", "rg_config", default="market_regime.yaml", help="Regime config file")
@click.option("--tickers", default=None, help="Comma-separated override (default: core + all tickers with cached 15-min bars)")
@click.option("--months", default=None, type=int, help="Lookback months override")
def weekend_regime_study(ws_config, rg_config, tickers, months):
    """
    Backtest the weekend strategies with every trade tagged by its Monday's
    regime, then report results by pressure/fear-greed bucket and what the
    configured regime policy would have changed (full range and each half).
    """
    import pandas as pd

    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient
    from signal_engine.market_regime.policy import RegimePolicy
    from signal_engine.market_regime.provider import RegimeProvider
    from signal_engine.weekend_swing.backtest import WeekendSwingBacktester
    from signal_engine.weekend_swing.data_store import IntradayBarStore
    from signal_engine.weekend_swing.regime_study import run_study, trades_to_dicts

    ws_cfg, rg_cfg = _load_config(ws_config), _load_config(rg_config)
    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    finnhub = None
    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError:
        pass

    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else _cached_universe(ws_cfg)
    lookback = months or ws_cfg.get("backtest", {}).get("lookback_months", 12)
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.DateOffset(months=lookback)
    console.print(f"[dim]{len(ticker_list)} tickers, {start.date()} -> {end.date()}[/dim]")

    bar_store = IntradayBarStore(alpaca)
    provider = RegimeProvider(rg_cfg, bar_store, ticker_list)
    provider.for_range(start, end)
    bt = WeekendSwingBacktester(ws_cfg, bar_store, finnhub, regime_provider=provider)

    with console.status("[bold green]Replaying strategies with regime tags..."):
        trades = bt.collect_strategy_1_trades(ticker_list, start, end)
        trades += bt.collect_strategy_2_trades(ticker_list, start, end)
        go, fail = bt.collect_strategy_3_trades(ticker_list, start, end)
        trades += go + fail

    # Study uses an always-on policy built from the config's thresholds so it can be
    # evaluated before the live switch is flipped.
    pol_cfg = dict(rg_cfg)
    pol_cfg["policy"] = {**rg_cfg.get("policy", {}), "enabled": True}
    study = run_study(trades_to_dicts(trades), RegimePolicy(pol_cfg))

    console.print(f"\n[bold]{study['n_trades']} trades ({study['n_tagged']} regime-tagged)[/bold]")
    for title, key in [("By pressure label / direction", "by_pressure_label"),
                       ("With vs against pressure", "by_alignment"),
                       ("By fear/greed / direction", "by_fear_greed")]:
        t = Table(title=title, box=box.SIMPLE)
        t.add_column("Bucket"); t.add_column("N", justify="right")
        t.add_column("Win%", justify="right"); t.add_column("EV/trade %", justify="right")
        for k, v in study[key].items():
            t.add_row(k, str(v["n"]), f"{v['win_rate_pct']}", f"{v['ev_pct']:+.3f}")
        console.print(t)

    t = Table(title="Regime policy effect (EV per $ deployed)", box=box.SIMPLE)
    t.add_column("Window"); t.add_column("Baseline N", justify="right"); t.add_column("Baseline EV %", justify="right")
    t.add_column("Policy N", justify="right"); t.add_column("Policy EV %", justify="right"); t.add_column("Skipped EV %", justify="right")
    for label, k in [("All", "policy"), ("First half", "policy_first_half"), ("Second half", "policy_second_half")]:
        p = study[k]
        fmt = lambda x: "n/a" if x["ev_pct"] is None else f"{x['ev_pct']:+.3f}"
        t.add_row(label, str(p["baseline"]["n"]), fmt(p["baseline"]), str(p["with_policy"]["n"]),
                  fmt(p["with_policy"]), fmt(p["skipped"]))
    console.print(t)
    console.print("[dim]Enable the policy (config/market_regime.yaml -> policy.enabled) only if it helps in BOTH halves.[/dim]\n")


@main.command(name="weekly-cycle")
@click.option("--phase", type=click.Choice(["auto", "scan", "trigger", "manage", "review"]), default="auto",
              help="auto picks the phases that apply right now (US Eastern time)")
@click.option("--config", "ws_config", default="weekend_swing.yaml", help="Weekend swing config")
@click.option("--regime-config", "rg_config", default="market_regime.yaml", help="Regime config file")
@click.option("--dry-run", is_flag=True, help="Log what would be submitted/closed; send no orders")
@click.option("--print-cron", is_flag=True, help="Print the crontab lines that run the weekly cycle, then exit")
def weekly_cycle(phase, ws_config, rg_config, dry_run, print_cron):
    """
    Weekly paper-trading cycle (Alpaca PAPER account only): scan -> Monday
    triggers + auto-submit -> stop enforcement / Friday close -> weekly review.
    Safe to run repeatedly (every phase is idempotent); schedule it with cron.
    """
    import pandas as pd

    from signal_engine.data.alpaca_client import AlpacaDataClient
    from signal_engine.data.news_client import FinnhubNewsClient
    from signal_engine.weekend_swing.data_store import IntradayBarStore
    from signal_engine.weekend_swing.tracker import WeekendSwingTracker
    from signal_engine.weekend_swing.weekly_cycle import CRON_LINES, WeeklyCycle

    if print_cron:
        click.echo(CRON_LINES)
        return

    ws_cfg, rg_cfg = _load_config(ws_config), _load_config(rg_config)
    try:
        alpaca = AlpacaDataClient()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    finnhub = None
    try:
        finnhub = FinnhubNewsClient()
    except EnvironmentError:
        pass

    stamp = lambda msg: console.print(f"[dim]{datetime.now().isoformat(timespec='seconds')}[/dim] {msg}", highlight=False)
    cycle = WeeklyCycle(ws_cfg, rg_cfg, alpaca, finnhub, IntradayBarStore(alpaca), WeekendSwingTracker(),
                        dry_run=dry_run, log=stamp)
    cycle.run(phase, pd.Timestamp.now(tz="UTC"))


# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────

@main.command()
@click.option("--host", default="127.0.0.1", help="Host address to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
def dashboard(host, port):
    """Start the interactive web dashboard server."""
    from signal_engine.dashboard.server import start_server
    
    console.print()
    console.rule("[bold cyan]signal-engine Web Dashboard[/bold cyan]")
    console.print(f"[bold green]Starting FastAPI server at http://{host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")
    
    start_server(host=host, port=port)


if __name__ == "__main__":
    main()
