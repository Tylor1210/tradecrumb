"""
Live intraday day-trading trigger.

Each call to `run_once()` is one "tick": near the close it exits any open
intraday positions, otherwise it re-scans the watchlist and submits
fixed-notional paper day trades for anything crossing the execution
threshold that isn't already held.

Intended to be invoked repeatedly during market hours — either as a
foreground poll loop (`signal-engine live`) or as a single tick from cron
(`signal-engine live --once`). Only ever touches the Alpaca *paper* endpoint
via PaperTradingBroker.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)
EXIT_CUTOFF = (15, 45)  # start closing intraday positions 15 min before the close


def _at(now: datetime, hour: int, minute: int) -> datetime:
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def is_market_hours(now: Optional[datetime] = None) -> bool:
    now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    return _at(now, *MARKET_OPEN) <= now <= _at(now, *MARKET_CLOSE)


def is_near_close(now: Optional[datetime] = None) -> bool:
    now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    return _at(now, *EXIT_CUTOFF) <= now <= _at(now, *MARKET_CLOSE)


def _report_to_dict(report) -> dict:
    d = asdict(report)
    d["date"] = report.date.isoformat() if isinstance(report.date, datetime) else str(report.date)
    return d


class LiveDayTradeTrigger:
    """Signal-driven, fixed-notional paper day trading."""

    def __init__(self, pt_config: dict, sig_config: dict, wl_config: dict):
        from signal_engine.paper_trading.broker import PaperTradingBroker
        from signal_engine.paper_trading.tracker import TradeTracker

        self.pt_config = pt_config
        self.sig_config = sig_config
        self.wl_config = wl_config
        self.trade_size = float(pt_config.get("fixed_trade_size") or 1000.0)
        self.execution_threshold = float(pt_config.get("execution_threshold", 0.55))
        self.max_positions = int(pt_config.get("max_positions", 10))
        self.tracker = TradeTracker()
        self.broker = PaperTradingBroker()

    # ── Signal computation ───────────────────────────────────────────────

    def _compute_signals(self, scan_date: datetime) -> list:
        import os

        from signal_engine.data.alpaca_client import AlpacaDataClient
        from signal_engine.data.cache import ParquetCache
        from signal_engine.data.news_client import FinnhubNewsClient, NewsAPIClient
        from signal_engine.signals.composite import CompositeScorer
        from signal_engine.signals.political import PoliticalTriggerDetector
        from signal_engine.signals.sentiment import SentimentAnalyzer
        from signal_engine.signals.technical import TechnicalAnalyzer

        watchlist = self.wl_config.get("tickers", [])
        public_tickers = [t for t in watchlist if t.get("public", True)]
        symbols = [t["symbol"] for t in public_tickers]
        if not symbols:
            return []

        alpaca = AlpacaDataClient()
        cache = ParquetCache()
        finnhub = FinnhubNewsClient() if os.getenv("FINNHUB_API_KEY") else None
        newsapi = NewsAPIClient() if os.getenv("NEWSAPI_KEY") else None

        price_start = scan_date - timedelta(days=400)
        cache_key = cache.cache_key_for_bars(symbols, price_start, scan_date, "1Day")
        price_data = cache.get(cache_key, ttl_hours=1)  # short TTL — intraday wants fresh bars
        if not isinstance(price_data, dict):
            price_data = alpaca.get_historical_bars(symbols, price_start, scan_date)

        from_date_str = (scan_date - timedelta(days=3)).strftime("%Y-%m-%d")
        to_date_str = scan_date.strftime("%Y-%m-%d")
        all_news = []
        if finnhub:
            for sym in symbols[:15]:  # rate limit protection
                try:
                    all_news.extend(finnhub.get_company_news(sym, from_date_str, to_date_str))
                except Exception:
                    pass
            try:
                all_news.extend(finnhub.get_market_news("general"))
            except Exception:
                pass
        if newsapi:
            flat_kws = [
                kw
                for kws in self.sig_config.get("political", {}).get("political_keywords", {}).values()
                for kw in kws
            ]
            try:
                all_news.extend(newsapi.search_political_news(flat_kws[:6], from_date_str, to_date_str))
            except Exception:
                pass

        tech_analyzer = TechnicalAnalyzer(self.sig_config.get("technical", {}))
        sent_analyzer = SentimentAnalyzer(self.sig_config.get("sentiment", {}))
        watchlist_map = {t["symbol"]: t for t in watchlist}
        pol_detector = PoliticalTriggerDetector(self.sig_config.get("political", {}), watchlist_map)
        weights = self.sig_config.get("weights", {"technical": 0.40, "sentiment": 0.35, "political": 0.25})
        thresholds = self.sig_config.get("thresholds", {"buy_signal": 0.55, "sell_signal": -0.40})
        scorer = CompositeScorer(weights, thresholds)

        reports = []
        for ticker in symbols:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            technical = tech_analyzer.compute(ticker, df)
            sentiment = sent_analyzer.compute(ticker, all_news, scan_date)
            political = pol_detector.compute(ticker, all_news, scan_date)
            reports.append(scorer.compute(ticker, technical, sentiment, political))
        return reports

    # ── Exit ──────────────────────────────────────────────────────────────

    def exit_intraday_positions(self, now: datetime) -> list:
        closed = []
        for trade in self.tracker.get_open_trades():
            if trade.get("hold_period") != "intraday":
                continue
            try:
                order = self.broker.close_position(trade["ticker"])
                exit_price = order.get("filled_avg_price")
                if exit_price is None:
                    # Market order fills async — fall back to entry price as a placeholder;
                    # the reporter reconciles the real fill from Alpaca's order history.
                    exit_price = trade["entry_price"]
                outcome = self.tracker.log_exit(
                    trade["trade_id"], now, float(exit_price), notes="auto-exit: intraday close"
                )
                closed.append({"ticker": trade["ticker"], "trade_id": trade["trade_id"], "outcome_pct": outcome})
            except Exception as e:
                logger.warning("Exit failed for %s: %s", trade["ticker"], e)
        return closed

    # ── Entry ─────────────────────────────────────────────────────────────

    def enter_new_trades(self, now: datetime) -> list:
        open_tickers = {t["ticker"] for t in self.tracker.get_open_trades()}
        slots = self.max_positions - len(open_tickers)
        if slots <= 0:
            return []

        reports = self._compute_signals(now)
        candidates = [
            r for r in reports if r.composite_score >= self.execution_threshold and r.ticker not in open_tickers
        ]
        candidates.sort(key=lambda r: r.composite_score, reverse=True)
        candidates = candidates[:slots]

        entered = []
        for r in candidates:
            try:
                order = self.broker.submit_market_buy(r.ticker, notional=self.trade_size)
                entry_price = order.get("filled_avg_price")
                qty = order.get("qty") or (self.trade_size / entry_price if entry_price else 0)
                trade_id = self.tracker.log_entry(
                    ticker=r.ticker,
                    signal_date=now,
                    entry_date=now,
                    entry_price=float(entry_price) if entry_price else 0.0,
                    qty=float(qty or 0),
                    notional=self.trade_size,
                    recommendation="BUY",
                    composite_score=r.composite_score,
                    signal_breakdown=_report_to_dict(r),
                    hold_period="intraday",
                    trade_id=order.get("id"),
                )
                entered.append({"ticker": r.ticker, "trade_id": trade_id, "score": r.composite_score})
            except Exception as e:
                logger.warning("Entry failed for %s: %s", r.ticker, e)
        return entered

    # ── One tick ──────────────────────────────────────────────────────────

    def run_once(self, force: bool = False) -> dict:
        now = datetime.now(MARKET_TZ)
        result: dict = {"timestamp": now.isoformat(), "entered": [], "exited": []}

        if not force and not is_market_hours(now):
            result["skipped"] = "outside market hours"
            return result

        if not force and is_near_close(now):
            result["exited"] = self.exit_intraday_positions(now)
        else:
            result["entered"] = self.enter_new_trades(now)

        return result
