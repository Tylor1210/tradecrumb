"""
FastAPI Dashboard Server for signal-engine.

Serves the REST API endpoints and static assets.
Provides real-time watchlist scans, backtesting simulations,
and paper-trading dashboards.

Enforces a graceful fallback to simulated demo data when API keys are missing,
allowing immediate interactive exploration of the dashboard.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path to support running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signal_engine.data.alpaca_client import AlpacaDataClient
from signal_engine.data.news_client import FinnhubNewsClient, NewsAPIClient, NewsItem
from signal_engine.data.cache import ParquetCache
from signal_engine.signals.technical import TechnicalAnalyzer
from signal_engine.signals.sentiment import SentimentAnalyzer
from signal_engine.signals.political import PoliticalTriggerDetector
from signal_engine.signals.composite import CompositeScorer

from signal_engine.backtest.engine import BacktestConfig, BacktestEngine
from signal_engine.backtest.metrics import compute_directional_accuracy
from signal_engine.paper_trading.tracker import TradeTracker
from signal_engine.leaps.candidate_screener import LEAPSScreener
from signal_engine.trader_tracking.tracker import TraderCallTracker

logger = logging.getLogger("signal_engine.dashboard")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="signal-engine Dashboard",
    description="Interactive research and paper-trading terminal.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# State / Config Helpers
# ─────────────────────────────────────────────────────────────

def get_watchlist() -> list[dict]:
    wl_path = Path("config/watchlist.yaml")
    if not wl_path.exists():
        return []
    with open(wl_path) as f:
        cfg = yaml.safe_load(f) or {}
        return cfg.get("tickers", [])

def get_signals_config() -> dict:
    sig_path = Path("config/signals.yaml")
    if not sig_path.exists():
        return {}
    with open(sig_path) as f:
        return yaml.safe_load(f) or {}

def get_backtest_config() -> dict:
    bt_path = Path("config/backtest.yaml")
    if not bt_path.exists():
        return {}
    with open(bt_path) as f:
        return yaml.safe_load(f) or {}

def has_credentials() -> bool:
    """Check if minimum API keys are set in environment."""
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


# ─────────────────────────────────────────────────────────────
# REST API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return API key configuration status."""
    return {
        "alpaca_connected": has_credentials(),
        "finnhub_connected": bool(os.getenv("FINNHUB_API_KEY")),
        "newsapi_connected": bool(os.getenv("NEWSAPI_KEY")),
        "polygon_connected": bool(os.getenv("POLYGON_API_KEY")),
        "alpaca_paper_connected": bool(os.getenv("ALPACA_PAPER_API_KEY") and os.getenv("ALPACA_PAPER_SECRET_KEY")),
        "timestamp": datetime.now().isoformat(),
        "mode": "REAL-TIME" if has_credentials() else "SIMULATED_DEMO",
    }

@app.get("/api/watchlist")
def api_watchlist():
    """Get the current watchlist of tickers."""
    return get_watchlist()

@app.get("/api/ticker_history/{symbol}")
def api_ticker_history(symbol: str):
    """
    Get 50 days of pricing history + indicators (SMA-20/50/200, rolling support/resistance)
    for a given ticker. Used to draw the pop-up technical analysis chart.
    """
    symbol = symbol.upper()
    
    if not has_credentials():
        # High quality simulated daily prices & indicators
        import random
        random.seed(hash(symbol))
        
        # Determine drift based on ticker mock composite score
        watchlist = get_watchlist()
        ticker_info = next((t for t in watchlist if t["symbol"] == symbol), {})
        h = hash(symbol)
        comp_score = round(((h % 200) - 100) / 100.0, 3)
        drift = comp_score * 0.4  # positive composite = upward trend
        
        # Realistic AAPL current price matching the mock scan results
        base_price = 185.0 if symbol == "AAPL" else (250.0 if symbol == "TSLA" else 150.0)
        
        closes = []
        highs = []
        lows = []
        curr = base_price
        
        # Generate 100 days so we can calculate rolling stats for the last 50 days
        for i in range(100):
            # daily steps
            change = (random.random() - 0.45 + (drift / 50)) * 0.02
            curr = max(10.0, curr * (1 + change))
            closes.append(curr)
            highs.append(curr * (1 + random.random() * 0.015))
            lows.append(curr * (1 - random.random() * 0.015))
            
        import pandas as pd
        df = pd.DataFrame({"close": closes, "high": highs, "low": lows})
        sma20 = df["close"].rolling(20).mean()
        sma50 = df["close"].rolling(50).mean()
        sma200 = df["close"].rolling(200).mean().fillna(df["close"].rolling(50).mean() * 0.95) # mock fallback
        res20 = df["high"].rolling(20).max()
        sup20 = df["low"].rolling(20).min()
        
        # Get last 50 days
        start_idx = 50
        history = []
        start_date = datetime.now() - timedelta(days=70)
        
        for i in range(start_idx, 100):
            date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            history.append({
                "date": date_str,
                "close": round(df["close"].iloc[i], 2),
                "sma20": round(sma20.iloc[i], 2) if not pd.isna(sma20.iloc[i]) else None,
                "sma50": round(sma50.iloc[i], 2) if not pd.isna(sma50.iloc[i]) else None,
                "sma200": round(sma200.iloc[i], 2) if not pd.isna(sma200.iloc[i]) else None,
                "support": round(sup20.iloc[i], 2) if not pd.isna(sup20.iloc[i]) else None,
                "resistance": round(res20.iloc[i], 2) if not pd.isna(res20.iloc[i]) else None,
            })
            
        return history

    try:
        # Real market data fetching
        alpaca = AlpacaDataClient()
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)  # enough for SMA-200
        
        price_data = alpaca.get_historical_bars([symbol], start_dt, end_dt)
        df = price_data.get(symbol)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
            
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        res20 = high.rolling(20).max()
        sup20 = low.rolling(20).min()
        
        # Take last 50 trading days
        history = []
        for i in range(-50, 0):
            try:
                idx = close.index[i]
                history.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": round(float(close.iloc[i]), 2),
                    "sma20": round(float(sma20.iloc[i]), 2) if not pd.isna(sma20.iloc[i]) else None,
                    "sma50": round(float(sma50.iloc[i]), 2) if not pd.isna(sma50.iloc[i]) else None,
                    "sma200": round(float(sma200.iloc[i]), 2) if not pd.isna(sma200.iloc[i]) else None,
                    "support": round(float(sup20.iloc[i]), 2) if not pd.isna(sup20.iloc[i]) else None,
                    "resistance": round(float(res20.iloc[i]), 2) if not pd.isna(res20.iloc[i]) else None,
                })
            except IndexError:
                continue
                
        return history
        
    except Exception as e:
        logger.error("Ticker history failed for %s: %s", symbol, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scan")
def api_scan(date: Optional[str] = Query(None, description="Scan date YYYY-MM-DD")):
    """
    Run daily signal scan for all watchlist tickers.
    Falls back to high-quality simulated data if keys are missing.
    """
    scan_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    watchlist = get_watchlist()
    public_tickers = [t for t in watchlist if t.get("public", True)]
    symbols = [t["symbol"] for t in public_tickers]

    if not symbols:
        return {"error": "Watchlist is empty."}

    # Fallback to simulated data if no credentials
    if not has_credentials():
        import random
        random.seed(scan_date.toordinal())
        simulated_results = []
        for i, t in enumerate(public_tickers):
            sym = t["symbol"]
            # Stable score per ticker per day
            h = hash(sym + str(scan_date.date()))
            comp_score = round(((h % 200) - 100) / 100.0, 3) # [-1, 1]
            tech = round(((h % 70) - 35) / 50.0, 3)
            sent = round(((h % 90) - 45) / 50.0, 3)
            pol = round(((h % 110) - 55) / 60.0, 3)

            rec = "BUY" if comp_score >= 0.55 else "SELL" if comp_score <= -0.40 else "HOLD"
            conf = "HIGH" if (tech > 0 and sent > 0 and pol > 0) or (tech < 0 and sent < 0 and pol < 0) else "MEDIUM"

            rules = []
            if rec == "BUY":
                rules = [
                    f"✅ Price above SMA-20/50/200 — bullish trend alignment",
                    f"✅ Positive sentiment on Finnhub news",
                    f"✅ Bullish government contract wins"
                ]
            elif rec == "SELL":
                rules = [
                    f"🔴 Price below SMA-20/50/200 — death cross trend",
                    f"🔴 Bearish political trigger: new supply chain tariffs",
                    f"🔴 RSI overbought at 76.5 — mean reversion indicator"
                ]
            else:
                rules = ["⚪ No significant triggers — HOLD signal"]

            simulated_results.append({
                "ticker": sym,
                "name": t.get("name", sym),
                "sector": t.get("sector", "unknown"),
                "date": scan_date.date().isoformat(),
                "composite_score": comp_score,
                "recommendation": rec,
                "confidence": conf,
                "technical_score": tech,
                "sentiment_score": sent,
                "political_score": pol,
                "triggered_rules": " | ".join(rules),
                "top_headlines": f"General news for {sym} shows stable outlook. | Retail segment performs strongly.",
                "political_keywords": "tariff, contract" if pol != 0 else "",
            })
        # Sort by composite score descending
        simulated_results.sort(key=lambda x: x["composite_score"], reverse=True)
        return {
            "mode": "SIMULATED_DEMO",
            "results": simulated_results,
        }

    # Real implementation using loaded clients
    try:
        alpaca = AlpacaDataClient()
        finnhub = FinnhubNewsClient() if os.getenv("FINNHUB_API_KEY") else None
        newsapi = NewsAPIClient() if os.getenv("NEWSAPI_KEY") else None
        cache = ParquetCache()

        sig_cfg = get_signals_config()
        tech_config = sig_cfg.get("technical", {})
        sent_config = sig_cfg.get("sentiment", {})
        pol_config = sig_cfg.get("political", {})
        weights = sig_cfg.get("weights", {"technical": 0.40, "sentiment": 0.35, "political": 0.25})
        thresholds = sig_cfg.get("thresholds", {"buy_signal": 0.55, "sell_signal": -0.40})

        tech_analyzer = TechnicalAnalyzer(tech_config)
        sent_analyzer = SentimentAnalyzer(sent_config)
        watchlist_map = {t["symbol"]: t for t in watchlist}
        pol_detector = PoliticalTriggerDetector(pol_config, watchlist_map)
        scorer = CompositeScorer(weights, thresholds)

        # Load prices
        price_start = scan_date - timedelta(days=400)
        cache_key = cache.cache_key_for_bars(symbols, price_start, scan_date, "1Day")
        price_data = cache.get(cache_key, ttl_hours=24)
        if price_data is None:
            price_data = alpaca.get_historical_bars(symbols, price_start, scan_date)
        elif not isinstance(price_data, dict):
            price_data = alpaca.get_historical_bars(symbols, price_start, scan_date)

        # Load news
        from_date_str = (scan_date - timedelta(days=3)).strftime("%Y-%m-%d")
        to_date_str = scan_date.strftime("%Y-%m-%d")
        all_news = []
        if finnhub:
            for sym in symbols[:15]:  # protect rate limit
                all_news.extend(finnhub.get_company_news(sym, from_date_str, to_date_str))
            all_news.extend(finnhub.get_market_news("general"))
        if newsapi:
            flat_kws = [kw for kws in pol_config.get("political_keywords", {}).values() for kw in kws]
            all_news.extend(newsapi.search_political_news(flat_kws[:6], from_date_str, to_date_str))

        reports = []
        for ticker in symbols:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            technical = tech_analyzer.compute(ticker, df)
            sentiment = sent_analyzer.compute(ticker, all_news, scan_date)
            political = pol_detector.compute(ticker, all_news, scan_date)
            report = scorer.compute(ticker, technical, sentiment, political)
            reports.append(report)

        ranked = scorer.rank_signals(reports)
        res_list = []
        for i, row in ranked.iterrows():
            res_list.append(row.to_dict())

        return {
            "mode": "REAL-TIME",
            "results": res_list,
        }

    except Exception as e:
        logger.error("Real-time scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest")
def api_backtest():
    """
    Run backtest or fetch metrics.
    Generates high-quality simulated curve if no real price data.
    """
    if not has_credentials():
        # High quality simulated equity curve
        import random
        random.seed(42)
        base_val = 100000.0
        curve = []
        start_date = datetime(2020, 1, 1)
        curr_val = base_val
        for i in range(120): # 10 years monthly / 120 weeks
            date_str = (start_date + timedelta(weeks=i*2)).strftime("%Y-%m-%d")
            # general upward drift with volatility
            change = (random.random() - 0.42) * 0.05
            curr_val = curr_val * (1 + change)
            curve.append({"date": date_str, "value": round(curr_val, 2)})

        return {
            "mode": "SIMULATED_DEMO",
            "metrics": {
                "total_return_pct": 74.32,
                "win_rate_pct": 72.5,
                "meets_70pct_target": True,
                "total_trades": 88,
                "winning_trades": 64,
                "losing_trades": 24,
                "max_drawdown_pct": 12.4,
                "sharpe_ratio": 1.64,
                "directional_accuracy_pct": 71.4,
            },
            "equity_curve": curve,
        }

    try:
        alpaca = AlpacaDataClient()
        bt_cfg = get_backtest_config()
        sig_cfg = get_signals_config()
        watchlist = get_watchlist()
        symbols = [t["symbol"] for t in watchlist if t.get("public", True)]

        start_dt = datetime.strptime(bt_cfg.get("start_date", "2020-01-01"), "%Y-%m-%d")
        end_dt = datetime.strptime(bt_cfg.get("end_date", "2025-01-01"), "%Y-%m-%d")

        price_data = alpaca.get_historical_bars(symbols, start_dt, end_dt)

        tech_analyzer = TechnicalAnalyzer(sig_cfg.get("technical", {}))
        buy_threshold = sig_cfg.get("thresholds", {}).get("buy_signal", 0.55)

        import pandas as pd
        signals = {}
        for ticker, df in price_data.items():
            sig_dates = []
            sig_vals = []
            for i in range(200, len(df)):
                sub = df.iloc[:i + 1]
                ts = tech_analyzer.compute(ticker, sub)
                sig_dates.append(df.index[i])
                sig_vals.append(ts.score >= buy_threshold if ts else False)
            if sig_dates:
                signals[ticker] = pd.Series(sig_vals, index=sig_dates)

        bt_config_obj = BacktestConfig(
            start_date=bt_cfg.get("start_date", "2020-01-01"),
            end_date=bt_cfg.get("end_date", "2025-01-01"),
            hold_period_days=bt_cfg.get("hold_period_days", 5),
            stop_loss_pct=bt_cfg.get("stop_loss_pct", 5.0),
            take_profit_pct=bt_cfg.get("take_profit_pct", 10.0),
            initial_capital=bt_cfg.get("initial_capital", 100000.0),
            position_size_pct=bt_cfg.get("position_size_pct", 5.0),
        )

        engine = BacktestEngine(bt_config_obj)
        result = engine.run(price_data, signals)
        metrics = result.get_metrics()

        dir_acc_values = []
        for ticker, df in price_data.items():
            if ticker in signals:
                da = compute_directional_accuracy(df, signals[ticker], bt_config_obj.hold_period_days)
                dir_acc_values.append(da)
        if dir_acc_values:
            metrics.directional_accuracy_pct = round(sum(dir_acc_values) / len(dir_acc_values), 2)

        # Format equity curve for rendering
        vbt_curve = result.portfolio.value()
        # Resample curve to avoid sending thousands of points to JS
        vbt_curve = vbt_curve.resample("W").last().dropna()
        curve = [{"date": idx.strftime("%Y-%m-%d"), "value": round(val, 2)} for idx, val in vbt_curve.items()]

        return {
            "mode": "REAL-TIME",
            "metrics": metrics.to_dict(),
            "equity_curve": curve,
        }

    except Exception as e:
        logger.error("Backtest endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/paper")
def api_paper():
    """Get paper trading positions, account data, and persistent SQL trade log."""
    tracker = TradeTracker()
    summary = tracker.get_performance_summary()
    open_trades = tracker.get_open_trades()
    closed_trades = tracker.get_closed_trades()

    # Try connecting to live Alpaca Paper API
    account_info = None
    positions = []
    try:
        if os.getenv("ALPACA_PAPER_API_KEY"):
            from signal_engine.paper_trading.broker import PaperTradingBroker
            broker = PaperTradingBroker()
            account_info = broker.get_account()
            positions = broker.get_positions()
    except Exception as e:
        logger.debug("Alpaca broker fetch failed: %s", e)

    # Mock brokerage status if not connected
    if account_info is None:
        account_info = {
            "portfolio_value": 105432.10,
            "buying_power": 210864.20,
            "cash": 45432.10,
            "status": "SIMULATED_ACTIVE",
            "paper": True,
        }
        # If open_trades is empty, add a mock live position
        if not open_trades:
            positions = [{
                "symbol": "NVDA",
                "qty": 50.0,
                "avg_entry_price": 120.50,
                "current_price": 124.30,
                "unrealized_pl": 190.00,
                "unrealized_plpc": 0.0315,
                "market_value": 6215.00,
            }]

    return {
        "account": account_info,
        "positions": positions,
        "logged_open_trades": open_trades,
        "logged_closed_trades": closed_trades,
        "performance_summary": summary,
    }

@app.get("/api/leaps")
def api_leaps():
    """Run LEAPS options screening and return candidates."""
    watchlist = get_watchlist()
    public_tickers = [t for t in watchlist if t.get("public", True)]
    symbols = [t["symbol"] for t in public_tickers]

    if not has_credentials():
        # Renders beautiful mock LEAPS candidates
        sim_candidates = []
        import random
        random.seed(99)
        for t in public_tickers[:8]:
            sym = t["symbol"]
            score = round(random.uniform(0.4, 0.95), 3)
            cats = [
                "Relative Strength vs SPY: +14.5% over 180d",
                f"Active macro theme: {t.get('themes', ['AI_infrastructure'])[0].replace('_', ' ').title()}"
            ]
            if score >= 0.7:
                cats.append("Positive earnings trend: 3 consecutive quarters")

            sim_candidates.append({
                "ticker": sym,
                "leaps_score": score,
                "current_price": round(random.uniform(50, 400), 2),
                "suggested_strike": round(random.uniform(55, 440), 2),
                "suggested_expiry_min_dte": 365,
                "catalysts": cats,
                "options_data_available": False,
                "breakdown": {
                    "relative_strength": {"score": score * 0.9},
                    "earnings_trend": {"score": score * 0.8},
                    "macro_theme_alignment": {"score": 1.0},
                    "long_term_trend": {"score": score * 0.75},
                }
            })
        sim_candidates.sort(key=lambda c: c["leaps_score"], reverse=True)
        return {
            "mode": "SIMULATED_DEMO",
            "candidates": sim_candidates,
        }

    try:
        alpaca = AlpacaDataClient()
        finnhub = FinnhubNewsClient() if os.getenv("FINNHUB_API_KEY") else None
        sig_cfg = get_signals_config()
        leaps_cfg = sig_cfg.get("leaps", {})

        start_dt = datetime.now() - timedelta(days=400)
        end_dt = datetime.now()

        fetch_symbols = list(set(symbols + ["SPY"]))
        price_data = alpaca.get_historical_bars(fetch_symbols, start_dt, end_dt)

        spy_df = price_data.get("SPY")
        if spy_df is None or spy_df.empty:
            raise HTTPException(status_code=500, detail="Could not fetch SPY data for relative strength.")

        screener = LEAPSScreener(leaps_cfg, finnhub_client=finnhub)
        watchlist_map = {t["symbol"]: t for t in watchlist}

        candidates = []
        for ticker in symbols:
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue
            cand = screener.screen(ticker, df, spy_df, watchlist_map.get(ticker, {}))
            # Convert LEAPSCandidate to a dict
            candidates.append({
                "ticker": cand.ticker,
                "leaps_score": cand.leaps_score,
                "direction": cand.direction,
                "direction_rationale": cand.direction_rationale,
                "current_price": cand.current_price,
                "suggested_strike": cand.suggested_strike,
                "suggested_expiry_min_dte": cand.suggested_expiry_min_dte,
                "catalysts": cand.catalysts,
                "options_data_available": cand.options_data_available,
                "hist_volatility": cand.hist_volatility,
                "target_price_1yr": cand.target_price_1yr,
                "target_upside_pct": cand.target_upside_pct,
                "est_call_premium": cand.est_call_premium,
                "est_cost_per_contract": cand.est_cost_per_contract,
                "est_return_on_1k": cand.est_return_on_1k,
                "est_return_pct_on_1k": cand.est_return_pct_on_1k,
                "breakdown": cand.breakdown,
            })

        candidates.sort(key=lambda c: c["leaps_score"], reverse=True)
        return {
            "mode": "REAL-TIME",
            "candidates": candidates,
        }

    except Exception as e:
        logger.error("LEAPS screening failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/traders/calls")
def api_traders_list_calls(
    trader_name: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """List logged trader calls, optionally filtered by trader/ticker/status."""
    tracker = TraderCallTracker()
    return tracker.list_calls(trader_name=trader_name, ticker=ticker, status=status)


@app.get("/api/traders/summary")
def api_traders_summary():
    """Per-trader win rate summary across all logged calls."""
    tracker = TraderCallTracker()
    return tracker.get_trader_summary()


class TraderCallCreate(BaseModel):
    trader_name: str
    platform: str  # discord | x | instagram | other
    ticker: str
    call_type: str = "BUY"  # BUY | SELL | WATCH
    handle: Optional[str] = None
    price_at_call: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    timeframe: str = "unspecified"
    notes: Optional[str] = None
    source_url: Optional[str] = None
    called_at: Optional[str] = None  # ISO date/datetime; defaults to now


@app.post("/api/traders/calls")
def api_traders_add_call(call: TraderCallCreate):
    """Log a trade call you saw from an outside trader (manual entry)."""
    tracker = TraderCallTracker()
    called_at = datetime.fromisoformat(call.called_at) if call.called_at else datetime.now()
    call_id = tracker.add_call(
        trader_name=call.trader_name,
        platform=call.platform,
        ticker=call.ticker,
        called_at=called_at,
        call_type=call.call_type,
        handle=call.handle,
        price_at_call=call.price_at_call,
        target_price=call.target_price,
        stop_price=call.stop_price,
        timeframe=call.timeframe,
        notes=call.notes,
        source_url=call.source_url,
    )
    return {"call_id": call_id}


class TraderCallClose(BaseModel):
    exit_price: float
    notes: Optional[str] = None


@app.post("/api/traders/calls/{call_id}/close")
def api_traders_close_call(call_id: str, body: TraderCallClose):
    """Close a call with an exit price and compute its outcome %."""
    tracker = TraderCallTracker()
    outcome = tracker.close_call(call_id, exit_price=body.exit_price, notes=body.notes)
    if outcome is None and not tracker.list_calls():
        raise HTTPException(status_code=404, detail="Call not found")
    return {"call_id": call_id, "outcome_pct": outcome}


@app.delete("/api/traders/calls/{call_id}")
def api_traders_delete_call(call_id: str):
    """Delete a logged call (e.g. entered by mistake)."""
    tracker = TraderCallTracker()
    deleted = tracker.delete_call(call_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Call not found")
    return {"deleted": True}


# ─────────────────────────────────────────────────────────────
# STATIC STATIC ASSETS ROUTING
# ─────────────────────────────────────────────────────────────

@app.get("/")
def get_root():
    """Serve index.html at root url."""
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

# Mount remaining static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────
# LAUNCH SERVER HELPER
# ─────────────────────────────────────────────────────────────

def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    logger.info("Starting signal-engine dashboard server at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
