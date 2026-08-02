"""
Composite signal scorer and ranker.

Combines technical, sentiment, political, and candlestick/chart-pattern
signals into a single inspectable composite score per ticker. All signal
components and their weights are visible in the output — no black box.

Output: SignalReport dataclass + ranked DataFrame for the daily scan.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from signal_engine.signals.technical import TechnicalSignal
from signal_engine.signals.sentiment import SentimentSignal
from signal_engine.signals.political import PoliticalSignal
from signal_engine.signals.patterns import PatternSignal

logger = logging.getLogger(__name__)


@dataclass
class SignalReport:
    """
    Full signal report for a single ticker on a single date.

    Every field here is inspectable — the composite score is not a black box.
    """

    ticker: str
    date: datetime
    composite_score: float
    recommendation: str      # "BUY" | "SHORT" | "HOLD"
    confidence: str          # "HIGH" | "MEDIUM" | "LOW"

    # Component scores (each in [-1, 1])
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    political_score: float = 0.0
    pattern_score: float = 0.0

    # Normalized weights used for this computation
    weight_technical: float = 0.0
    weight_sentiment: float = 0.0
    weight_political: float = 0.0
    weight_pattern: float = 0.0

    # Full breakdowns from each signal module
    technical_breakdown: dict = field(default_factory=dict)
    sentiment_breakdown: dict = field(default_factory=dict)
    political_breakdown: dict = field(default_factory=dict)
    pattern_breakdown: dict = field(default_factory=dict)

    # Human-readable rule triggers (the "why" in plain English)
    triggered_rules: list[str] = field(default_factory=list)

    # Top headlines driving sentiment/political signals
    sentiment_headlines: list[str] = field(default_factory=list)
    political_headlines: list[str] = field(default_factory=list)
    political_keywords: list[str] = field(default_factory=list)

    # Candlestick/chart patterns matched (e.g. "Bullish Engulfing")
    matched_patterns: list[dict] = field(default_factory=list)


class CompositeScorer:
    """
    Combines technical, sentiment, and political signals into a composite score.

    Weights are configurable via config/signals.yaml and normalized to sum to 1.0.
    """

    def __init__(self, weights: dict, thresholds: dict) -> None:
        """
        Args:
            weights:    {"technical": float, "sentiment": float, "political": float, "pattern": float}
            thresholds: {"buy_signal": float, "sell_signal": float}
        """
        total = sum(weights.values()) or 1.0
        self.w_tech = weights.get("technical", 0.30) / total
        self.w_sent = weights.get("sentiment", 0.25) / total
        self.w_pol = weights.get("political", 0.15) / total
        self.w_pattern = weights.get("pattern", 0.30) / total

        self.buy_threshold = thresholds.get("buy_signal", 0.55)
        self.sell_threshold = thresholds.get("sell_signal", -0.40)

    def compute(
        self,
        ticker: str,
        technical: Optional[TechnicalSignal],
        sentiment: Optional[SentimentSignal],
        political: Optional[PoliticalSignal],
        pattern: Optional[PatternSignal] = None,
    ) -> SignalReport:
        """
        Compute the composite signal report for a ticker.

        None signals are treated as score = 0.0.
        At least one signal must be provided.
        """
        tech_score = technical.score if technical else 0.0
        sent_score = sentiment.score if sentiment else 0.0
        pol_score = political.score if political else 0.0
        pattern_score = pattern.score if pattern else 0.0

        # Get date from whichever signal is available
        signal_date = None
        for sig in [technical, sentiment, political, pattern]:
            if sig is not None:
                signal_date = sig.date
                break
        if signal_date is None:
            signal_date = datetime.now()

        # Composite score
        composite = (
            self.w_tech * tech_score
            + self.w_sent * sent_score
            + self.w_pol * pol_score
            + self.w_pattern * pattern_score
        )
        composite = float(max(-1.0, min(1.0, composite)))

        # Recommendation — SHORT means "open a short position", not "sell an
        # existing long" (see paper_trading/live_trigger.py for execution).
        if composite >= self.buy_threshold:
            recommendation = "BUY"
        elif composite <= self.sell_threshold:
            recommendation = "SHORT"
        else:
            recommendation = "HOLD"

        # Confidence from signal agreement (technical, sentiment, political, pattern)
        scores_signed = [tech_score >= 0, sent_score >= 0, pol_score >= 0, pattern_score >= 0]
        agreement = sum(scores_signed)
        if agreement == 0 or agreement == 4:
            confidence = "HIGH"     # all four agree
        elif agreement == 1 or agreement == 3:
            confidence = "MEDIUM"   # three of four agree
        else:
            confidence = "LOW"      # split 2-2

        # ── Triggered rules (plain English) ─────────────────────────────
        triggered_rules: list[str] = []

        if technical:
            ma_breakdown = technical.breakdown.get("moving_averages", {})
            ma_signal = ma_breakdown.get("signal", "")

            if ma_signal == "full_bull_alignment":
                triggered_rules.append("✅ Price above SMA-20/50/200 — full bullish trend alignment")
            elif ma_signal == "full_bear_alignment":
                triggered_rules.append("🔴 Price below SMA-20/50/200 — full bearish trend alignment")
            elif ma_signal == "pullback_in_uptrend":
                triggered_rules.append("⚠️ Pullback in uptrend — price below fast MA but above SMA-200")

            rsi_info = technical.breakdown.get("rsi", {})
            rsi_signal = rsi_info.get("signal", "")
            rsi_val = rsi_info.get("rsi_value", "?")
            if "oversold" in rsi_signal:
                triggered_rules.append(f"✅ RSI oversold at {rsi_val} — potential mean reversion")
            elif "overbought" in rsi_signal:
                triggered_rules.append(f"🔴 RSI overbought at {rsi_val} — potential pullback")

            breakout_info = technical.breakdown.get("breakout", {})
            breakout_signal = breakout_info.get("signal", "")
            if breakout_signal == "resistance_breakout_bullish":
                triggered_rules.append(
                    f"✅ Resistance breakout above ${breakout_info.get('resistance_level', '?')}"
                )
            elif breakout_signal == "support_breakdown_bearish":
                triggered_rules.append(
                    f"🔴 Support breakdown below ${breakout_info.get('support_level', '?')}"
                )

            vol_info = technical.breakdown.get("volume", {})
            vol_signal = vol_info.get("signal", "")
            rel_vol = vol_info.get("relative_volume", 0)
            if "very_high_vol" in vol_signal:
                triggered_rules.append(
                    f"📊 Very high volume ({rel_vol:.1f}x avg) — {'up' if 'up' in vol_signal else 'down'} day"
                )
            elif "high_vol" in vol_signal:
                triggered_rules.append(
                    f"📊 High volume ({rel_vol:.1f}x avg) — {'up' if 'up' in vol_signal else 'down'} day"
                )

        if sentiment and abs(sent_score) > 0.05:
            count = sentiment.breakdown.get("item_count", 0)
            method = sentiment.breakdown.get("dominant_scoring_method", "")
            if sent_score > 0:
                triggered_rules.append(
                    f"✅ Positive news sentiment ({count} articles, method: {method})"
                )
            else:
                triggered_rules.append(
                    f"🔴 Negative news sentiment ({count} articles, method: {method})"
                )

        if political and abs(pol_score) > 0.0:
            kws = political.matched_keywords[:3]
            cats = political.breakdown.get("categories_triggered", [])
            if pol_score > 0:
                triggered_rules.append(
                    f"✅ Bullish political trigger — keywords: {', '.join(kws)} | categories: {', '.join(cats)}"
                )
            else:
                triggered_rules.append(
                    f"🔴 Bearish political trigger — keywords: {', '.join(kws)} | categories: {', '.join(cats)}"
                )

        if pattern and pattern.matched_patterns:
            for m in pattern.matched_patterns:
                icon = "✅" if m["direction"] == "bullish" else "🔴"
                triggered_rules.append(f"{icon} Pattern: {m['name']} ({m['direction']})")

        if not triggered_rules:
            triggered_rules.append("⚪ No significant triggers — HOLD signal")

        return SignalReport(
            ticker=ticker,
            date=signal_date,
            composite_score=round(composite, 4),
            recommendation=recommendation,
            confidence=confidence,
            technical_score=round(tech_score, 4),
            sentiment_score=round(sent_score, 4),
            political_score=round(pol_score, 4),
            pattern_score=round(pattern_score, 4),
            weight_technical=round(self.w_tech, 3),
            weight_sentiment=round(self.w_sent, 3),
            weight_political=round(self.w_pol, 3),
            weight_pattern=round(self.w_pattern, 3),
            technical_breakdown=technical.breakdown if technical else {},
            sentiment_breakdown=sentiment.breakdown if sentiment else {},
            political_breakdown=political.breakdown if political else {},
            pattern_breakdown=pattern.breakdown if pattern else {},
            triggered_rules=triggered_rules,
            sentiment_headlines=sentiment.headlines_used if sentiment else [],
            political_headlines=political.headlines if political else [],
            political_keywords=political.matched_keywords if political else [],
            matched_patterns=pattern.matched_patterns if pattern else [],
        )

    def rank_signals(self, reports: list[SignalReport]) -> pd.DataFrame:
        """
        Produce a ranked DataFrame of SignalReports sorted by composite score.
        BUY candidates at top, SHORT candidates at bottom.

        Returns columns:
            ticker, date, composite_score, recommendation, confidence,
            technical_score, sentiment_score, political_score, pattern_score,
            triggered_rules, top_headlines, matched_patterns
        """
        rows = []
        for r in reports:
            rows.append(
                {
                    "ticker": r.ticker,
                    "date": r.date,
                    "composite_score": r.composite_score,
                    "recommendation": r.recommendation,
                    "confidence": r.confidence,
                    "technical_score": r.technical_score,
                    "sentiment_score": r.sentiment_score,
                    "political_score": r.political_score,
                    "pattern_score": r.pattern_score,
                    "triggered_rules": " | ".join(r.triggered_rules),
                    "top_headlines": " | ".join(r.sentiment_headlines[:2]),
                    "political_keywords": ", ".join(r.political_keywords[:5]),
                    "matched_patterns": ", ".join(m["name"] for m in r.matched_patterns),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df
