"""
Sentiment signal computation.

Aggregates news articles for a ticker into a single sentiment score in [-1, 1].
Uses Finnhub's pre-computed sentiment scores as the primary source,
falling back to VADER NLP analysis when pre-computed scores are unavailable.

Scoring:
   1.0 = unanimously positive recent news
  -1.0 = unanimously negative recent news
   0.0 = no news or neutral

Recency weighting: news from today counts 1.0x; yesterday counts 0.7x;
the day before counts 0.49x; etc. (configurable decay factor).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from signal_engine.data.news_client import NewsItem

logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Aggregated news sentiment for a single ticker on a single date."""

    ticker: str
    date: datetime
    score: float            # composite score [-1, 1]
    headlines_used: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


class SentimentAnalyzer:
    """
    Computes a news sentiment score from a list of NewsItem objects.

    Scoring method:
      1. Filters articles by ticker and lookback window
      2. Applies recency decay weighting (newer = more weight)
      3. Uses Finnhub pre-computed score if available; VADER otherwise
      4. Returns weighted average in [-1, 1]
    """

    def __init__(self, config: dict) -> None:
        self.lookback_days = config.get("lookback_days", 3)
        self.recency_decay = config.get("recency_decay", 0.7)
        self.use_finnhub_precomputed = config.get("use_finnhub_precomputed", True)

        # Lazy-load VADER to avoid slow import at startup
        self._vader = None

    def _get_vader(self):
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("vaderSentiment not installed — using raw_sentiment_score only")
        return self._vader

    def compute(
        self,
        ticker: str,
        news_items: list[NewsItem],
        as_of: datetime,
    ) -> SentimentSignal:
        """
        Compute sentiment score for a ticker as of a given date.

        Args:
            ticker:     Ticker symbol to score
            news_items: All news items (will be filtered by ticker + date)
            as_of:      Reference date for lookback window

        Returns:
            SentimentSignal with score and breakdown
        """
        # Normalize as_of to UTC-aware so it can be compared with API timestamps
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        cutoff = as_of - timedelta(days=self.lookback_days)

        def _to_utc(dt: datetime) -> datetime:
            """Ensure datetime is UTC-aware (treat naive as UTC)."""
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        # Filter: only news for this ticker within the lookback window
        # Also include general news if Finnhub/NewsAPI gives no ticker field
        relevant = [
            n for n in news_items
            if _to_utc(n.published_at) >= cutoff
            and (n.ticker == ticker or n.ticker is None)
            and n.headline.strip()
        ]

        # Prefer ticker-specific news
        ticker_specific = [n for n in relevant if n.ticker == ticker]
        general = [n for n in relevant if n.ticker is None]

        # Use ticker-specific first; supplement with general only if sparse
        final_news = ticker_specific
        if len(final_news) < 3:
            final_news = ticker_specific + general[:5]

        if not final_news:
            return SentimentSignal(
                ticker=ticker,
                date=as_of,
                score=0.0,
                breakdown={
                    "reason": "no_news",
                    "item_count": 0,
                    "lookback_days": self.lookback_days,
                },
            )

        # Sort newest first for recency weighting
        final_news.sort(key=lambda n: _to_utc(n.published_at), reverse=True)

        scores: list[float] = []
        weights: list[float] = []
        headlines_used: list[str] = []
        scoring_methods: list[str] = []

        vader = self._get_vader()

        for i, item in enumerate(final_news):
            weight = self.recency_decay ** i

            # Choose scoring method
            if (
                self.use_finnhub_precomputed
                and abs(item.raw_sentiment_score) > 0.001
            ):
                score = float(item.raw_sentiment_score)
                method = "finnhub"
            elif vader is not None:
                text = f"{item.headline}. {item.summary}".strip()
                vader_result = vader.polarity_scores(text)
                score = float(vader_result["compound"])  # already [-1, 1]
                method = "vader"
            else:
                score = 0.0
                method = "unavailable"

            scores.append(score * weight)
            weights.append(weight)
            headlines_used.append(item.headline)
            scoring_methods.append(method)

        total_weight = sum(weights)
        if total_weight > 0:
            composite = sum(scores) / total_weight
        else:
            composite = 0.0

        composite = float(max(-1.0, min(1.0, composite)))

        # Identify dominant method
        method_counts = {}
        for m in scoring_methods:
            method_counts[m] = method_counts.get(m, 0) + 1
        dominant_method = max(method_counts, key=method_counts.get) if method_counts else "none"

        return SentimentSignal(
            ticker=ticker,
            date=as_of,
            score=round(composite, 4),
            headlines_used=headlines_used[:5],  # top 5 most recent
            breakdown={
                "item_count": len(final_news),
                "ticker_specific_count": len(ticker_specific),
                "general_news_count": len(general),
                "lookback_days": self.lookback_days,
                "dominant_scoring_method": dominant_method,
                "recency_decay_factor": self.recency_decay,
                "raw_weighted_score": round(composite, 4),
            },
        )
