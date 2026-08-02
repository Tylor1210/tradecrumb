"""
Unit tests for sentiment signal computation.

Tests use mock NewsItem objects — no API calls.
"""

from datetime import datetime, timedelta

import pytest

from signal_engine.data.news_client import NewsItem
from signal_engine.signals.sentiment import SentimentAnalyzer, SentimentSignal

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "lookback_days": 3,
    "recency_decay": 0.7,
    "use_finnhub_precomputed": True,
}


def make_news(
    ticker: str,
    headline: str,
    raw_sentiment_score: float = 0.0,
    age_hours: int = 1,
    summary: str = "",
) -> NewsItem:
    return NewsItem(
        ticker=ticker,
        headline=headline,
        summary=summary,
        published_at=datetime.now() - timedelta(hours=age_hours),
        source="test_source",
        url="https://test.example.com",
        raw_sentiment_score=raw_sentiment_score,
        category="company",
    )


def make_general_news(
    headline: str,
    raw_sentiment_score: float = 0.0,
    age_hours: int = 1,
) -> NewsItem:
    return NewsItem(
        ticker=None,
        headline=headline,
        summary="",
        published_at=datetime.now() - timedelta(hours=age_hours),
        source="newsapi",
        url="https://test.example.com",
        raw_sentiment_score=raw_sentiment_score,
        category="political",
    )


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestSentimentAnalyzer:

    def test_positive_precomputed_scores_yield_positive_result(self):
        """Strong positive Finnhub scores should result in a positive signal."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        news = [
            make_news("AAPL", "Apple smashes earnings expectations", raw_sentiment_score=0.85),
            make_news("AAPL", "Apple announces record share buyback", raw_sentiment_score=0.90),
        ]
        result = analyzer.compute("AAPL", news, datetime.now())
        assert isinstance(result, SentimentSignal)
        assert result.score > 0, f"Expected positive score, got {result.score}"

    def test_negative_precomputed_scores_yield_negative_result(self):
        """Strong negative scores should result in a negative signal."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        news = [
            make_news("TSLA", "Tesla faces DOJ criminal probe", raw_sentiment_score=-0.80),
            make_news("TSLA", "Tesla misses delivery targets by 30%", raw_sentiment_score=-0.75),
        ]
        result = analyzer.compute("TSLA", news, datetime.now())
        assert result.score < 0, f"Expected negative score, got {result.score}"

    def test_no_news_returns_zero_score(self):
        """No relevant news should return score of exactly 0.0."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        result = analyzer.compute("AAPL", [], datetime.now())
        assert result.score == 0.0
        assert result.breakdown.get("reason") == "no_news"

    def test_score_always_in_bounds(self):
        """Score must always be in [-1, 1] regardless of input values."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        # Extreme scores
        news = [make_news("AAPL", f"Headline {i}", raw_sentiment_score=1.0) for i in range(50)]
        result = analyzer.compute("AAPL", news, datetime.now())
        assert -1.0 <= result.score <= 1.0, f"Score out of bounds: {result.score}"

    def test_ticker_filter_excludes_other_tickers(self):
        """News for MSFT should not affect AAPL's score."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        news = [
            make_news("AAPL", "Apple great news", raw_sentiment_score=0.9),
            make_news("MSFT", "Microsoft terrible news", raw_sentiment_score=-0.95),
        ]
        result = analyzer.compute("AAPL", news, datetime.now())
        assert result.score > 0, f"MSFT's negative news should not affect AAPL. Got: {result.score}"

    def test_recency_weighting_recent_beats_old(self):
        """Recent news should outweigh older news of opposite sentiment."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        now = datetime.now()
        news = [
            # Very recent — strongly positive
            make_news("NVDA", "NVIDIA announces AI breakthrough", raw_sentiment_score=1.0, age_hours=1),
            # 2 days old — negative
            make_news("NVDA", "NVIDIA faces supply shortage", raw_sentiment_score=-1.0, age_hours=48),
        ]
        result = analyzer.compute("NVDA", news, now)
        assert result.score > 0, (
            "Recent positive news should outweigh 2-day-old negative news. "
            f"Got score: {result.score}"
        )

    def test_headlines_used_in_output(self):
        """Top headlines should appear in the result."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        headlines = [f"Headline {i}" for i in range(5)]
        news = [make_news("JPM", h, raw_sentiment_score=0.5, age_hours=i + 1) for i, h in enumerate(headlines)]
        result = analyzer.compute("JPM", news, datetime.now())
        assert len(result.headlines_used) > 0
        assert result.headlines_used[0] == headlines[0]  # most recent first

    def test_lookback_window_filters_old_news(self):
        """News older than lookback_days should be excluded."""
        config = dict(DEFAULT_CONFIG)
        config["lookback_days"] = 1  # only look back 1 day
        analyzer = SentimentAnalyzer(config)

        news = [
            make_news("AAPL", "Recent positive", raw_sentiment_score=0.8, age_hours=12),
            make_news("AAPL", "Old negative", raw_sentiment_score=-0.9, age_hours=72),  # 3 days old
        ]
        result = analyzer.compute("AAPL", news, datetime.now())
        assert result.score > 0, "Old negative news outside window should be excluded"

    def test_vader_fallback_detects_positive_sentiment(self):
        """When precomputed=False, VADER should detect positive sentiment."""
        config = dict(DEFAULT_CONFIG)
        config["use_finnhub_precomputed"] = False  # force VADER

        news = [
            make_news(
                "GOOGL",
                "Alphabet reports outstanding quarterly results, exceeding all expectations",
                raw_sentiment_score=0.0,  # no precomputed score
            )
        ]
        analyzer = SentimentAnalyzer(config)
        result = analyzer.compute("GOOGL", news, datetime.now())
        assert result.score > 0, f"VADER should detect positive sentiment. Got: {result.score}"

    def test_vader_fallback_detects_negative_sentiment(self):
        """When precomputed=False, VADER should detect negative sentiment."""
        config = dict(DEFAULT_CONFIG)
        config["use_finnhub_precomputed"] = False

        news = [
            make_news(
                "META",
                "Facebook faces catastrophic data breach, millions of users affected, stock plummets",
                raw_sentiment_score=0.0,
            )
        ]
        analyzer = SentimentAnalyzer(config)
        result = analyzer.compute("META", news, datetime.now())
        assert result.score < 0, f"VADER should detect negative sentiment. Got: {result.score}"

    def test_breakdown_contains_item_count(self):
        """Breakdown should always report how many articles were used."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        news = [make_news("V", "Visa news", raw_sentiment_score=0.3)]
        result = analyzer.compute("V", news, datetime.now())
        assert "item_count" in result.breakdown
        assert result.breakdown["item_count"] >= 1

    def test_mixed_ticker_and_general_news(self):
        """General news (ticker=None) supplements ticker-specific news."""
        analyzer = SentimentAnalyzer(DEFAULT_CONFIG)
        news = [
            make_news("AMZN", "Amazon earnings beat", raw_sentiment_score=0.7),
            make_general_news("Market rally continues", raw_sentiment_score=0.4),
        ]
        result = analyzer.compute("AMZN", news, datetime.now())
        # AMZN specific news dominates; general news supplements if < 3 items
        assert result.score > 0
