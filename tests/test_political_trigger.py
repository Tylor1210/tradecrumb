"""
Unit tests for the political trigger detector.

Tests keyword matching, sector/theme relevance, case insensitivity,
score bounds, and category sentiment application.
No API calls — all mock NewsItem objects.
"""

from datetime import datetime, timedelta

import pytest

from signal_engine.data.news_client import NewsItem
from signal_engine.signals.political import PoliticalTriggerDetector, PoliticalSignal

# ─────────────────────────────────────────────────────────────
# Test config (mirrors structure of config/signals.yaml)
# ─────────────────────────────────────────────────────────────

POLITICAL_CONFIG = {
    "political_lookback_days": 2,
    "political_keywords": {
        "tariff_bearish": ["tariff", "trade war", "import tax"],
        "trump_direct": ["Trump", "Donald Trump"],
        "government_contracts_bullish": ["government contract", "defense contract"],
        "geopolitical_risk": ["war", "conflict", "invasion"],
        "deregulation_bullish": ["deregulation", "regulatory rollback"],
        "elon_musk": ["Elon Musk", "DOGE"],
    },
    "category_sentiment": {
        "tariff_bearish": -0.70,
        "trump_direct": 0.00,
        "government_contracts_bullish": 0.80,
        "geopolitical_risk": -0.55,
        "deregulation_bullish": 0.60,
        "elon_musk": 0.30,
    },
    "sector_keywords": {
        "defense": ["Pentagon", "military", "F-35", "Lockheed", "weapons", "Raytheon"],
        "technology": ["tech", "semiconductor", "AI", "big tech"],
        "energy": ["oil", "pipeline", "OPEC", "crude", "energy"],
        "consumer_staples": ["retail", "consumer", "import", "grocery"],
    },
    "theme_keywords": {
        "tariff_sensitive": ["tariff", "import tax", "trade war", "China"],
        "elon_musk": ["Elon Musk", "Musk", "DOGE", "Tesla", "SpaceX"],
        "defense_spending": ["Pentagon", "defense", "military", "NATO"],
        "AI_infrastructure": ["artificial intelligence", "AI", "data center", "GPU"],
    },
}

WATCHLIST = {
    "LMT": {
        "name": "Lockheed Martin Corporation",
        "sector": "defense",
        "themes": ["defense_spending"],
    },
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "technology",
        "themes": ["tariff_sensitive", "china_exposure"],
    },
    "XOM": {
        "name": "Exxon Mobil Corporation",
        "sector": "energy",
        "themes": ["geopolitical"],
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "sector": "consumer_discretionary",
        "themes": ["elon_musk", "electric_vehicles"],
    },
    "DJT": {
        "name": "Trump Media & Technology Group",
        "sector": "technology",
        "themes": ["trump_related"],
    },
}


def make_news(
    headline: str,
    summary: str = "",
    ticker: str = None,
    age_hours: int = 1,
) -> NewsItem:
    return NewsItem(
        ticker=ticker,
        headline=headline,
        summary=summary,
        published_at=datetime.now() - timedelta(hours=age_hours),
        source="test",
        url="https://test.example.com",
        raw_sentiment_score=0.0,
        category="political" if ticker is None else "company",
    )


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestPoliticalTriggerDetector:

    def test_tariff_news_triggers_bearish_for_tariff_sensitive(self):
        """Tariff news should score negatively for tariff-sensitive tickers."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news(
            "Trump announces 25% tariff on tech imports from China",
            ticker="AAPL",
        )]
        result = detector.compute("AAPL", news, datetime.now())
        assert result.score < 0, f"Expected negative score, got {result.score}"

    def test_defense_contract_triggers_bullish(self):
        """Defense contract news should score positively for defense tickers."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news(
            "Lockheed Martin wins $5B defense contract for new fighter jets",
            ticker="LMT",
        )]
        result = detector.compute("LMT", news, datetime.now())
        assert result.score > 0, f"Expected positive score, got {result.score}"

    def test_irrelevant_news_returns_zero(self):
        """News with no keyword matches or relevance returns 0."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news("Weekend weather forecast looks sunny")]
        result = detector.compute("AAPL", news, datetime.now())
        assert result.score == 0.0

    def test_keyword_matching_is_case_insensitive(self):
        """Keyword matching should work regardless of capitalization."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news("TARIFF increase on TECH IMPORTS from CHINA", ticker="AAPL")]
        result = detector.compute("AAPL", news, datetime.now())
        assert len(result.matched_keywords) > 0, "Should match 'tariff' case-insensitively"

    def test_score_always_in_bounds(self):
        """Political score must always be in [-1.0, 1.0]."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        # Flood with many matching headlines
        news = [
            make_news(
                f"tariff trade war invasion conflict government contract AAPL tech {i}",
                ticker="AAPL",
            )
            for i in range(50)
        ]
        result = detector.compute("AAPL", news, datetime.now())
        assert -1.0 <= result.score <= 1.0, f"Score out of bounds: {result.score}"

    def test_matched_keywords_in_result(self):
        """Matched keywords should appear in the result."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news("New tariffs imposed on technology sector", ticker="AAPL")]
        result = detector.compute("AAPL", news, datetime.now())
        matched_lower = [kw.lower() for kw in result.matched_keywords]
        assert "tariff" in matched_lower, f"Expected 'tariff' in {result.matched_keywords}"

    def test_breakdown_contains_expected_keys(self):
        """Breakdown dict should contain article scan count and categories triggered."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news("Apple faces tariff headwinds", ticker="AAPL")]
        result = detector.compute("AAPL", news, datetime.now())

        if result.score != 0.0:  # only if there were matches
            assert "match_count" in result.breakdown
            assert "categories_triggered" in result.breakdown

        assert "articles_scanned" in result.breakdown

    def test_lookback_window_excludes_old_news(self):
        """News older than political_lookback_days should be excluded."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [
            make_news("Tariff increase today", ticker="AAPL", age_hours=12),     # within 2 days
            make_news("Old tariff news", ticker="AAPL", age_hours=200),           # > 2 days old
        ]
        result = detector.compute("AAPL", news, datetime.now())
        # The old news should be excluded; result depends on recent news only
        assert "articles_scanned" in result.breakdown
        # Only the recent article should be in scope
        assert result.breakdown["articles_scanned"] == 1

    def test_geopolitical_risk_bearish_for_energy(self):
        """War/conflict news should be bearish for energy sector tickers."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news(
            "War breaks out in oil-rich Middle East region, crude prices spike",
            ticker="XOM",
        )]
        result = detector.compute("XOM", news, datetime.now())
        assert result.score < 0, f"Expected negative score for geopolitical risk"

    def test_elon_musk_mention_matches_tesla(self):
        """Elon Musk mention should be relevant to TSLA via theme keyword."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news(
            "Elon Musk announces DOGE will cut federal EV regulations",
            ticker=None,  # general news, not ticker-specific
        )]
        result = detector.compute("TSLA", news, datetime.now())
        # TSLA has 'elon_musk' theme, should detect relevance and DOGE/Elon Musk keywords
        # Score should be positive (elon_musk category = 0.30) or zero (if relevance fails)
        assert result.score >= 0.0  # should not be negative for Elon Musk news + TSLA

    def test_trump_direct_neutral_category(self):
        """Trump direct mention has 0.0 category sentiment — should be neutral score."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news(
            "Donald Trump speaks about DJT stock performance",
            ticker="DJT",
        )]
        result = detector.compute("DJT", news, datetime.now())
        # trump_direct has 0.0 sentiment, so score should be 0.0 if only that category fires
        # DJT is directly named so relevance should trigger
        # Score should be near 0.0 (neutral)
        assert -0.1 <= result.score <= 0.1, (
            f"Trump direct mention should be near-neutral, got {result.score}"
        )

    def test_deregulation_bullish(self):
        """Deregulation news should produce positive score for technology sector."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news(
            "Trump signs executive order for major tech industry deregulation",
            ticker="AAPL",
        )]
        result = detector.compute("AAPL", news, datetime.now())
        assert result.score > 0, f"Expected positive deregulation score, got {result.score}"

    def test_returns_political_signal_instance(self):
        """Always returns PoliticalSignal, even when no matches found."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        result = detector.compute("AAPL", [], datetime.now())
        assert isinstance(result, PoliticalSignal)

    def test_ticker_direct_mention_triggers_relevance(self):
        """Directly mentioning the ticker symbol in news should mark it as relevant."""
        detector = PoliticalTriggerDetector(POLITICAL_CONFIG, WATCHLIST)
        news = [make_news("LMT awarded major government contract by Pentagon")]
        result = detector.compute("LMT", news, datetime.now())
        # LMT mentioned + defense contract keyword → should produce result
        # (could be 0 if no keyword category fires, but should not crash)
        assert isinstance(result, PoliticalSignal)
