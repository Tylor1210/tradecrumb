"""
Unit tests for the composite signal scorer.

Tests weight math, recommendation thresholds, confidence levels,
and the ranked DataFrame output.
"""

from datetime import datetime

import pandas as pd
import pytest

from signal_engine.signals.composite import CompositeScorer, SignalReport
from signal_engine.signals.technical import TechnicalSignal
from signal_engine.signals.sentiment import SentimentSignal
from signal_engine.signals.political import PoliticalSignal

# ─────────────────────────────────────────────────────────────
# Fixtures / Helpers
# ─────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {"technical": 0.40, "sentiment": 0.35, "political": 0.25}
DEFAULT_THRESHOLDS = {"buy_signal": 0.55, "sell_signal": -0.40}


def make_tech(score: float, ticker: str = "TEST") -> TechnicalSignal:
    return TechnicalSignal(
        ticker=ticker,
        date=pd.Timestamp.now(),
        score=score,
        breakdown={
            "moving_averages": {"signal": "neutral", "above_sma200": score > 0},
            "rsi": {"rsi_value": 50.0, "signal": "neutral"},
            "macd": {"crossover": "bullish" if score > 0 else "bearish"},
            "volume": {"signal": "normal_volume", "relative_volume": 1.0},
            "breakout": {"signal": "within_range"},
            "component_weights": {"ma": 0.3, "rsi": 0.2, "macd": 0.25, "volume": 0.1, "breakout": 0.15},
        },
    )


def make_sent(score: float, ticker: str = "TEST") -> SentimentSignal:
    return SentimentSignal(
        ticker=ticker,
        date=datetime.now(),
        score=score,
        headlines_used=["Test headline"] if score != 0 else [],
        breakdown={"item_count": 1, "dominant_scoring_method": "finnhub"},
    )


def make_pol(score: float, keywords: list = None, ticker: str = "TEST") -> PoliticalSignal:
    return PoliticalSignal(
        ticker=ticker,
        date=datetime.now(),
        score=score,
        matched_keywords=keywords or (["tariff"] if score < 0 else ["contract"]),
        breakdown={
            "match_count": 1,
            "categories_triggered": ["tariff_bearish"] if score < 0 else ["government_contracts_bullish"],
        },
    )


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestCompositeScorer:

    def test_returns_signal_report_instance(self):
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.5), make_sent(0.5), make_pol(0.5))
        assert isinstance(result, SignalReport)

    def test_buy_signal_fires_above_threshold(self):
        """All strongly positive signals → BUY."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.9), make_sent(0.9), make_pol(0.9))
        assert result.recommendation == "BUY"
        assert result.composite_score >= DEFAULT_THRESHOLDS["buy_signal"]

    def test_sell_signal_fires_below_threshold(self):
        """All strongly negative signals → SELL."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(-0.9), make_sent(-0.9), make_pol(-0.9))
        assert result.recommendation == "SELL"
        assert result.composite_score <= DEFAULT_THRESHOLDS["sell_signal"]

    def test_hold_for_mixed_signals(self):
        """Weak or mixed signals should result in HOLD."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.2), make_sent(-0.1), make_pol(0.0))
        assert result.recommendation == "HOLD"

    def test_composite_score_always_in_bounds(self):
        """Composite score must always be in [-1.0, 1.0]."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        test_cases = [
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (0.0, 0.0, 0.0),
        ]
        for t, s, p in test_cases:
            result = scorer.compute("TEST", make_tech(t), make_sent(s), make_pol(p))
            assert -1.0 <= result.composite_score <= 1.0, (
                f"Out of bounds for tech={t}, sent={s}, pol={p}: {result.composite_score}"
            )

    def test_weight_math_is_correct(self):
        """Composite score = weighted sum of components (normalized weights)."""
        weights = {"technical": 0.5, "sentiment": 0.3, "political": 0.2}
        scorer = CompositeScorer(weights, DEFAULT_THRESHOLDS)

        t_score, s_score, p_score = 0.8, 0.4, 0.2
        result = scorer.compute("TEST", make_tech(t_score), make_sent(s_score), make_pol(p_score))

        # Normalized: tech=0.5, sent=0.3, pol=0.2 (already sums to 1.0)
        expected = 0.5 * t_score + 0.3 * s_score + 0.2 * p_score
        assert abs(result.composite_score - expected) < 0.001, (
            f"Expected {expected:.4f}, got {result.composite_score:.4f}"
        )

    def test_weights_are_normalized(self):
        """Unnormalized weights should be automatically normalized."""
        weights = {"technical": 4.0, "sentiment": 3.5, "political": 2.5}  # sum=10
        scorer = CompositeScorer(weights, DEFAULT_THRESHOLDS)
        # Check normalized weights
        assert abs(scorer.w_tech - 0.40) < 0.01
        assert abs(scorer.w_sent - 0.35) < 0.01
        assert abs(scorer.w_pol - 0.25) < 0.01

    def test_handles_none_technical(self):
        """None technical signal should not crash — treated as 0."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", None, make_sent(0.6), make_pol(0.3))
        assert isinstance(result, SignalReport)
        assert result.technical_score == 0.0

    def test_handles_none_sentiment(self):
        """None sentiment signal treated as 0."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.7), None, make_pol(0.3))
        assert isinstance(result, SignalReport)
        assert result.sentiment_score == 0.0

    def test_handles_none_political(self):
        """None political signal treated as 0."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.7), make_sent(0.5), None)
        assert isinstance(result, SignalReport)
        assert result.political_score == 0.0

    def test_all_signals_agree_gives_high_confidence(self):
        """All three signals in same direction → HIGH confidence."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.8), make_sent(0.7), make_pol(0.6))
        assert result.confidence == "HIGH"

    def test_mixed_signals_gives_medium_confidence(self):
        """Two of three signals agree → MEDIUM confidence."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.8), make_sent(0.5), make_pol(-0.4))
        assert result.confidence == "MEDIUM"

    def test_triggered_rules_are_not_empty(self):
        """triggered_rules should always have at least one entry."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.0), make_sent(0.0), make_pol(0.0))
        assert len(result.triggered_rules) > 0

    def test_triggered_rules_contain_bullish_marker_for_buy(self):
        """BUY signal triggered rules should contain ✅ markers."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        tech = make_tech(0.9)
        tech.breakdown["moving_averages"]["signal"] = "full_bull_alignment"
        result = scorer.compute("TEST", tech, make_sent(0.8), make_pol(0.7))
        all_rules = " ".join(result.triggered_rules)
        assert "✅" in all_rules

    def test_political_keywords_in_report(self):
        """Political keywords should appear in the SignalReport."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        pol = make_pol(-0.7, keywords=["tariff", "trade war"])
        result = scorer.compute("TEST", make_tech(0.0), make_sent(0.0), pol)
        assert "tariff" in result.political_keywords or "trade war" in result.political_keywords

    def test_rank_signals_sorted_descending(self):
        """rank_signals should return DataFrame sorted by composite_score descending."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        reports = [
            scorer.compute("LOW", make_tech(-0.5), make_sent(-0.3), make_pol(-0.2)),
            scorer.compute("MID", make_tech(0.3), make_sent(0.2), make_pol(0.1)),
            scorer.compute("HIGH", make_tech(0.8), make_sent(0.7), make_pol(0.6)),
        ]
        df = scorer.rank_signals(reports)
        assert isinstance(df, pd.DataFrame)
        assert list(df["ticker"]) == ["HIGH", "MID", "LOW"], (
            f"Wrong order: {list(df['ticker'])}"
        )

    def test_rank_signals_returns_empty_for_no_reports(self):
        """rank_signals with empty list should return empty DataFrame."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        df = scorer.rank_signals([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_component_scores_stored_in_report(self):
        """All three component scores should be stored in the report."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.6), make_sent(0.4), make_pol(-0.3))
        assert result.technical_score == pytest.approx(0.6, abs=0.001)
        assert result.sentiment_score == pytest.approx(0.4, abs=0.001)
        assert result.political_score == pytest.approx(-0.3, abs=0.001)

    def test_weights_stored_in_report(self):
        """Normalized weights used for computation should be in report."""
        scorer = CompositeScorer(DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS)
        result = scorer.compute("TEST", make_tech(0.5), make_sent(0.5), make_pol(0.5))
        assert result.weight_technical == pytest.approx(0.40, abs=0.01)
        assert result.weight_sentiment == pytest.approx(0.35, abs=0.01)
        assert result.weight_political == pytest.approx(0.25, abs=0.01)
