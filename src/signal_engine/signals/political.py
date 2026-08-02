"""
Political trigger detector.

Detects political and geopolitical news that may impact specific tickers
using keyword/entity matching against configurable keyword lists.

How it works:
  1. Scans all recent news (company + general/political)
  2. For each article, checks if it is RELEVANT to the target ticker
     (via ticker mention, sector keywords, or theme keywords)
  3. If relevant, checks which political keyword categories match
  4. Applies the category's sentiment score (+/-) to build a composite

Scoring:
   1.0 = strong bullish political catalyst (e.g. defense contract won)
  -1.0 = strong bearish political catalyst (e.g. new tariffs on sector)
   0.0 = no political matches or neutral

Upgrade path:
  # TODO(upgrade): Replace keyword matching with an LLM-based classifier
  # (e.g., Gemini Flash via AI Studio API) for nuanced named-entity +
  # contextual sentiment understanding. The classifier would:
  #   1. Identify political entities mentioned (person, agency, law)
  #   2. Understand the context (is Trump praising or attacking the company?)
  #   3. Classify the likely market impact with a probability and explanation
  #   4. Handle ambiguous cases (e.g. "Trump tweets about Tesla" can be +/-)
  # Hook point: replace _score_matches() and _is_ticker_relevant() below.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from signal_engine.data.news_client import NewsItem

logger = logging.getLogger(__name__)


@dataclass
class PoliticalSignal:
    """Political/geopolitical trigger signal for a single ticker."""

    ticker: str
    date: datetime
    score: float            # composite score [-1, 1]
    matched_keywords: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


class PoliticalTriggerDetector:
    """
    Keyword/entity-based political news trigger.

    # TODO(upgrade): Replace with LLM classifier for context-aware scoring.
    # See module docstring for upgrade path details.
    """

    def __init__(self, config: dict, watchlist: dict[str, dict]) -> None:
        """
        Args:
            config:    political section from config/signals.yaml
            watchlist: dict mapping ticker → {sector, themes, name, ...}
        """
        self.config = config
        self.watchlist = watchlist
        self.political_keywords: dict[str, list[str]] = config.get("political_keywords", {})
        self.category_sentiment: dict[str, float] = config.get("category_sentiment", {})
        self.sector_keywords: dict[str, list[str]] = config.get("sector_keywords", {})
        self.theme_keywords: dict[str, list[str]] = config.get("theme_keywords", {})
        self.lookback_days = config.get("political_lookback_days", 2)

    def compute(
        self,
        ticker: str,
        all_news: list[NewsItem],
        as_of: datetime,
    ) -> PoliticalSignal:
        """
        Compute political signal for a ticker.

        Args:
            ticker:   Target ticker symbol
            all_news: All news items (company + general/political)
            as_of:    Reference date

        Returns:
            PoliticalSignal with score, matched keywords, and breakdown
        """
        # Normalize as_of to UTC-aware so it can be compared with API timestamps
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        def _to_utc(dt: datetime) -> datetime:
            """Ensure datetime is UTC-aware (treat naive as UTC)."""
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        cutoff = as_of - timedelta(days=self.lookback_days)
        recent_news = [n for n in all_news if _to_utc(n.published_at) >= cutoff]

        ticker_info = self.watchlist.get(ticker, {})
        ticker_sector = ticker_info.get("sector", "")
        ticker_themes = ticker_info.get("themes", [])
        ticker_name = ticker_info.get("name", "")

        matches: list[dict] = []
        matched_headlines: set[str] = set()

        for news_item in recent_news:
            text_lower = f"{news_item.headline} {news_item.summary}".lower()

            # Check relevance to this ticker
            if not self._is_ticker_relevant(
                text_lower, ticker, ticker_name, ticker_sector, ticker_themes
            ):
                continue

            # Find keyword matches
            for category, keywords in self.political_keywords.items():
                for keyword in keywords:
                    pattern = r"\b" + re.escape(keyword.lower()) + r"s?\b"
                    if re.search(pattern, text_lower):
                        matches.append(
                            {
                                "keyword": keyword,
                                "category": category,
                                "headline": news_item.headline,
                                "published_at": news_item.published_at.isoformat(),
                                "source": news_item.source,
                            }
                        )
                        if news_item.headline:
                            matched_headlines.add(news_item.headline)
                        break  # one match per keyword per article per category

        if not matches:
            return PoliticalSignal(
                ticker=ticker,
                date=as_of,
                score=0.0,
                breakdown={
                    "reason": "no_political_matches",
                    "articles_scanned": len(recent_news),
                },
            )

        score = self._score_matches(matches)

        return PoliticalSignal(
            ticker=ticker,
            date=as_of,
            score=round(score, 4),
            matched_keywords=list({m["keyword"] for m in matches}),
            headlines=list(matched_headlines)[:5],
            breakdown={
                "match_count": len(matches),
                "categories_triggered": list({m["category"] for m in matches}),
                "articles_scanned": len(recent_news),
                "lookback_days": self.lookback_days,
                "top_matches": matches[:8],
            },
        )

    def _is_ticker_relevant(
        self,
        text_lower: str,
        ticker: str,
        ticker_name: str,
        sector: str,
        themes: list[str],
    ) -> bool:
        """
        Determine if a news article is relevant to a specific ticker.

        Relevance checks (in priority order):
        1. Direct ticker symbol mention (with word boundary, min 2 chars)
        2. Company name mention
        3. Sector keyword match
        4. Theme keyword match

        # TODO(upgrade): Replace with LLM entity extraction for precision.
        """
        # 1. Ticker mention (skip 1-char tickers like "V" to avoid false positives)
        if len(ticker) >= 2:
            pattern = r"\b" + re.escape(ticker.lower()) + r"\b"
            if re.search(pattern, text_lower):
                return True

        # 2. Company name mention (use first two words of name for robustness)
        if ticker_name:
            name_words = ticker_name.lower().split()
            # Check first word if it's long enough (avoid "The", "Inc" etc)
            if len(name_words) >= 1 and len(name_words[0]) >= 4:
                if name_words[0] in text_lower:
                    return True
            # Check two-word combination
            if len(name_words) >= 2:
                two_word = f"{name_words[0]} {name_words[1]}"
                if two_word in text_lower:
                    return True

        # 3. Sector keyword match
        for kw in self.sector_keywords.get(sector, []):
            if len(kw) >= 3 and kw.lower() in text_lower:
                return True

        # 4. Theme keyword match
        for theme in themes:
            for kw in self.theme_keywords.get(theme, []):
                if len(kw) >= 3 and kw.lower() in text_lower:
                    return True

        return False

    def _score_matches(self, matches: list[dict]) -> float:
        """
        Compute aggregate score from keyword matches.

        Uses the category_sentiment mapping from config.
        Multiple matches are averaged.

        # TODO(upgrade): Replace with LLM confidence scoring per match.
        """
        if not matches:
            return 0.0

        # Deduplicate by category — don't over-weight one category
        category_scores: dict[str, float] = {}
        for match in matches:
            cat = match["category"]
            if cat not in category_scores:
                category_scores[cat] = self.category_sentiment.get(cat, 0.0)

        if not category_scores:
            return 0.0

        avg_score = sum(category_scores.values()) / len(category_scores)
        return max(-1.0, min(1.0, avg_score))
