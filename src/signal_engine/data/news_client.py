"""
News data clients — Finnhub (primary) and NewsAPI (political sweep).

NewsItem is the normalized data model used across the signal engine.

Finnhub free tier:
  - 60 API calls/minute
  - Company news: /company-news?symbol={ticker}&from={date}&to={date}
  - General news: /news?category=general
  - Pre-computed sentiment scores on most articles

NewsAPI free tier:
  - 100 requests/day (developer: 1000/day)
  - Best for geopolitical/political keyword sweeps
  - Headlines only (full content on paid tier)
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """
    Normalized news article used across all signal modules.
    Both Finnhub and NewsAPI articles are normalized to this schema.
    """

    ticker: Optional[str]       # None for general/political news
    headline: str
    summary: str
    published_at: datetime
    source: str
    url: str
    raw_sentiment_score: float = 0.0  # Finnhub pre-computed; 0.0 when unavailable
    category: str = "general"          # "general", "company", "political"


# ─────────────────────────────────────────────────────────────
# FINNHUB CLIENT
# ─────────────────────────────────────────────────────────────

class FinnhubNewsClient:
    """
    Finnhub API adapter.

    Provides:
      - get_company_news()     — ticker-specific news with sentiment
      - get_market_news()      — general market news
      - get_basic_financials() — EPS/revenue data for LEAPS screening
    """

    def __init__(self) -> None:
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "FINNHUB_API_KEY must be set in .env\n"
                "Get a free key at: https://finnhub.io/"
            )
        import finnhub  # noqa: PLC0415
        self._client = finnhub.Client(api_key=api_key)

    def get_company_news(
        self,
        ticker: str,
        from_date: str,  # "YYYY-MM-DD"
        to_date: str,    # "YYYY-MM-DD"
    ) -> list[NewsItem]:
        """
        Fetch company-specific news for a ticker.
        Returns up to ~50 articles per call (Finnhub limit).
        """
        try:
            raw = self._client.company_news(ticker, _from=from_date, to=to_date)
        except Exception as exc:
            logger.warning("Finnhub company news failed for %s: %s", ticker, exc)
            return []

        items = []
        for item in raw or []:
            try:
                published_at = datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc)
            except (OSError, OverflowError):
                published_at = datetime.now(timezone.utc)

            # Finnhub sentiment: in news response as "sentiment" field
            # Values are typically floats or absent
            raw_sent = item.get("sentiment", 0.0)
            if isinstance(raw_sent, dict):
                # Sometimes nested: {"bullishPercent": ..., "bearishPercent": ...}
                bull = raw_sent.get("bullishPercent", 0.5)
                bear = raw_sent.get("bearishPercent", 0.5)
                raw_sent = (bull - bear)  # normalize to [-1, 1] range approx
            raw_sent = float(raw_sent) if raw_sent else 0.0

            items.append(
                NewsItem(
                    ticker=ticker,
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    published_at=published_at,
                    source=item.get("source", "finnhub"),
                    url=item.get("url", ""),
                    raw_sentiment_score=raw_sent,
                    category="company",
                )
            )

        logger.debug("Finnhub: %d articles for %s", len(items), ticker)
        return items

    def get_market_news(self, category: str = "general") -> list[NewsItem]:
        """
        Fetch general market news (not ticker-specific).
        Categories: 'general', 'forex', 'crypto', 'merger'
        """
        try:
            raw = self._client.general_news(category, min_id=0)
        except Exception as exc:
            logger.warning("Finnhub general news failed: %s", exc)
            return []

        items = []
        for item in raw or []:
            try:
                published_at = datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc)
            except (OSError, OverflowError):
                published_at = datetime.now(timezone.utc)

            items.append(
                NewsItem(
                    ticker=None,
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    published_at=published_at,
                    source=item.get("source", "finnhub"),
                    url=item.get("url", ""),
                    raw_sentiment_score=0.0,
                    category=category,
                )
            )

        return items

    def get_basic_financials(self, ticker: str, metric: str = "all") -> dict:
        """
        Fetch basic financial metrics for a ticker.
        Used by the LEAPS screener for earnings trend analysis.

        Returns raw Finnhub basicFinancials dict:
          data["series"]["quarterly"]["eps"] → list of {period, v} dicts
        """
        try:
            return self._client.company_basic_financials(ticker, metric)
        except Exception as exc:
            logger.warning("Finnhub basic financials failed for %s: %s", ticker, exc)
            return {}

    def get_earnings_calendar(
        self,
        from_date: str,  # "YYYY-MM-DD"
        to_date: str,    # "YYYY-MM-DD"
        symbol: str = "",
    ) -> list[dict]:
        """
        Fetch upcoming earnings report dates for a date range.
        Used by weekend_swing.earnings for the earnings-week exclusion filter.

        Args:
            symbol: single ticker, or "" for all upcoming earnings in range.

        Returns raw Finnhub calendar entries (each has "symbol", "date", ...).
        """
        try:
            result = self._client.earnings_calendar(_from=from_date, to=to_date, symbol=symbol)
            return result.get("earningsCalendar", [])
        except Exception as exc:
            logger.warning("Finnhub earnings calendar failed (%s -> %s): %s", from_date, to_date, exc)
            return []


# ─────────────────────────────────────────────────────────────
# NEWSAPI CLIENT
# ─────────────────────────────────────────────────────────────

class NewsAPIClient:
    """
    NewsAPI adapter — used for the political/geopolitical keyword sweep.

    Best for:
      - Searching by keyword across all sources (not ticker-specific)
      - Trump mentions, tariff news, geopolitical events
      - Headlines from major outlets

    Free tier: 100 req/day. Use sparingly — cache results.
    """

    def __init__(self) -> None:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            raise EnvironmentError(
                "NEWSAPI_KEY must be set in .env\n"
                "Get a free key at: https://newsapi.org/register"
            )
        from newsapi import NewsApiClient as _NewsApiClient  # noqa: PLC0415
        self._client = _NewsApiClient(api_key=api_key)

    def search_political_news(
        self,
        keywords: list[str],
        from_date: str,   # "YYYY-MM-DD"
        to_date: str,     # "YYYY-MM-DD"
        language: str = "en",
        max_results: int = 100,
    ) -> list[NewsItem]:
        """
        Search for political/geopolitical news across all sources.

        Args:
            keywords:    List of search terms (ORed together)
            from_date:   Start date string "YYYY-MM-DD"
            to_date:     End date string "YYYY-MM-DD"
            language:    Article language (default: "en")
            max_results: Cap on articles returned

        Returns:
            List of NewsItem (ticker=None, category="political")
        """
        if not keywords:
            return []

        # NewsAPI supports OR queries with "a OR b" syntax
        # Cap to 5 keywords to avoid overly broad queries that eat rate limit
        kw_sample = keywords[:5]
        query = " OR ".join(f'"{kw}"' for kw in kw_sample)

        try:
            raw = self._client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by="relevancy",
                page_size=min(max_results, 100),
            )
        except Exception as exc:
            logger.warning("NewsAPI search failed (keywords=%s): %s", kw_sample, exc)
            return []

        articles = raw.get("articles", []) or []
        items = []

        for article in articles[:max_results]:
            try:
                raw_dt = article.get("publishedAt", "")
                if raw_dt:
                    published_at = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now()
            except ValueError:
                published_at = datetime.now(timezone.utc)

            source_name = ""
            src = article.get("source")
            if isinstance(src, dict):
                source_name = src.get("name", "")
            elif isinstance(src, str):
                source_name = src

            items.append(
                NewsItem(
                    ticker=None,
                    headline=article.get("title", ""),
                    summary=article.get("description", "") or "",
                    published_at=published_at,
                    source=source_name,
                    url=article.get("url", ""),
                    raw_sentiment_score=0.0,
                    category="political",
                )
            )

        logger.info("NewsAPI: %d articles for keywords %s", len(items), kw_sample)
        return items
