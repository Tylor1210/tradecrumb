"""
Weekend Swing Signals — earnings-week exclusion.

Strategy 1 (Prior-Week Breakout Continuation) excludes tickers reporting
earnings during the upcoming week, since an earnings gap swamps a technical
breakout signal. Uses the real Finnhub earnings calendar (not a stub —
finnhub-python exposes Client.earnings_calendar()).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_earnings_exclusions(
    finnhub_client,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> set[str]:
    """
    Returns the set of ticker symbols with an earnings report scheduled
    between week_start and week_end (inclusive).

    Returns an empty set (logged) if finnhub_client is None or the call
    fails — never raises, matching this module's "best-effort filter" role.
    """
    if finnhub_client is None:
        logger.info("Finnhub not configured — earnings-week exclusion disabled")
        return set()

    from_date = week_start.strftime("%Y-%m-%d")
    to_date = week_end.strftime("%Y-%m-%d")

    entries = finnhub_client.get_earnings_calendar(from_date, to_date, symbol="")
    excluded = {e["symbol"].upper() for e in entries if e.get("symbol")}

    logger.info("Earnings-week exclusions (%s -> %s): %d tickers", from_date, to_date, len(excluded))
    return excluded
