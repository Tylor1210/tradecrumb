"""
Weekend Swing Signals — the cheap (daily-bar-only) weekend scan.

Shared by both CLI entry points: `weekend-scan` (Sat/Sun report) and
`weekend-monitor` (Monday — rebuilds the same shortlist before spending the
expensive intraday-data budget only on tickers that passed this filter).
Re-running this cheap pass on Monday morning is intentional — it's a single
daily-bar fetch per ticker, not the expensive part — rather than requiring a
persisted scan-result store (Part 6, deferred).
"""

import logging

import pandas as pd

from signal_engine.weekend_swing.earnings import get_earnings_exclusions
from signal_engine.weekend_swing.weekend_swing_signals import (
    GapStrengthStrategy,
    OvernightRangeBreakoutStrategy,
    PriorWeekBreakoutStrategy,
)

logger = logging.getLogger(__name__)


def next_or_current_monday(as_of: pd.Timestamp) -> pd.Timestamp:
    """Returns `as_of`'s Monday if it's already a Monday, otherwise the upcoming one."""
    as_of = as_of.normalize()
    offset = (0 - as_of.weekday()) % 7
    return as_of + pd.Timedelta(days=offset)


def build_weekend_watchlist(
    ws_config: dict,
    alpaca_client,
    finnhub_client,
    bar_store,
    week_start: pd.Timestamp,
) -> dict:
    """
    Cheap, daily-bar-only scan across the full universe for all 3 strategies.

    Returns:
        {
          "week_start", "week_end", "earnings_excluded": set,
          "candidates": [ {ticker, strategy, pivot_high, pivot_low, current_price,
                            distance_pct, earnings_excluded, data_quality}, ... ]
                        sorted by distance_pct ascending,
          "skipped": [(ticker, reason), ...],
        }
    """
    from signal_engine.weekend_swing.universe import WeekendUniverseBuilder

    week_end = week_start + pd.Timedelta(days=4)
    universe = WeekendUniverseBuilder(ws_config).build(alpaca_client)
    earnings_excluded = get_earnings_exclusions(finnhub_client, week_start, week_end)

    proximity_pct = ws_config.get("universe", {}).get("pivot_proximity_pct", 2.0)

    s1 = PriorWeekBreakoutStrategy(ws_config)
    s2 = OvernightRangeBreakoutStrategy(ws_config)
    s3 = GapStrengthStrategy(ws_config)

    candidates = []
    skipped = []

    for ticker in universe:
        try:
            daily_df = bar_store.get_bars(
                ticker, "1Day", week_start - pd.Timedelta(days=90), week_start - pd.Timedelta(days=1)
            )
        except Exception as exc:
            skipped.append((ticker, f"data fetch failed: {exc}"))
            continue

        if daily_df.empty:
            skipped.append((ticker, "no daily data available"))
            continue

        current_price = float(daily_df.iloc[-1]["close"])

        strategy_levels = [
            ("prior_week_breakout", s1.scan(ticker, daily_df, week_start, None, earnings_excluded), ("week_high", "week_low")),
            ("overnight_range_breakout", s2.scan(ticker, daily_df, week_start), ("overnight_high", "overnight_low")),
            ("gap_strength", s3.scan(ticker, daily_df, week_start), ("pivot_high", "pivot_low")),
        ]

        for strategy_name, levels, (hi_key, lo_key) in strategy_levels:
            if levels is None:
                continue
            pivot_high, pivot_low = levels[hi_key], levels[lo_key]
            distance_pct = min(
                abs(current_price - pivot_high) / current_price * 100,
                abs(current_price - pivot_low) / current_price * 100,
            )
            if distance_pct > proximity_pct:
                continue

            candidates.append({
                "ticker": ticker,
                "strategy": strategy_name,
                "pivot_high": pivot_high,
                "pivot_low": pivot_low,
                "current_price": current_price,
                "distance_pct": round(distance_pct, 3),
                "earnings_excluded": ticker.upper() in earnings_excluded,
                "data_quality": levels.get("data_quality", ""),
            })

    candidates.sort(key=lambda c: c["distance_pct"])

    return {
        "week_start": week_start,
        "week_end": week_end,
        "universe_size": len(universe),
        "earnings_excluded": earnings_excluded,
        "candidates": candidates,
        "skipped": skipped,
    }
