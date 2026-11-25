"""
Monday trigger check for one shortlisted (ticker, strategy) candidate.

Called by: weekend_swing/weekly_cycle.py. Depends on: the 3 strategy classes
and IntradayBarStore.

WHY a third copy of this logic: cli.py `weekend-monitor` and the dashboard's
/triggered/check endpoint each inline the same per-strategy bar plumbing.
Those are left untouched so they keep behaving exactly as validated; fold them
onto this function when convenient.
"""

import logging
from typing import Optional

import pandas as pd

from signal_engine.weekend_swing.levels import monday_session_bars, session_open_time

logger = logging.getLogger(__name__)


def check_candidate_trigger(
    candidate: dict,
    strategies: dict,
    bar_store,
    week_start: pd.Timestamp,
    earnings_excluded: set,
    now: Optional[pd.Timestamp] = None,
):
    """Returns a WeekendSwingSignal if the candidate has triggered by `now`, else None."""
    now = now or pd.Timestamp.now(tz="UTC")
    ticker, name = candidate["ticker"], candidate["strategy"]
    strategy = strategies[name]

    if name == "prior_week_breakout":
        daily = bar_store.get_bars(ticker, "1Day", week_start - pd.Timedelta(days=90), week_start)
        hist_15 = bar_store.get_bars(ticker, "15Min", week_start - pd.Timedelta(days=40), week_start)
        levels = strategy.scan(ticker, daily, week_start, hist_15, earnings_excluded)
        if levels is None:
            return None
        bars = bar_store.get_bars(ticker, "15Min", week_start, now)
        return strategy.check_trigger(levels, monday_session_bars(bars, week_start))

    daily = bar_store.get_bars(ticker, "1Day", week_start - pd.Timedelta(days=60), week_start)
    levels = strategy.scan(ticker, daily, week_start)
    if levels is None:
        return None
    bars = bar_store.get_bars(ticker, "5Min", week_start, now)
    bars = monday_session_bars(bars, week_start)
    if bars.empty:
        return None

    if name == "overnight_range_breakout":
        return strategy.check_trigger(levels, bars, session_open_time(week_start, bars.index))

    # gap_strength: needs Monday's official open from the daily bar
    monday_daily = daily.loc[daily.index >= week_start]
    if monday_daily.empty:
        # WHY: the daily bar for "today" may not exist yet; the first 5-min bar's open is the session open.
        monday_open = float(bars.iloc[0]["open"])
    else:
        monday_open = float(monday_daily.iloc[0]["open"])
    return strategy.check_trigger(levels, monday_open, bars.iloc[0])
