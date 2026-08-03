"""
Weekend Swing Signals — pivot/range/ATR math.

Pure functions only (no I/O, no API calls) so they're trivial to unit test
against synthetic OHLC data. All functions take an explicit `week_start`
(the Monday being evaluated) rather than reading the clock, so backtests and
live scans share exactly the same code path.
"""

from typing import Optional

import pandas as pd


def prior_week_range(daily_df: pd.DataFrame, week_start: pd.Timestamp) -> Optional[tuple[float, float]]:
    """
    Prior week's High/Low (Mon open -> Fri close) for the week immediately
    before `week_start`. Returns (high, low), or None if no bars fall in
    that window.
    """
    prior_week_end = week_start - pd.Timedelta(days=1)
    prior_week_start = prior_week_end - pd.Timedelta(days=6)
    week_df = daily_df.loc[(daily_df.index >= prior_week_start) & (daily_df.index <= prior_week_end)]
    if week_df.empty:
        return None
    return float(week_df["high"].max()), float(week_df["low"].min())


def swing_pivot(daily_df: pd.DataFrame, week_start: pd.Timestamp) -> Optional[tuple[float, float]]:
    """
    Prior swing high/low pivot, used by Strategy 3 (Gap Strength/Fade).
    Same calculation as prior_week_range — kept as a separate name because
    the strategies refer to it as a distinct concept in the spec.
    """
    return prior_week_range(daily_df, week_start)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's Average True Range.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder-smoothed (EWM with alpha=1/period) moving average of TR.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def avg_volume_by_time_slot(intraday_df: pd.DataFrame, lookback_days: int = 20) -> dict:
    """
    Groups intraday bars by time-of-day (HH:MM) and computes the trailing
    N-trading-day average volume for each slot. Used by Strategy 1's volume
    filter ("trigger candle volume > 1.2x the 20-day average volume for
    that same time-of-day slot").
    """
    if intraday_df is None or intraday_df.empty:
        return {}

    df = intraday_df.copy()
    df["_date"] = df.index.date
    recent_dates = sorted(df["_date"].unique())[-lookback_days:]
    df = df[df["_date"].isin(recent_dates)]
    df["_slot"] = df.index.strftime("%H:%M")

    return df.groupby("_slot")["volume"].mean().to_dict()


def overnight_range_proxy(daily_df: pd.DataFrame, week_start: pd.Timestamp) -> Optional[dict]:
    """
    Proxy "overnight range" using only regular-session data we actually have:
    Friday's close through Monday's open.

    CAVEAT: this is NOT a true extended-hours/overnight session — Alpaca's
    free IEX feed doesn't reliably carry Sunday-night prints. Every result
    is tagged with data_quality="regular_session_proxy_no_extended_hours" so
    downstream signals can flag it (see config/weekend_swing.yaml).
    """
    prior_friday = week_start - pd.Timedelta(days=3)
    friday_rows = daily_df.loc[daily_df.index <= prior_friday]
    monday_rows = daily_df.loc[daily_df.index >= week_start]
    if friday_rows.empty or monday_rows.empty:
        return None

    friday_close = float(friday_rows.iloc[-1]["close"])
    monday_open = float(monday_rows.iloc[0]["open"])

    return {
        "high": max(friday_close, monday_open),
        "low": min(friday_close, monday_open),
        "friday_close": friday_close,
        "monday_open": monday_open,
        "data_quality": "regular_session_proxy_no_extended_hours",
    }
