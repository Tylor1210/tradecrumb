"""
Weekend Swing Signals — pivot/range/ATR math.

Pure functions only (no I/O, no API calls) so they're trivial to unit test
against synthetic OHLC data. All functions take an explicit `week_start`
(the Monday being evaluated) rather than reading the clock, so backtests and
live scans share exactly the same code path.
"""

from typing import Optional

import pandas as pd

ET = "America/New_York"
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
SESSION_CLOSE = pd.Timedelta(hours=16)


# ── Time alignment ────────────────────────────────────────────
# Alpaca bars are UTC-indexed. A daily bar is stamped at midnight New York time
# (05:00 UTC in winter, 04:00 UTC in summer), and 5/15-min bars include IEX
# pre-market and after-hours prints. week_start is a calendar Monday (UTC
# midnight from the CLI). So bars are compared by their New York calendar date
# and wall-clock time, never by raw UTC timestamp. Naive indexes (test
# fixtures) are taken as already being New York wall-clock time.

def _ny_wall_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    return index.tz_convert(ET).tz_localize(None) if index.tz is not None else index


def bar_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """New York calendar date of each bar (naive, midnight)."""
    return _ny_wall_clock(df.index).normalize()


def calendar_day(ts: pd.Timestamp) -> pd.Timestamp:
    """The calendar date `ts` names (week_start is 'Monday', whatever tz it carries)."""
    return pd.Timestamp(pd.Timestamp(ts).date())


def session_open_time(day: pd.Timestamp, like_index: pd.DatetimeIndex) -> pd.Timestamp:
    """09:30 New York on `day`, expressed in the same tz as `like_index`."""
    naive = calendar_day(day) + SESSION_OPEN
    tz = pd.DatetimeIndex(like_index).tz
    return naive.tz_localize(ET).tz_convert(tz) if tz is not None else naive


def monday_session_bars(intraday_df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Intraday bars from `day`'s regular session only (09:30 to 16:00 New York)."""
    if intraday_df is None or intraday_df.empty:
        return intraday_df
    wall = _ny_wall_clock(intraday_df.index)
    since_midnight = wall - wall.normalize()
    mask = (
        (wall.normalize() == calendar_day(day))
        & (since_midnight >= SESSION_OPEN)
        & (since_midnight < SESSION_CLOSE)
    )
    return intraday_df.loc[mask]


def is_data_sane(daily_df: pd.DataFrame, max_single_day_move_pct: float = 50.0) -> bool:
    """
    Rejects tickers whose daily bars contain a single-day close-to-close move
    bigger than `max_single_day_move_pct` — a cheap guard against
    unadjusted-split or otherwise corrupted data from the volume screener
    (real backtest example: LBGJ's price swung $92 -> ~$12,800 avg -> $270
    within the same 12 months; one trade on it alone returned -194%).

    Not a liquidity/quality filter — just a "is this data even usable" check.
    """
    if daily_df is None or len(daily_df) < 2:
        return True  # not enough data to judge; let the normal empty-data checks handle it
    close = daily_df["close"].astype(float)
    daily_moves_pct = close.pct_change().abs() * 100
    return bool(daily_moves_pct.max() <= max_single_day_move_pct)


def prior_week_range(daily_df: pd.DataFrame, week_start: pd.Timestamp) -> Optional[tuple[float, float]]:
    """
    Prior week's High/Low (Mon open -> Fri close) for the week immediately
    before `week_start`. Returns (high, low), or None if no bars fall in
    that window.
    """
    monday = calendar_day(week_start)
    dates = bar_dates(daily_df)
    week_df = daily_df.loc[(dates >= monday - pd.Timedelta(days=7)) & (dates < monday)]
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
    monday = calendar_day(week_start)
    dates = bar_dates(daily_df)
    # WHY dates, not timestamps: Alpaca stamps Friday's bar at Fri 05:00 UTC, which is
    # after "Friday 00:00", so a timestamp cut-off silently picked Thursday's close.
    friday_rows = daily_df.loc[dates < monday]
    monday_rows = daily_df.loc[dates >= monday]
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
