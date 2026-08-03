"""
Weekend Swing Signals — three independent Monday-trigger strategies.

Each strategy follows the same shape: scan (weekend, cheap daily-bar calc)
-> Monday trigger (intraday bars) -> stop-loss -> Friday-close-or-stopped-out
exit (exit timing is handled by the backtest harness / live monitor, not
here — these classes only decide entry + stop).

Output is a WeekendSwingSignal, matching the (ticker, date, score, breakdown)
shape used by TechnicalSignal/SentimentSignal/PoliticalSignal/PatternSignal
elsewhere in signal_engine, so it's structurally compatible with
CompositeScorer.compute() — but this feature is NOT wired into the daily
scan/live-trigger flow. It's a separate, standalone lane (see
src/signal_engine/weekend_swing/README.md).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from signal_engine.weekend_swing.levels import (
    atr,
    avg_volume_by_time_slot,
    overnight_range_proxy,
    swing_pivot,
    prior_week_range,
)

logger = logging.getLogger(__name__)


@dataclass
class WeekendSwingSignal:
    """
    A single triggered weekend-swing signal.

    score: [-1, 1] — sign is direction (positive=long, negative=short),
    magnitude is conviction. Mirrors the other signal_engine dataclasses.
    """

    ticker: str
    date: object  # trigger timestamp (pd.Timestamp)
    strategy: str  # "prior_week_breakout" | "overnight_range_breakout" | "gap_strength"
    direction: str  # "long" | "short"
    score: float
    entry_price: float
    stop_price: float
    pivot_high: Optional[float] = None
    pivot_low: Optional[float] = None
    breakdown: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# STRATEGY 1 — Prior-Week Breakout Continuation
# ─────────────────────────────────────────────────────────────

class PriorWeekBreakoutStrategy:
    """
    Weekend: prior week's High/Low + ATR14 + 20-day avg volume by time slot.
    Monday: first confirmed 15-min candle closing beyond the range, with a
    volume filter, triggers entry.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("strategy_1_breakout", {})
        self.volume_filter_multiplier = cfg.get("volume_filter_multiplier", 1.2)
        self.atr_period = cfg.get("atr_period", 14)
        self.atr_stop_multiplier = cfg.get("atr_stop_multiplier", 0.3)
        self.exclude_earnings_week = cfg.get("exclude_earnings_week", True)

    def scan(
        self,
        ticker: str,
        daily_df: pd.DataFrame,
        week_start: pd.Timestamp,
        historical_15min_df: Optional[pd.DataFrame],
        earnings_excluded: set,
    ) -> Optional[dict]:
        """Weekend pass. Returns a levels dict ready for check_trigger(), or None."""
        if self.exclude_earnings_week and ticker.upper() in earnings_excluded:
            return None

        week_range = prior_week_range(daily_df, week_start)
        if week_range is None:
            return None
        week_high, week_low = week_range

        atr_series = atr(daily_df, period=self.atr_period)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return None
        atr_value = float(atr_series.iloc[-1])

        avg_volume_by_slot = {}
        if historical_15min_df is not None and not historical_15min_df.empty:
            avg_volume_by_slot = avg_volume_by_time_slot(historical_15min_df)

        return {
            "ticker": ticker,
            "strategy": "prior_week_breakout",
            "week_high": week_high,
            "week_low": week_low,
            "atr14": atr_value,
            "avg_volume_by_slot": avg_volume_by_slot,
        }

    def check_trigger(
        self,
        levels: dict,
        monday_15min_df: pd.DataFrame,
    ) -> Optional[WeekendSwingSignal]:
        """
        Monday pass. monday_15min_df must be sorted ascending, confirmed
        (closed) 15-min bars only, for Monday's regular session.
        Returns the first triggering signal, or None.
        """
        week_high = levels["week_high"]
        week_low = levels["week_low"]
        atr_value = levels["atr14"]
        avg_volume_by_slot = levels["avg_volume_by_slot"]

        for idx, row in monday_15min_df.iterrows():
            close = float(row["close"])
            volume = float(row["volume"])
            slot = idx.strftime("%H:%M")
            avg_vol = avg_volume_by_slot.get(slot)
            if not avg_vol or avg_vol <= 0:
                continue
            vol_ratio = volume / avg_vol
            if vol_ratio <= self.volume_filter_multiplier:
                continue

            conviction = round(min(1.0, 0.5 + 0.25 * (vol_ratio - self.volume_filter_multiplier)), 4)
            breakdown = {
                "trigger_time": slot, "trigger_close": close, "trigger_volume": volume,
                "avg_volume_for_slot": avg_vol, "volume_multiplier_actual": round(vol_ratio, 3),
                "atr14": atr_value, "week_high": week_high, "week_low": week_low,
            }

            if close > week_high:
                stop = min(float(row["low"]), week_high - self.atr_stop_multiplier * atr_value)
                return WeekendSwingSignal(
                    ticker=levels["ticker"], date=idx, strategy="prior_week_breakout",
                    direction="long", score=conviction, entry_price=close, stop_price=round(stop, 4),
                    pivot_high=week_high, pivot_low=week_low, breakdown=breakdown,
                )
            if close < week_low:
                stop = max(float(row["high"]), week_low + self.atr_stop_multiplier * atr_value)
                return WeekendSwingSignal(
                    ticker=levels["ticker"], date=idx, strategy="prior_week_breakout",
                    direction="short", score=-conviction, entry_price=close, stop_price=round(stop, 4),
                    pivot_high=week_high, pivot_low=week_low, breakdown=breakdown,
                )
        return None


# ─────────────────────────────────────────────────────────────
# STRATEGY 2 — Overnight Range Breakout
# ─────────────────────────────────────────────────────────────

class OvernightRangeBreakoutStrategy:
    """
    Weekend: proxy overnight range (Friday close -> Monday open — see
    levels.overnight_range_proxy for the data-quality caveat) + ATR14.
    Monday: first 5-min candle closing beyond the range, within the first
    N minutes of the regular session, triggers entry.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("strategy_2_overnight", {})
        self.atr_range_multiplier = cfg.get("atr_range_multiplier", 1.5)
        self.atr_period = cfg.get("atr_period", 14)
        self.trigger_window_minutes = cfg.get("trigger_window_minutes", 30)

    def scan(
        self,
        ticker: str,
        daily_df: pd.DataFrame,
        week_start: pd.Timestamp,
    ) -> Optional[dict]:
        proxy = overnight_range_proxy(daily_df, week_start)
        if proxy is None:
            return None

        atr_series = atr(daily_df, period=self.atr_period)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            return None
        atr_value = float(atr_series.iloc[-1])

        overnight_range = proxy["high"] - proxy["low"]
        skip_threshold = self.atr_range_multiplier * (atr_value / 24)
        if overnight_range > skip_threshold:
            logger.info(
                "%s skipped for overnight-range strategy: range %.2f > threshold %.2f (already moved too much)",
                ticker, overnight_range, skip_threshold,
            )
            return None

        return {
            "ticker": ticker,
            "strategy": "overnight_range_breakout",
            "overnight_high": proxy["high"],
            "overnight_low": proxy["low"],
            "atr14": atr_value,
            "data_quality": proxy["data_quality"],
        }

    def check_trigger(
        self,
        levels: dict,
        monday_5min_df: pd.DataFrame,
        session_open_time: pd.Timestamp,
    ) -> Optional[WeekendSwingSignal]:
        """
        monday_5min_df: Monday's 5-min bars, sorted ascending, confirmed only.
        session_open_time: regular-session open timestamp for that Monday
        (e.g. 9:30 ET) — the trigger window is only checked from here.
        """
        cutoff = session_open_time + pd.Timedelta(minutes=self.trigger_window_minutes)
        window_df = monday_5min_df.loc[
            (monday_5min_df.index >= session_open_time) & (monday_5min_df.index <= cutoff)
        ]

        high = levels["overnight_high"]
        low = levels["overnight_low"]

        for idx, row in window_df.iterrows():
            close = float(row["close"])
            breakdown = {
                "trigger_time": idx.strftime("%H:%M"), "overnight_high": high, "overnight_low": low,
                "data_quality": levels["data_quality"],
            }
            if close > high:
                return WeekendSwingSignal(
                    ticker=levels["ticker"], date=idx, strategy="overnight_range_breakout",
                    direction="long", score=0.7, entry_price=close, stop_price=low,
                    pivot_high=high, pivot_low=low, breakdown=breakdown,
                )
            if close < low:
                return WeekendSwingSignal(
                    ticker=levels["ticker"], date=idx, strategy="overnight_range_breakout",
                    direction="short", score=-0.7, entry_price=close, stop_price=high,
                    pivot_high=high, pivot_low=low, breakdown=breakdown,
                )
        return None


# ─────────────────────────────────────────────────────────────
# STRATEGY 3 — Gap Strength / Gap Fade
# ─────────────────────────────────────────────────────────────

class GapStrengthStrategy:
    """
    SPY/QQQ only (v1 scope, enforced here — not just via config).
    Weekend: prior swing high/low pivot (same calc as Strategy 1's range).
    Monday: two independent sub-signals from the open + first 5-min candle —
    gap-and-go (holds beyond pivot) or gap-and-fail (closes back through it).
    Results are never combined — see check_trigger()'s gap_type tag.
    """

    _HARD_RESTRICTED_TICKERS = {"SPY", "QQQ"}

    def __init__(self, config: dict) -> None:
        cfg = config.get("strategy_3_gap", {})
        configured = {t.upper() for t in cfg.get("restricted_tickers", ["SPY", "QQQ"])}
        # Code-enforced intersection with the hard v1 scope, per spec.
        self.allowed_tickers = (configured & self._HARD_RESTRICTED_TICKERS) or self._HARD_RESTRICTED_TICKERS

    def scan(
        self,
        ticker: str,
        daily_df: pd.DataFrame,
        week_start: pd.Timestamp,
    ) -> Optional[dict]:
        if ticker.upper() not in self.allowed_tickers:
            return None
        pivot = swing_pivot(daily_df, week_start)
        if pivot is None:
            return None
        pivot_high, pivot_low = pivot
        return {"ticker": ticker, "strategy": "gap_strength", "pivot_high": pivot_high, "pivot_low": pivot_low}

    def check_trigger(
        self,
        levels: dict,
        monday_open_price: float,
        first_5min_bar: pd.Series,
    ) -> Optional[WeekendSwingSignal]:
        """
        first_5min_bar: the first confirmed 5-min bar of Monday's regular
        session (its .name is the bar timestamp).

        Returns a single WeekendSwingSignal tagged gap_type="gap_and_go" or
        "gap_and_fail" in breakdown, or None if the open didn't gap beyond
        the pivot in either direction.
        """
        pivot_high = levels["pivot_high"]
        pivot_low = levels["pivot_low"]
        ticker = levels["ticker"]
        ts = first_5min_bar.name
        close = float(first_5min_bar["close"])

        if monday_open_price > pivot_high:
            if close > pivot_high:
                return WeekendSwingSignal(
                    ticker=ticker, date=ts, strategy="gap_strength", direction="long", score=0.75,
                    entry_price=close, stop_price=pivot_high, pivot_high=pivot_high, pivot_low=pivot_low,
                    breakdown={"gap_type": "gap_and_go", "monday_open": monday_open_price},
                )
            stop = float(first_5min_bar["high"])
            return WeekendSwingSignal(
                ticker=ticker, date=ts, strategy="gap_strength", direction="short", score=-0.65,
                entry_price=close, stop_price=stop, pivot_high=pivot_high, pivot_low=pivot_low,
                breakdown={"gap_type": "gap_and_fail", "monday_open": monday_open_price},
            )

        if monday_open_price < pivot_low:
            if close < pivot_low:
                return WeekendSwingSignal(
                    ticker=ticker, date=ts, strategy="gap_strength", direction="short", score=-0.75,
                    entry_price=close, stop_price=pivot_low, pivot_high=pivot_high, pivot_low=pivot_low,
                    breakdown={"gap_type": "gap_and_go", "monday_open": monday_open_price},
                )
            stop = float(first_5min_bar["low"])
            return WeekendSwingSignal(
                ticker=ticker, date=ts, strategy="gap_strength", direction="long", score=0.65,
                entry_price=close, stop_price=stop, pivot_high=pivot_high, pivot_low=pivot_low,
                breakdown={"gap_type": "gap_and_fail", "monday_open": monday_open_price},
            )

        return None  # open didn't gap beyond the pivot in either direction
