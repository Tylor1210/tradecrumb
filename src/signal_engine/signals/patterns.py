"""
Candlestick and chart (trendline) pattern detection.

Scans the most recent bars of an OHLC DataFrame for a fixed set of
well-known reversal patterns and scores them on the same [-1, 1] scale
used by the other signal modules. This is a small, inspectable rule set
(not a black-box classifier) — every match names the exact pattern, its
direction, and which bar it matched on.

Candlestick patterns (last few bars):
    Bullish Engulfing, Bearish Engulfing, Hammer, Shooting Star,
    Morning Star, Evening Star

Chart / trendline patterns (lookback window):
    Double Top, Double Bottom

Scores:
   1.0 = strongly bullish (favor a paper BUY / long)
  -1.0 = strongly bearish (favor a paper SHORT)
   0.0 = no pattern detected
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CANDLE_STRENGTH = {
    "Bullish Engulfing": 0.65,
    "Bearish Engulfing": 0.65,
    "Hammer": 0.55,
    "Shooting Star": 0.55,
    "Morning Star": 0.80,
    "Evening Star": 0.80,
}
_CHART_STRENGTH = {
    "Double Top": 0.70,
    "Double Bottom": 0.70,
}


@dataclass
class PatternSignal:
    """Pattern-detection result for a single ticker on a single date."""

    ticker: str
    date: object
    score: float  # [-1, 1]
    matched_patterns: list = field(default_factory=list)  # [{name, direction, bar_offset}]
    breakdown: dict = field(default_factory=dict)


class PatternDetector:
    """
    Detects candlestick reversal patterns near the latest bar and simple
    double top/bottom chart patterns over a lookback window.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.lookback = config.get("chart_pattern_lookback", 40)
        self.peak_tolerance_pct = config.get("peak_tolerance_pct", 1.5)
        self.recency_window = config.get("candle_recency_window", 3)

    def compute(self, ticker: str, df: pd.DataFrame) -> Optional[PatternSignal]:
        if df is None or len(df) < 10:
            return None

        o = df["open"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        l = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        n = len(c)
        date = df.index[-1]

        matches: list = []
        for i in range(max(2, n - self.recency_window), n):
            m = self._match_candle(o, h, l, c, i)
            if m:
                matches.append(m)

        chart_match = self._match_double_top_bottom(h, l, c)
        if chart_match:
            matches.append(chart_match)

        if not matches:
            return PatternSignal(ticker=ticker, date=date, score=0.0, matched_patterns=[], breakdown={})

        scored = []
        for m in matches:
            bars_back = (n - 1) - m["bar_index"]
            recency_weight = max(0.4, 1.0 - 0.15 * bars_back)
            direction_sign = 1 if m["direction"] == "bullish" else -1
            scored.append(m["strength"] * recency_weight * direction_sign)

        score = float(max(-1.0, min(1.0, sum(scored))))
        matched_patterns = [
            {"name": m["name"], "direction": m["direction"], "bar_offset": m["bar_index"] - (n - 1)}
            for m in matches
        ]

        return PatternSignal(
            ticker=ticker,
            date=date,
            score=round(score, 4),
            matched_patterns=matched_patterns,
            breakdown={"matches": matched_patterns},
        )

    # ── Candlestick pattern rules ────────────────────────────────────────

    def _match_candle(self, o, h, l, c, i) -> Optional[dict]:
        if i >= 2:
            m = self._morning_evening_star(o, h, l, c, i)
            if m:
                return m
        if i >= 1:
            m = self._engulfing(o, h, l, c, i)
            if m:
                return m
        if i >= 3:
            m = self._hammer_shooting_star(o, h, l, c, i)
            if m:
                return m
        return None

    @staticmethod
    def _body(o, c, i) -> float:
        return abs(c[i] - o[i])

    @staticmethod
    def _range(h, l, i) -> float:
        return max(h[i] - l[i], 1e-9)

    def _engulfing(self, o, h, l, c, i) -> Optional[dict]:
        prev_low, prev_high = min(o[i - 1], c[i - 1]), max(o[i - 1], c[i - 1])
        cur_low, cur_high = min(o[i], c[i]), max(o[i], c[i])
        engulfs = cur_low <= prev_low and cur_high >= prev_high
        if not engulfs:
            return None

        prev_bearish = c[i - 1] < o[i - 1]
        prev_bullish = c[i - 1] > o[i - 1]
        cur_bullish = c[i] > o[i]
        cur_bearish = c[i] < o[i]

        if prev_bearish and cur_bullish:
            return {"name": "Bullish Engulfing", "direction": "bullish", "strength": _CANDLE_STRENGTH["Bullish Engulfing"], "bar_index": i}
        if prev_bullish and cur_bearish:
            return {"name": "Bearish Engulfing", "direction": "bearish", "strength": _CANDLE_STRENGTH["Bearish Engulfing"], "bar_index": i}
        return None

    def _hammer_shooting_star(self, o, h, l, c, i) -> Optional[dict]:
        body = self._body(o, c, i)
        rng = self._range(h, l, i)
        if body / rng > 0.35:
            return None

        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        prior_trend = c[i - 1] - c[i - 3]
        min_wick = max(body, rng * 0.05)

        if lower_wick >= 2 * min_wick and upper_wick <= body * 0.5 and prior_trend < 0:
            return {"name": "Hammer", "direction": "bullish", "strength": _CANDLE_STRENGTH["Hammer"], "bar_index": i}
        if upper_wick >= 2 * min_wick and lower_wick <= body * 0.5 and prior_trend > 0:
            return {"name": "Shooting Star", "direction": "bearish", "strength": _CANDLE_STRENGTH["Shooting Star"], "bar_index": i}
        return None

    def _morning_evening_star(self, o, h, l, c, i) -> Optional[dict]:
        first_bearish = c[i - 2] < o[i - 2]
        first_bullish = c[i - 2] > o[i - 2]
        first_body = self._body(o, c, i - 2)
        mid_body = self._body(o, c, i - 1)
        small_mid = mid_body < first_body * 0.5 if first_body > 0 else mid_body < self._range(h, l, i - 1) * 0.3

        if first_bearish and small_mid and c[i] > o[i] and c[i] > (o[i - 2] + c[i - 2]) / 2:
            return {"name": "Morning Star", "direction": "bullish", "strength": _CANDLE_STRENGTH["Morning Star"], "bar_index": i}
        if first_bullish and small_mid and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) / 2:
            return {"name": "Evening Star", "direction": "bearish", "strength": _CANDLE_STRENGTH["Evening Star"], "bar_index": i}
        return None

    # ── Chart pattern: double top / double bottom ──────────────────────────

    def _match_double_top_bottom(self, h, l, c) -> Optional[dict]:
        n = len(c)
        window = min(self.lookback, n)
        if window < 15:
            return None
        hi = h[-window:]
        lo = l[-window:]
        cl = c[-window:]

        peaks = self._local_extrema(hi, is_max=True)
        troughs = self._local_extrema(lo, is_max=False)

        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if p2 > p1:
                v1, v2 = hi[p1], hi[p2]
                if abs(v1 - v2) / max(v1, v2) <= self.peak_tolerance_pct / 100:
                    between_low = lo[p1:p2 + 1].min()
                    if cl[-1] < between_low:
                        return {
                            "name": "Double Top", "direction": "bearish",
                            "strength": _CHART_STRENGTH["Double Top"], "bar_index": n - window + p2,
                        }

        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if t2 > t1:
                v1, v2 = lo[t1], lo[t2]
                if abs(v1 - v2) / max(v1, v2) <= self.peak_tolerance_pct / 100:
                    between_high = hi[t1:t2 + 1].max()
                    if cl[-1] > between_high:
                        return {
                            "name": "Double Bottom", "direction": "bullish",
                            "strength": _CHART_STRENGTH["Double Bottom"], "bar_index": n - window + t2,
                        }
        return None

    @staticmethod
    def _local_extrema(arr: np.ndarray, is_max: bool, order: int = 3) -> list:
        """Indices (within `arr`) of local maxima/minima using a rolling window."""
        idxs = []
        for i in range(order, len(arr) - order):
            window = arr[i - order:i + order + 1]
            if is_max and arr[i] == window.max():
                idxs.append(i)
            elif not is_max and arr[i] == window.min():
                idxs.append(i)
        return idxs
