"""
Technical signal computation.

Computes moving averages, RSI, MACD, relative volume, and support/resistance
breakouts from OHLCV price data. All signals are combined into a single
composite technical score in [-1, 1] with a full, inspectable breakdown dict.

Scores:
   1.0 = strongly bullish
  -1.0 = strongly bearish
   0.0 = neutral

All component weights are configurable via config/signals.yaml.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """
    Technical analysis signal for a single ticker on a single date.

    breakdown is a dict of human-readable component details — the "why".
    """

    ticker: str
    date: pd.Timestamp
    score: float          # composite score [-1, 1]

    # Individual component scores
    ma_score: float = 0.0
    rsi_score: float = 0.0
    macd_score: float = 0.0
    volume_score: float = 0.0
    breakout_score: float = 0.0

    breakdown: dict = field(default_factory=dict)


class TechnicalAnalyzer:
    """
    Computes technical indicators and returns a TechnicalSignal.

    Requires at least sma_slow bars of data (default: 200 days).
    Returns None if insufficient data.
    """

    def __init__(self, config: dict) -> None:
        self.sma_fast = config.get("sma_fast", 20)
        self.sma_mid = config.get("sma_mid", 50)
        self.sma_slow = config.get("sma_slow", 200)
        self.rsi_period = config.get("rsi_period", 14)
        self.volume_window = config.get("volume_window", 20)
        self.breakout_window = config.get("breakout_window", 20)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)

        weights = config.get("component_weights", {})
        self._w_ma = weights.get("ma", 0.30)
        self._w_rsi = weights.get("rsi", 0.20)
        self._w_macd = weights.get("macd", 0.25)
        self._w_vol = weights.get("volume", 0.10)
        self._w_breakout = weights.get("breakout", 0.15)

    def compute(self, ticker: str, df: pd.DataFrame) -> Optional[TechnicalSignal]:
        """
        Compute all technical signals from an OHLCV DataFrame.

        Returns TechnicalSignal or None if data is insufficient.
        """
        required_rows = self.sma_slow + 10
        if df is None or len(df) < required_rows:
            logger.debug(
                "%s: insufficient data (%d rows, need %d)",
                ticker, len(df) if df is not None else 0, required_rows,
            )
            return None

        df = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # ── Moving Averages ─────────────────────────────────────────────
        sma_fast = close.rolling(self.sma_fast).mean()
        sma_mid = close.rolling(self.sma_mid).mean()
        sma_slow = close.rolling(self.sma_slow).mean()

        current_close = float(close.iloc[-1])
        ma_score, ma_breakdown = self._score_moving_averages(
            current_close,
            float(sma_fast.iloc[-1]),
            float(sma_mid.iloc[-1]),
            float(sma_slow.iloc[-1]),
        )

        # ── RSI ──────────────────────────────────────────────────────────
        rsi_series = self._compute_rsi(close, self.rsi_period)
        rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        rsi_score, rsi_breakdown = self._score_rsi(rsi_val)

        # ── MACD ──────────────────────────────────────────────────────────
        macd_line, signal_line, histogram = self._compute_macd(close)
        macd_score, macd_breakdown = self._score_macd(
            float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
        )

        # ── Relative Volume ───────────────────────────────────────────────
        vol_avg = volume.rolling(self.volume_window).mean()
        rel_vol = float(volume.iloc[-1]) / float(vol_avg.iloc[-1]) if float(vol_avg.iloc[-1]) > 0 else 1.0
        price_change = (current_close - float(close.iloc[-2])) / float(close.iloc[-2]) if len(close) >= 2 else 0.0
        vol_score, vol_breakdown = self._score_volume(rel_vol, price_change)

        # ── Support / Resistance Breakout ─────────────────────────────────
        # Use previous window (exclude today) to define the level
        resistance = float(high.iloc[-(self.breakout_window + 1):-1].max())
        support = float(low.iloc[-(self.breakout_window + 1):-1].min())
        breakout_score, breakout_breakdown = self._score_breakout(
            current_close, resistance, support
        )

        # ── Composite ─────────────────────────────────────────────────────
        composite = (
            self._w_ma * ma_score
            + self._w_rsi * rsi_score
            + self._w_macd * macd_score
            + self._w_vol * vol_score
            + self._w_breakout * breakout_score
        )
        composite = float(max(-1.0, min(1.0, composite)))

        return TechnicalSignal(
            ticker=ticker,
            date=df.index[-1],
            score=round(composite, 4),
            ma_score=round(ma_score, 4),
            rsi_score=round(rsi_score, 4),
            macd_score=round(macd_score, 4),
            volume_score=round(vol_score, 4),
            breakout_score=round(breakout_score, 4),
            breakdown={
                "moving_averages": ma_breakdown,
                "rsi": rsi_breakdown,
                "macd": macd_breakdown,
                "volume": vol_breakdown,
                "breakout": breakout_breakdown,
                "component_weights": {
                    "ma": self._w_ma,
                    "rsi": self._w_rsi,
                    "macd": self._w_macd,
                    "volume": self._w_vol,
                    "breakout": self._w_breakout,
                },
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers — each returns (score: float, breakdown: dict)
    # ──────────────────────────────────────────────────────────────────────

    def _score_moving_averages(
        self,
        close: float,
        sma_fast: float,
        sma_mid: float,
        sma_slow: float,
    ) -> tuple[float, dict]:
        """Price position relative to SMA-20/50/200."""
        score = 0.0
        breakdown: dict = {
            "current_price": round(close, 2),
            "sma_fast": round(sma_fast, 2),
            "sma_mid": round(sma_mid, 2),
            "sma_slow": round(sma_slow, 2),
        }

        if not any(pd.isna(v) for v in [sma_fast, sma_mid, sma_slow]):
            above_fast = close > sma_fast
            above_mid = close > sma_mid
            above_slow = close > sma_slow

            breakdown["above_sma20"] = bool(above_fast)
            breakdown["above_sma50"] = bool(above_mid)
            breakdown["above_sma200"] = bool(above_slow)

            score += (0.33 if above_fast else -0.33)
            score += (0.33 if above_mid else -0.33)
            score += (0.34 if above_slow else -0.34)

            # Golden cross / death cross detection
            if above_fast and above_mid and above_slow:
                breakdown["signal"] = "full_bull_alignment"
            elif not above_fast and not above_mid and not above_slow:
                breakdown["signal"] = "full_bear_alignment"
            elif above_slow and not above_fast:
                breakdown["signal"] = "pullback_in_uptrend"
            else:
                breakdown["signal"] = "mixed"

        score = max(-1.0, min(1.0, score))
        return score, breakdown

    def _compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Wilder's RSI."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _score_rsi(self, rsi: float) -> tuple[float, dict]:
        """Oversold RSI (<30) = bullish, overbought (>70) = bearish."""
        breakdown = {"rsi_value": round(rsi, 2)}

        if rsi < 20:
            score = 1.0
            breakdown["signal"] = "severely_oversold_strong_bullish"
        elif rsi < 30:
            score = 0.6 + (30 - rsi) / 30 * 0.4
            breakdown["signal"] = "oversold_bullish"
        elif rsi > 80:
            score = -1.0
            breakdown["signal"] = "severely_overbought_strong_bearish"
        elif rsi > 70:
            score = -(0.6 + (rsi - 70) / 30 * 0.4)
            breakdown["signal"] = "overbought_bearish"
        elif rsi < 40:
            score = 0.2
            breakdown["signal"] = "mildly_oversold_slight_bullish"
        elif rsi > 60:
            score = -0.2
            breakdown["signal"] = "mildly_overbought_slight_bearish"
        else:
            score = 0.0
            breakdown["signal"] = "neutral"

        return max(-1.0, min(1.0, score)), breakdown

    def _compute_macd(
        self, close: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD line, signal line, and histogram."""
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _score_macd(
        self, macd: float, signal: float, histogram: float
    ) -> tuple[float, dict]:
        """MACD above signal = bullish; strength from histogram magnitude."""
        breakdown = {
            "macd_line": round(macd, 4),
            "signal_line": round(signal, 4),
            "histogram": round(histogram, 4),
        }

        if macd > signal:
            # Normalize: scale histogram to [-1, 1] using 0.5% of price as reference
            score = min(1.0, abs(histogram) * 50)
            breakdown["crossover"] = "bullish"
        else:
            score = -min(1.0, abs(histogram) * 50)
            breakdown["crossover"] = "bearish"

        # Zero-line cross bonus
        if macd > 0 and signal > 0:
            score = min(1.0, score * 1.2)
            breakdown["zero_line"] = "above"
        elif macd < 0 and signal < 0:
            score = max(-1.0, score * 1.2)
            breakdown["zero_line"] = "below"

        return max(-1.0, min(1.0, score)), breakdown

    def _score_volume(
        self, rel_volume: float, price_change: float
    ) -> tuple[float, dict]:
        """High volume on up-day = bullish; high volume on down-day = bearish."""
        breakdown = {
            "relative_volume": round(rel_volume, 2),
            "price_change_pct": round(price_change * 100, 2),
        }

        if rel_volume >= 2.0:
            score = 0.8 if price_change > 0 else -0.8
            breakdown["signal"] = f"very_high_vol_{'up' if price_change > 0 else 'down'}"
        elif rel_volume >= 1.5:
            score = 0.5 if price_change > 0 else -0.5
            breakdown["signal"] = f"high_vol_{'up' if price_change > 0 else 'down'}"
        elif rel_volume <= 0.5:
            score = 0.0
            breakdown["signal"] = "low_volume_no_signal"
        else:
            score = 0.1 if price_change > 0 else -0.1
            breakdown["signal"] = "normal_volume"

        return max(-1.0, min(1.0, score)), breakdown

    def _score_breakout(
        self, close: float, resistance: float, support: float
    ) -> tuple[float, dict]:
        """Price breaking above resistance or below support."""
        breakdown = {
            "resistance_level": round(resistance, 2),
            "support_level": round(support, 2),
            "current_price": round(close, 2),
        }

        if close > resistance * 1.001:  # 0.1% buffer to avoid noise
            score = 1.0
            breakdown["signal"] = "resistance_breakout_bullish"
        elif close < support * 0.999:
            score = -1.0
            breakdown["signal"] = "support_breakdown_bearish"
        else:
            price_range = resistance - support
            if price_range > 0:
                position = (close - support) / price_range
                # Upper half of range = slight bullish, lower half = slight bearish
                score = (position - 0.5) * 0.4
                breakdown["range_position_pct"] = round(position * 100, 1)
                breakdown["signal"] = "within_range"
            else:
                score = 0.0
                breakdown["signal"] = "no_range"

        return max(-1.0, min(1.0, score)), breakdown
