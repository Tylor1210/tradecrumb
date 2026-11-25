"""
Entry and add rules for the "beaten-down quality stock" LEAPS idea.

The rules come from a trading video and are tested here on the stocks
themselves. Everything is computed from data up to and including the bar
being judged, so a signal on day t only uses closes up to day t. Trades
are entered at the next day's open.

Rules
  setup       close is down drawdown_min (50%) or more from the highest high
              seen so far (all available history, or high_lookback_bars)
  oversold    daily RSI(14) under rsi_max (30) on the signal bar or one of
              the rsi_lookback - 1 bars before it
  pin bar     bullish pin bar on the signal bar: long lower wick, small body
              sitting in the top part of the bar's range
  add         while holding, price makes a lower low than the signal bar's
              low while RSI makes a higher low than the signal bar's RSI,
              RSI is still weak (under divergence_rsi_max) and the bar closes
              green. One add by default.

Depends on: pandas/numpy only. Pure functions, no I/O.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DipRules:
    drawdown_min: float = 0.50        # 0.50 = close at least 50% below the high
    high_lookback_bars: Optional[int] = None  # None = highest high in all history loaded
    min_history_bars: int = 252       # need a year of bars before the high means anything
    rsi_period: int = 14
    rsi_max: float = 30.0
    rsi_lookback: int = 3             # RSI under rsi_max on the pin bar or the 2 bars before
    pin_wick_to_body: float = 2.0     # lower wick at least 2x the body
    pin_wick_to_range: float = 0.5    # lower wick at least half of the bar's range
    pin_body_top: float = 0.6         # bottom of the body sits in the top 40% of the range
    divergence_rsi_max: float = 45.0
    divergence_min_bars: int = 5      # bars after the signal before an add can trigger
    max_adds: int = 1

    @classmethod
    def from_config(cls, cfg: dict) -> "DipRules":
        known = {k: v for k, v in (cfg or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, same formula as signals/technical.py."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # No losses at all in the window means RSI is 100, not missing
    rsi = rsi.where(~(avg_loss == 0) | avg_gain.isna(), 100.0)
    return rsi


def bullish_pin_bar(df: pd.DataFrame, rules: DipRules) -> pd.Series:
    """True on bars with a long lower wick and a small body near the top."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    lower_wick = np.minimum(o, c) - l
    body_bottom_pos = (np.minimum(o, c) - l) / rng  # 0 = at the low, 1 = at the high
    is_pin = (
        (lower_wick >= rules.pin_wick_to_body * body)
        & (lower_wick >= rules.pin_wick_to_range * rng)
        & (body_bottom_pos >= rules.pin_body_top)
    )
    return is_pin.fillna(False).astype(bool)


def drawdown_from_high(df: pd.DataFrame, rules: DipRules) -> pd.Series:
    """Close vs the highest high so far, as a negative fraction (-0.6 = down 60%)."""
    if rules.high_lookback_bars:
        peak = df["high"].rolling(rules.high_lookback_bars, min_periods=1).max()
    else:
        peak = df["high"].cummax()
    dd = df["close"] / peak - 1.0
    bars_seen = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    return dd.where(bars_seen >= rules.min_history_bars)


def add_indicators(df: pd.DataFrame, rules: DipRules) -> pd.DataFrame:
    """Return a copy of df with rsi, drawdown, pin_bar and signal columns."""
    out = df.copy()
    out["rsi"] = wilder_rsi(out["close"], rules.rsi_period)
    out["drawdown"] = drawdown_from_high(out, rules)
    out["pin_bar"] = bullish_pin_bar(out, rules)
    recent_rsi_min = out["rsi"].rolling(rules.rsi_lookback, min_periods=1).min()
    out["ref_rsi"] = recent_rsi_min
    out["ref_low"] = out["low"].rolling(rules.rsi_lookback, min_periods=1).min()
    out["signal"] = (
        (out["drawdown"] <= -rules.drawdown_min)
        & (recent_rsi_min < rules.rsi_max)
        & out["pin_bar"]
    ).fillna(False)
    return out


def divergence_add(
    ind: pd.DataFrame,
    i: int,
    ref_low: float,
    ref_rsi: float,
    rules: DipRules,
) -> bool:
    """
    Bullish RSI divergence on bar i against the signal bar's low and RSI:
    lower low in price, higher low in RSI, RSI still weak, green close.
    """
    row = ind.iloc[i]
    if pd.isna(row["rsi"]):
        return False
    return bool(
        row["low"] < ref_low
        and row["rsi"] > ref_rsi
        and row["rsi"] < rules.divergence_rsi_max
        and row["close"] > row["open"]
    )
