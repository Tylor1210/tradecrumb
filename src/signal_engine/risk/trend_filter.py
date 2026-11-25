"""
Market trend filter: only open new longs while the market is in an uptrend.

What it does: drops buy signals on days when the index (SPY by default)
closes below its N-day simple moving average. Optionally also drops a
ticker's signal when that ticker closes below its own N-day average.
Trades already open are left to their normal exits.

Why: this signal set is long-only and leans on momentum (price above its
averages, MACD, breakouts). Long entries taken while the whole market is
falling (March 2020, most of 2022) are where the worst trades and the
deepest account drawdowns come from. A 200-day trend filter on the index is
the classic way to stand aside then (Faber 2007, "A Quantitative Approach to
Tactical Asset Allocation"; Moskowitz, Ooi and Pedersen 2012, "Time Series
Momentum"). It usually costs some upside in quick V-shaped recoveries.

Point in time: the filter on a given bar uses that bar's close and the
average through that bar, the same close the signal fires on and the entry
fills at. Nothing later is seen.

Called by: cli.py `backtest` (config trend_filter, --compare-trend-filter).
Depends on: pandas only. Pure functions, no I/O.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TrendFilterRule:
    index_symbol: str = "SPY"
    sma_days: int = 200
    require_ticker_above_sma: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["TrendFilterRule"]:
        """None when the config block is missing or enabled is false."""
        if not cfg or not cfg.get("enabled", False):
            return None
        return cls.with_defaults(cfg)

    @classmethod
    def with_defaults(cls, cfg: Optional[dict]) -> "TrendFilterRule":
        cfg = cfg or {}
        return cls(
            index_symbol=str(cfg.get("index_symbol", "SPY")),
            sma_days=int(cfg.get("sma_days", 200)),
            require_ticker_above_sma=bool(cfg.get("require_ticker_above_sma", False)),
        )

    def describe(self) -> str:
        text = f"{self.index_symbol} above its {self.sma_days}-day average"
        if self.require_ticker_above_sma:
            text += f", ticker above its own {self.sma_days}-day average"
        return text


def uptrend_mask(close: pd.Series, sma_days: int) -> pd.Series:
    """True on bars where close is above its sma_days simple average. False until the average exists."""
    close = close.astype(float)
    sma = close.rolling(sma_days, min_periods=sma_days).mean()
    return (close > sma).fillna(False)


def apply_trend_filter(
    signals: dict[str, pd.Series],
    price_data: dict[str, pd.DataFrame],
    rule: TrendFilterRule,
) -> tuple[dict[str, pd.Series], int]:
    """
    Return (filtered signals, number of buy signals removed).

    price_data must hold rule.index_symbol. Signal dates the index has no bar
    for take the index's last known state (forward filled), never a later one.
    """
    index_df = price_data.get(rule.index_symbol)
    if index_df is None or "close" not in index_df:
        raise ValueError(
            f"Trend filter needs daily bars for {rule.index_symbol}; add it to the data you load."
        )
    market_ok = uptrend_mask(index_df["close"].sort_index(), rule.sma_days)

    filtered: dict[str, pd.Series] = {}
    removed = 0
    for ticker, sig in signals.items():
        sig = sig.fillna(False).astype(bool)
        ok = market_ok.reindex(sig.index, method="ffill").fillna(False).astype(bool)
        if rule.require_ticker_above_sma and ticker in price_data:
            own = uptrend_mask(price_data[ticker]["close"].sort_index(), rule.sma_days)
            ok &= own.reindex(sig.index, method="ffill").fillna(False).astype(bool)
        new_sig = sig & ok
        removed += int(sig.sum() - new_sig.sum())
        filtered[ticker] = new_sig
    return filtered, removed
