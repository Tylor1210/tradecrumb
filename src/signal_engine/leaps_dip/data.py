"""
Universe and daily-bar loading for the dip backtest.

The universe lives in config/leaps_dip.yaml:
  universe.tickers   large caps under their current symbols
  universe.delisted  names that later failed, were bought out or renamed,
                     each with an asof date when the old symbol still traded

Why the delisted list matters: this strategy buys stocks after a crash, and
the crashes that never came back (SVB, First Republic, Chesapeake in 2020)
are exactly the trades a survivors-only list would quietly leave out.

Bars come from Alpaca's SIP feed (all exchanges, history back to 2016 for
free accounts), split and dividend adjusted, cached per symbol for 24h.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def universe_from_config(cfg: dict) -> list[tuple[str, Optional[str]]]:
    """[(symbol, asof or None), ...] deduped, current tickers first."""
    uni = cfg.get("universe", {}) or {}
    out: list[tuple[str, Optional[str]]] = []
    seen = set()
    for sym in uni.get("tickers", []) or []:
        s = str(sym).upper().strip()
        if s and s not in seen:
            seen.add(s)
            out.append((s, None))
    for item in uni.get("delisted", []) or []:
        s = str(item.get("symbol", "")).upper().strip()
        if s and s not in seen:
            seen.add(s)
            asof = item.get("asof")
            out.append((s, str(asof) if asof else None))
    return out


def load_daily_bars(
    alpaca,
    universe: list[tuple[str, Optional[str]]],
    start: datetime,
    end: datetime,
    cache=None,
    feed: str = "sip",
    chunk: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for every (symbol, asof) pair; cached per symbol."""
    result: dict[str, pd.DataFrame] = {}
    to_fetch: dict[Optional[str], list[str]] = {}
    for sym, asof in universe:
        key = f"leaps_dip:bars:{feed}:{sym}:{asof}:{start.date()}:{end.date()}"
        hit = cache.get(key) if cache is not None else None
        if hit is not None and not hit.empty:
            result[sym] = hit
        else:
            to_fetch.setdefault(asof, []).append(sym)

    for asof, syms in to_fetch.items():
        for i in range(0, len(syms), chunk):
            batch = syms[i:i + chunk]
            bars = alpaca.get_historical_bars(batch, start, end, "1Day", feed=feed, asof=asof)
            if not bars and feed != "iex":
                logger.warning("%s feed returned nothing for %s; trying iex", feed, batch[:3])
                bars = alpaca.get_historical_bars(batch, start, end, "1Day", feed="iex", asof=asof)
            for sym, df in bars.items():
                if df is None or df.empty:
                    continue
                result[sym] = df
                if cache is not None:
                    key = f"leaps_dip:bars:{feed}:{sym}:{asof}:{start.date()}:{end.date()}"
                    cache.set(key, df)
    return result
