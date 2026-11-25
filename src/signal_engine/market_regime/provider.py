"""
Point-in-time regime provider: fetches the daily bars the regime needs (through
the existing IntradayBarStore, so they're cached on disk) and computes the
regime as of any Monday.

Called by: cli.py (`regime`, `weekly-cycle`), weekend_swing/backtest.py,
dashboard/server.py. Depends on: regime.compute_regime, IntradayBarStore.

WHY one fetch for the whole range: a backtest asks for ~50 Mondays; fetching
the bars once and slicing per as_of keeps API calls at one per symbol.
"""

import logging
from typing import Optional

import pandas as pd

from signal_engine.market_regime.regime import MarketRegime, compute_regime

logger = logging.getLogger(__name__)

# Enough history for the longest lookback (125d SMA) plus slack for holidays.
_WARMUP_DAYS = 220


def regime_symbols(cfg: dict) -> list:
    s = cfg.get("symbols", {})
    out = [s.get("benchmark", "SPY"), s.get("bonds_long", "TLT"), s.get("junk", "HYG"),
           s.get("investment_grade", "LQD")]
    out += list(s.get("offensive", [])) + list(s.get("defensive", [])) + list(s.get("sectors", []))
    return list(dict.fromkeys(x for x in out if x))


class RegimeProvider:
    def __init__(self, cfg: dict, bar_store, breadth_symbols: Optional[list] = None) -> None:
        self.cfg = cfg
        self.bar_store = bar_store
        self.breadth_symbols = [t for t in (breadth_symbols or []) if t]
        self._daily: dict = {}
        self._loaded_range: Optional[tuple] = None
        self._cache: dict = {}

    def _load(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        if self._loaded_range and self._loaded_range[0] <= start and self._loaded_range[1] >= end:
            return
        fetch_start = start - pd.Timedelta(days=_WARMUP_DAYS)
        for sym in dict.fromkeys(regime_symbols(self.cfg) + self.breadth_symbols):
            try:
                df = self.bar_store.get_bars(sym, "1Day", fetch_start, end)
            except Exception as exc:
                logger.warning("Regime: could not load %s: %s", sym, exc)
                continue
            if df is not None and not df.empty:
                self._daily[sym] = df
        self._loaded_range = (start, end)

    def as_of(self, when: pd.Timestamp) -> Optional[MarketRegime]:
        """Regime using only bars strictly before `when` (pass the Monday for a weekly regime)."""
        when = when.tz_localize("UTC") if when.tzinfo is None else when
        key = when.normalize()
        if key not in self._cache:
            self._load(key, key + pd.Timedelta(days=1))
            self._cache[key] = compute_regime(self._daily, self.cfg, self.breadth_symbols, as_of=key)
        return self._cache[key]

    def for_range(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Pre-load once for a whole backtest range so per-week calls make no API requests."""
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        self._load(start, end)
