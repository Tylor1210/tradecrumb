"""
Weekend Swing Signals — persistent, incrementally-topped-up intraday bar store.

Distinct from data/cache.py's ParquetCache (TTL/expiry-based, built for
daily-scan freshness): this store never expires. A ticker's history only
grows — each call tops up whatever's missing at the edges of what's already
stored, so a second run over the same date range makes zero API calls. This
is what backs the "build a historical database, don't re-fetch next cycle"
requirement for backtesting.

Limitation: only extends at the start/end edges of what's stored, it doesn't
backfill arbitrary interior gaps. That's fine for this always-growing
time-series use case (each call typically asks for "up through today").

Fetches go through the existing AlpacaDataClient (no new data source).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path("data/weekend_swing_bars")


class IntradayBarStore:
    def __init__(self, alpaca_client, store_dir: Path = DEFAULT_STORE_DIR) -> None:
        self.alpaca_client = alpaca_client
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, timeframe: str) -> Path:
        return self.store_dir / f"{ticker.upper()}_{timeframe}.parquet"

    def _load(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._path(ticker, timeframe)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return df if not df.empty else None
        except Exception as exc:
            logger.warning("Failed to read cached bars for %s/%s: %s", ticker, timeframe, exc)
            return None

    def _save(self, ticker: str, timeframe: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(ticker, timeframe))

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Returns bars for `ticker`/`timeframe` covering [start, end], fetching
        only whatever isn't already covered at the edges of the cached range.
        """
        existing = self._load(ticker, timeframe)
        fetch_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        if existing is None:
            fetch_ranges.append((start, end))
        else:
            existing_min, existing_max = existing.index.min(), existing.index.max()
            if start < existing_min:
                fetch_ranges.append((start, existing_min))
            if end > existing_max:
                fetch_ranges.append((existing_max, end))

        if not fetch_ranges:
            combined = existing
        else:
            fetched_frames = []
            for fetch_start, fetch_end in fetch_ranges:
                logger.info(
                    "IntradayBarStore: fetching %s %s bars %s -> %s (cache miss)",
                    ticker, timeframe, fetch_start.date(), fetch_end.date(),
                )
                result = self.alpaca_client.get_historical_bars(
                    [ticker], fetch_start, fetch_end, timeframe_str=timeframe
                )
                df = result.get(ticker)
                if df is not None and not df.empty:
                    fetched_frames.append(df)
                else:
                    logger.warning(
                        "IntradayBarStore: no data returned for %s %s %s -> %s",
                        ticker, timeframe, fetch_start.date(), fetch_end.date(),
                    )

            parts = ([existing] if existing is not None else []) + fetched_frames
            if not parts:
                return pd.DataFrame()

            combined = pd.concat(parts)
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            self._save(ticker, timeframe, combined)

        if combined is None or combined.empty:
            return pd.DataFrame()
        return combined.loc[(combined.index >= start) & (combined.index <= end)]
