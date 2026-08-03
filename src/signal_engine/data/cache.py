"""
Parquet-backed local cache layer.

Prevents re-fetching API data on repeated runs (important for staying
within free-tier rate limits). Cache files are stored in data/cache/
which is gitignored.

TTL is configurable per use case:
  - Price data: 24h (markets close daily)
  - News data: 6h (news updates throughout the day)
  - Financial metrics: 7 days (quarterly updates)
"""

import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directory — relative to project root
DEFAULT_CACHE_DIR = Path("data/cache")


class ParquetCache:
    """
    Simple Parquet file cache with TTL-based expiry.

    Keys are hashed to prevent filesystem path issues.
    Each cached item consists of:
      - {hash}.parquet  — the DataFrame
      - {hash}.meta     — pickle with creation timestamp + original key
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        default_ttl_hours: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def _parquet_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._hash_key(key)}.parquet"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._hash_key(key)}.meta"

    def get(self, key: str, ttl_hours: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve a cached DataFrame if it exists and hasn't expired.

        Returns None on cache miss or expiry.
        """
        parquet_path = self._parquet_path(key)
        meta_path = self._meta_path(key)

        if not parquet_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            return None

        ttl = timedelta(hours=ttl_hours) if ttl_hours is not None else self.default_ttl
        age = datetime.now() - meta.get("created_at", datetime.min)

        if age > ttl:
            logger.debug("Cache expired for key: %s", meta.get("key", key))
            self._evict(key)
            return None

        try:
            df = pd.read_parquet(parquet_path)
            logger.debug("Cache hit: %s (age: %s)", meta.get("key", key), age)
            return df
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", key, exc)
            self._evict(key)
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store a DataFrame in the cache."""
        if df is None or df.empty:
            logger.debug("Skipping cache write — empty DataFrame for key: %s", key)
            return

        parquet_path = self._parquet_path(key)
        meta_path = self._meta_path(key)

        try:
            df.to_parquet(parquet_path, index=True)
            with open(meta_path, "wb") as f:
                pickle.dump({"created_at": datetime.now(), "key": key}, f)
            logger.debug("Cache written: %s", key)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", key, exc)

    def _evict(self, key: str) -> None:
        """Remove a cached item."""
        self._parquet_path(key).unlink(missing_ok=True)
        self._meta_path(key).unlink(missing_ok=True)

    def invalidate(self, key: str) -> None:
        """Explicitly invalidate a cache entry."""
        self._evict(key)
        logger.debug("Cache invalidated: %s", key)

    def clear_all(self) -> None:
        """Nuke the entire cache. Use with care."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink(missing_ok=True)
        for f in self.cache_dir.glob("*.meta"):
            f.unlink(missing_ok=True)
        logger.info("Cache cleared: %s", self.cache_dir)

    def cache_key_for_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> str:
        """Generate a deterministic cache key for a price data request."""
        symbols_str = ",".join(sorted(symbols))
        return f"bars:{symbols_str}:{start.date()}:{end.date()}:{timeframe}"

    def cache_key_for_news(self, ticker: str, from_date: str, to_date: str) -> str:
        """Generate a deterministic cache key for a news request."""
        return f"news:{ticker}:{from_date}:{to_date}"
