"""
Alpaca Market Data client.

Fetches historical OHLCV bars and latest quotes for watchlist tickers.
Uses alpaca-py (the modern SDK). All credentials are loaded from environment
variables — never hardcoded.

Free tier limits:
  - Historical bars: up to 2 years of minute bars, unlimited daily bars
  - Rate limits: ~200 requests/minute on free tier
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaDataClient:
    """
    Wrapper around alpaca-py's StockHistoricalDataClient.

    Provides:
      - get_historical_bars()  — bulk OHLCV data for a list of symbols
      - get_latest_bar()       — most recent bar for a single symbol
    """

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env\n"
                "Get free keys at: https://app.alpaca.markets/"
            )

        # Import here so tests can mock without alpaca-py installed
        from alpaca.data.historical import StockHistoricalDataClient

        self._api_key = api_key
        self._secret_key = secret_key
        self._client = StockHistoricalDataClient(api_key, secret_key)

    def get_historical_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe_str: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for a list of symbols.

        Args:
            symbols:       List of ticker symbols (e.g. ["AAPL", "NVDA"])
            start:         Start datetime (UTC)
            end:           End datetime (UTC)
            timeframe_str: "1Day", "1Hour", "15Min", "5Min", "1Min"

        Returns:
            Dict mapping ticker → DataFrame with columns:
            [open, high, low, close, volume, trade_count, vwap]
            Index is DatetimeIndex (UTC).
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        _tf_map = {
            "1Day": TimeFrame.Day,
            "1Hour": TimeFrame.Hour,
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "1Min": TimeFrame.Minute,
        }
        timeframe = _tf_map.get(timeframe_str, TimeFrame.Day)

        # Filter out private/unlisted symbols (e.g. SPACEX)
        valid_symbols = [s for s in symbols if s.upper() != "SPACEX"]

        if not valid_symbols:
            return {}

        logger.info(
            "Fetching %s bars for %d symbols (%s → %s)",
            timeframe_str,
            len(valid_symbols),
            start.date(),
            end.date(),
        )

        request = StockBarsRequest(
            symbol_or_symbols=valid_symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment="all",  # split + dividend adjusted
            feed="iex",        # use IEX feed on free tier
        )

        try:
            bars = self._client.get_stock_bars(request)
        except Exception as exc:
            logger.error("Alpaca bar fetch failed: %s", exc)
            return {}

        raw_df = bars.df
        if raw_df.empty:
            logger.warning("Alpaca returned empty DataFrame for %s", valid_symbols)
            return {}

        result: dict[str, pd.DataFrame] = {}

        # Multi-index: (symbol, timestamp)
        if isinstance(raw_df.index, pd.MultiIndex):
            for symbol in valid_symbols:
                try:
                    ticker_df = raw_df.loc[symbol].copy()
                    ticker_df.index = pd.to_datetime(ticker_df.index, utc=True)
                    # Normalize column names to lowercase
                    ticker_df.columns = [c.lower() for c in ticker_df.columns]
                    result[symbol] = ticker_df
                except KeyError:
                    logger.warning("No data returned for %s", symbol)
        else:
            # Single symbol — index is just timestamps
            raw_df.index = pd.to_datetime(raw_df.index, utc=True)
            raw_df.columns = [c.lower() for c in raw_df.columns]
            if valid_symbols:
                result[valid_symbols[0]] = raw_df

        logger.info("Successfully fetched data for %d / %d symbols", len(result), len(valid_symbols))
        return result

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        """
        Fetch the most recent bar for a single symbol.
        Used by paper trading to get current price.
        """
        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(symbol_or_symbols=symbol)
        try:
            bars = self._client.get_stock_latest_bar(request)
            bar = bars.get(symbol)
            if bar:
                return {
                    "symbol": symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "timestamp": bar.timestamp,
                }
        except Exception as exc:
            logger.error("Could not fetch latest bar for %s: %s", symbol, exc)
        return None

    def get_most_active_symbols(self, top: int = 100, by: str = "volume") -> list[str]:
        """
        Fetch the top N most-actively-traded US symbols via Alpaca's screener API.

        Args:
            top: number of symbols to return.
            by:  "volume" or "trades" (see alpaca.data.requests.MostActivesBy).

        Returns [] (logged) on failure — callers should treat this as best-effort
        and fall back to whatever core watchlist they have.
        """
        from alpaca.data.historical.screener import ScreenerClient
        from alpaca.data.requests import MostActivesBy, MostActivesRequest

        by_enum = MostActivesBy.TRADES if by == "trades" else MostActivesBy.VOLUME

        try:
            screener = ScreenerClient(self._api_key, self._secret_key)
            request = MostActivesRequest(top=top, by=by_enum)
            response = screener.get_most_actives(request)
            symbols = [a.symbol for a in response.most_actives]
            logger.info("Fetched %d most-active symbols (by=%s)", len(symbols), by)
            return symbols
        except Exception as exc:
            logger.error("Most-actives screener call failed: %s", exc)
            return []
