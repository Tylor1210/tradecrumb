"""
LEAPS Options Chain Data Stub.

This module provides a structured-but-non-functional interface for
options chain data. It is intentionally stubbed until credentials
for an options data provider are configured.

═══════════════════════════════════════════════════════════════════
DATA SOURCE OPTIONS (choose one to activate)
═══════════════════════════════════════════════════════════════════

PRIMARY RECOMMENDATION: Polygon.io (free tier available)
  Status:   ✅ Free tier with API key (5 req/min, end-of-day data)
            💰 Paid tiers for real-time options data
  Endpoint: GET https://api.polygon.io/v3/snapshot/options/{symbol}
  Chain:    GET https://api.polygon.io/v3/snapshot/options/{symbol}?expiration_date.gte=...
  Greeks:   Included in snapshot response (delta, gamma, theta, vega)
  Docs:     https://polygon.io/docs/options/get_v3_snapshot_options_ticker
  Activate: Set POLYGON_API_KEY in .env → calls _get_chain_polygon()

OPTION 2: Alpaca Options API (limited beta as of mid-2026)
  Status:   🔒 Limited beta — requires upgraded brokerage account
            Not available on free/paper-only accounts
  Endpoint: GET https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}
  Docs:     https://docs.alpaca.markets/reference/optionsnapshots
  Activate: Set ALPACA_OPTIONS_ENABLED=true in .env once account is approved

OPTION 3: Tradier Brokerage API (free paper account)
  Status:   ✅ Free paper account available
  Endpoint: GET https://sandbox.tradier.com/v1/markets/options/chains
  Docs:     https://documentation.tradier.com/brokerage-api/markets/get-options-chains
  Activate: Set TRADIER_API_KEY and TRADIER_ACCOUNT_ID in .env

═══════════════════════════════════════════════════════════════════
DATA FIELDS NEEDED FOR LEAPS ANALYSIS
═══════════════════════════════════════════════════════════════════
  - Strike price
  - Expiration date (filtering DTE ≥ 365)
  - Option type ("call" / "put")
  - Days to expiration (DTE)
  - Implied volatility (IV) surface
  - Open interest
  - Bid / ask spread (for liquidity assessment)
  - Greeks: delta, gamma, theta, vega
  - Last traded price
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class OptionsContract:
    """A single options contract in the chain."""

    ticker: str
    strike: float
    expiration: str       # "YYYY-MM-DD"
    option_type: str      # "call" or "put"
    dte: int              # days to expiration

    # Populated when data source is connected
    iv: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @property
    def bid_ask_spread(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return round(self.ask - self.bid, 4)
        return None

    @property
    def is_leaps(self) -> bool:
        return self.dte >= 365


@dataclass
class OptionsChain:
    """Full options chain for a single underlying ticker."""

    ticker: str
    as_of: datetime
    contracts: list[OptionsContract] = field(default_factory=list)
    source: str = "stub"

    @property
    def leaps_contracts(self) -> list[OptionsContract]:
        """Return only LEAPS contracts (DTE ≥ 365)."""
        return [c for c in self.contracts if c.is_leaps]

    @property
    def leaps_calls(self) -> list[OptionsContract]:
        return [c for c in self.leaps_contracts if c.option_type == "call"]

    @property
    def leaps_puts(self) -> list[OptionsContract]:
        return [c for c in self.leaps_contracts if c.option_type == "put"]


class OptionsChainClient:
    """
    Stubbed options chain client.

    Automatically selects the available data source based on configured env vars.
    Raises NotImplementedError with setup instructions if none are configured.

    Activation priority:
      1. Alpaca Options (if ALPACA_OPTIONS_ENABLED=true)
      2. Polygon.io (if POLYGON_API_KEY is set)  ← recommended
      3. Tradier (if TRADIER_API_KEY is set)
    """

    def __init__(self) -> None:
        self.alpaca_options_enabled = (
            os.getenv("ALPACA_OPTIONS_ENABLED", "false").lower() == "true"
        )
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.tradier_key = os.getenv("TRADIER_API_KEY", "")
        self.tradier_account = os.getenv("TRADIER_ACCOUNT_ID", "")

    @property
    def is_connected(self) -> bool:
        return bool(
            self.alpaca_options_enabled
            or self.polygon_key
            or self.tradier_key
        )

    def get_chain(self, ticker: str, min_dte: int = 365) -> OptionsChain:
        """
        Fetch options chain for a ticker, filtered to DTE ≥ min_dte.

        Selects the appropriate provider based on configured env vars.
        Raises NotImplementedError if no provider is configured.
        """
        if self.alpaca_options_enabled:
            return self._get_chain_alpaca(ticker, min_dte)
        elif self.polygon_key:
            return self._get_chain_polygon(ticker, min_dte)
        elif self.tradier_key:
            return self._get_chain_tradier(ticker, min_dte)
        else:
            raise NotImplementedError(
                "\n"
                "Options chain data is not configured.\n"
                "To activate, set one of these in your .env file:\n\n"
                "  # Recommended (free tier available):\n"
                "  POLYGON_API_KEY=your_polygon_key\n\n"
                "  # Alpaca (requires upgraded account):\n"
                "  ALPACA_OPTIONS_ENABLED=true\n\n"
                "  # Tradier (free paper account):\n"
                "  TRADIER_API_KEY=your_tradier_key\n"
                "  TRADIER_ACCOUNT_ID=your_account_id\n\n"
                "See src/signal_engine/leaps/options_stub.py for full docs."
            )

    # ── Provider implementations ──────────────────────────────────────────

    def _get_chain_polygon(self, ticker: str, min_dte: int) -> OptionsChain:
        """
        Fetch options chain from Polygon.io.

        TODO(connect): Implement this method using the Polygon Python SDK.

        Endpoint:
          GET https://api.polygon.io/v3/snapshot/options/{underlyingAsset}
          Params:
            expiration_date.gte = (today + min_dte).strftime("%Y-%m-%d")
            limit               = 250
            apiKey              = POLYGON_API_KEY

        Response mapping:
          result.details.strike_price    → OptionsContract.strike
          result.details.expiration_date → OptionsContract.expiration
          result.details.contract_type   → OptionsContract.option_type
          result.greeks.delta            → OptionsContract.delta
          result.greeks.gamma            → OptionsContract.gamma
          result.greeks.theta            → OptionsContract.theta
          result.greeks.vega             → OptionsContract.vega
          result.implied_volatility      → OptionsContract.iv
          result.open_interest           → OptionsContract.open_interest
          result.day.volume              → OptionsContract.volume
          result.last_quote.bid          → OptionsContract.bid
          result.last_quote.ask          → OptionsContract.ask

        Example using polygon-api-client:
          from polygon import RESTClient
          client = RESTClient(api_key=self.polygon_key)
          min_exp = (datetime.now() + timedelta(days=min_dte)).strftime("%Y-%m-%d")
          for result in client.list_snapshot_options_chain(
              ticker,
              params={"expiration_date.gte": min_exp, "limit": 250},
          ):
              # map result → OptionsContract
              ...
        """
        raise NotImplementedError(
            "Polygon.io options chain: TODO(connect) in _get_chain_polygon().\n"
            "See docstring for response mapping and example code."
        )

    def _get_chain_alpaca(self, ticker: str, min_dte: int) -> OptionsChain:
        """
        Fetch options chain from Alpaca Options API (limited beta).

        TODO(connect): Implement when Alpaca options account is provisioned.

        Endpoint:
          GET https://data.alpaca.markets/v1beta1/options/snapshots/{ticker}
          Headers:
            APCA-API-KEY-ID:     ALPACA_API_KEY
            APCA-API-SECRET-KEY: ALPACA_SECRET_KEY

        Note: Requires upgraded Alpaca account with options trading enabled.
              Not available on free-tier or paper-only accounts as of mid-2026.

        Response shape:
          snapshots[contractSymbol].greeks.delta → delta
          snapshots[contractSymbol].impliedVolatility → iv
          (See Alpaca options API reference for full schema)
        """
        raise NotImplementedError(
            "Alpaca options API: ALPACA_OPTIONS_ENABLED=true but account may not have options access.\n"
            "Contact Alpaca to enable options trading on your account.\n"
            "See TODO(connect) in _get_chain_alpaca() when access is granted."
        )

    def _get_chain_tradier(self, ticker: str, min_dte: int) -> OptionsChain:
        """
        Fetch options chain from Tradier API.

        TODO(connect): Implement using Tradier's options chain endpoint.

        Endpoint:
          GET https://sandbox.tradier.com/v1/markets/options/chains
          Params:
            symbol     = ticker
            expiration = date string (need to fetch expiration dates first)
            greeks     = true
          Headers:
            Authorization: Bearer {TRADIER_API_KEY}
            Accept: application/json

        Note: Tradier requires fetching available expiration dates first:
          GET /v1/markets/options/expirations?symbol={ticker}
          Then loop over expirations where DTE ≥ min_dte.
        """
        raise NotImplementedError(
            "Tradier options chain: TODO(connect) in _get_chain_tradier().\n"
            "See docstring for API details."
        )
