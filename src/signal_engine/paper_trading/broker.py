"""
Alpaca Paper Trading broker wrapper.

⚠️ SAFETY: This module ONLY uses paper-api.alpaca.markets.
Live order execution is NEVER implemented in signal-engine.
The paper=True flag in TradingClient is enforced unconditionally.

Uses separate environment variables from live keys:
  ALPACA_PAPER_API_KEY
  ALPACA_PAPER_SECRET_KEY

These should be the keys from your Alpaca PAPER account,
not your live brokerage account.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Safety constant — this URL is what alpaca-py uses when paper=True
_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class PaperTradingBroker:
    """
    Alpaca paper trading broker.

    Provides:
      - get_account()          — paper account info
      - submit_market_buy()    — paper market buy
      - submit_market_sell()   — paper market sell
      - get_positions()        — current open positions
      - close_position()       — close a specific position

    All methods use the paper endpoint. No live order routing ever occurs.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_PAPER_API_KEY")
        secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY must be set in .env\n"
                "These are your Alpaca PAPER account keys (not live trading keys).\n"
                "Get them at: https://app.alpaca.markets/ → Paper account → API Keys"
            )

        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # paper=True is hardcoded — never use paper=False here
        self._client = TradingClient(api_key, secret_key, paper=True)
        self._MarketOrderRequest = MarketOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce

        logger.info("Paper trading broker initialized (endpoint: %s)", _PAPER_BASE_URL)

    def get_account(self) -> dict:
        """Get paper account info (portfolio value, buying power, cash)."""
        account = self._client.get_account()
        return {
            "id": str(account.id),
            "status": str(account.status),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "paper": True,  # explicit flag
        }

    def submit_market_buy(
        self,
        symbol: str,
        notional: Optional[float] = None,
        qty: Optional[float] = None,
    ) -> dict:
        """
        Submit a paper market buy order.

        Provide either notional (dollar amount) or qty (shares).
        """
        if notional is None and qty is None:
            raise ValueError("Provide either notional (dollars) or qty (shares)")

        kwargs: dict = {
            "symbol": symbol,
            "side": self._OrderSide.BUY,
            "time_in_force": self._TimeInForce.DAY,
        }
        if notional is not None:
            kwargs["notional"] = round(notional, 2)
        else:
            kwargs["qty"] = qty

        request = self._MarketOrderRequest(**kwargs)

        logger.info(
            "Submitting paper BUY: %s notional=$%s qty=%s",
            symbol, notional, qty,
        )
        order = self._client.submit_order(request)
        return self._order_to_dict(order)

    def submit_market_sell(
        self,
        symbol: str,
        qty: float,
    ) -> dict:
        """Submit a paper market sell order for a specified quantity (closing a long)."""
        request = self._MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=self._OrderSide.SELL,
            time_in_force=self._TimeInForce.DAY,
        )
        logger.info("Submitting paper SELL: %s qty=%s", symbol, qty)
        order = self._client.submit_order(request)
        return self._order_to_dict(order)

    def submit_market_short(
        self,
        symbol: str,
        qty: float,
    ) -> dict:
        """
        Open a paper short position (a sell order with no existing long position).

        Requires the paper account to have margin/shorting enabled for the
        symbol — true by default for marginable symbols on a standard Alpaca
        paper account. Requires whole-share qty; Alpaca does not support
        notional (fractional) orders on the sell side.
        """
        request = self._MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=self._OrderSide.SELL,
            time_in_force=self._TimeInForce.DAY,
        )
        logger.info("Submitting paper SHORT: %s qty=%s", symbol, qty)
        order = self._client.submit_order(request)
        return self._order_to_dict(order)

    def get_positions(self) -> list[dict]:
        """Return all current open positions in the paper account."""
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            logger.warning("Could not fetch positions: %s", exc)
            return []

        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
                "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
                "market_value": float(p.market_value) if p.market_value else None,
            }
            for p in positions
        ]

    def close_position(self, symbol: str) -> dict:
        """Close an entire position for a symbol."""
        logger.info("Closing paper position: %s", symbol)
        order = self._client.close_position(symbol)
        return self._order_to_dict(order)

    def get_open_orders(self) -> list[dict]:
        """Return all open (pending) orders."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self._client.get_orders(filter=request)
        return [self._order_to_dict(o) for o in orders]

    def _order_to_dict(self, order) -> dict:
        """Convert an Alpaca order object to a plain dict."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": str(order.side),
            "order_type": str(order.order_type) if hasattr(order, "order_type") else "market",
            "qty": float(order.qty) if order.qty else None,
            "notional": float(order.notional) if order.notional else None,
            "status": str(order.status),
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "paper": True,
        }
