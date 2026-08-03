"""
Weekend Swing Signals — universe builder.

Pulls the top-N most-actively-traded US symbols via Alpaca's screener API
(volume-ranked by default, unfiltered by price/quality — this matches the
"most-actively-traded" spec literally, so low-priced high-volume tickers can
show up alongside mega-caps) and merges in an always-included core set.
"""

import logging

logger = logging.getLogger(__name__)


class WeekendUniverseBuilder:
    """
    Builds the ticker universe for a weekend scan: top-N most active symbols
    (via AlpacaDataClient.get_most_active_symbols) plus a core set that is
    always included regardless of screener ranking.
    """

    def __init__(self, config: dict) -> None:
        universe_cfg = config.get("universe", {})
        self.top_n = universe_cfg.get("top_n", 100)
        self.by = universe_cfg.get("by", "volume")
        self.core_tickers = [t.upper() for t in universe_cfg.get("core_tickers", [])]

    def build(self, alpaca_client) -> list[str]:
        """
        Returns the deduped ticker universe: core tickers first, then the
        screener's most-actives, in screener rank order. Falls back to the
        core set alone (logged) if the screener call fails.
        """
        most_active = alpaca_client.get_most_active_symbols(top=self.top_n, by=self.by)
        if not most_active:
            logger.warning(
                "Most-actives screener returned nothing — falling back to core set only (%s)",
                self.core_tickers,
            )

        seen = set()
        universe: list[str] = []
        for symbol in self.core_tickers + most_active:
            symbol = symbol.upper()
            if symbol not in seen:
                seen.add(symbol)
                universe.append(symbol)

        logger.info(
            "Weekend swing universe: %d tickers (%d core + %d from screener, deduped)",
            len(universe), len(self.core_tickers), len(most_active),
        )
        return universe
