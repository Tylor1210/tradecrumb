"""Tests for weekend_swing.universe.WeekendUniverseBuilder."""

from unittest.mock import MagicMock

from signal_engine.weekend_swing.universe import WeekendUniverseBuilder

CONFIG = {
    "universe": {
        "top_n": 100,
        "by": "volume",
        "core_tickers": ["SPY", "QQQ", "NVDA", "MSFT", "GOOG"],
    }
}


class TestWeekendUniverseBuilder:
    def test_core_set_always_included(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = ["AAPL", "TSLA"]

        universe = WeekendUniverseBuilder(CONFIG).build(alpaca)

        for core in ["SPY", "QQQ", "NVDA", "MSFT", "GOOG"]:
            assert core in universe

    def test_core_set_survives_screener_failure(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = []

        universe = WeekendUniverseBuilder(CONFIG).build(alpaca)

        assert universe == ["SPY", "QQQ", "NVDA", "MSFT", "GOOG"]

    def test_dedupes_when_core_ticker_also_in_screener_results(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = ["NVDA", "AAPL", "SPY"]

        universe = WeekendUniverseBuilder(CONFIG).build(alpaca)

        assert universe.count("NVDA") == 1
        assert universe.count("SPY") == 1
        assert universe == ["SPY", "QQQ", "NVDA", "MSFT", "GOOG", "AAPL"]

    def test_passes_configured_top_n_and_by_to_client(self):
        alpaca = MagicMock()
        alpaca.get_most_active_symbols.return_value = []
        cfg = {"universe": {"top_n": 50, "by": "trades", "core_tickers": []}}

        WeekendUniverseBuilder(cfg).build(alpaca)

        alpaca.get_most_active_symbols.assert_called_once_with(top=50, by="trades")
