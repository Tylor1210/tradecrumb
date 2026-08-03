"""Tests for weekend_swing.earnings.get_earnings_exclusions."""

from unittest.mock import MagicMock

import pandas as pd

from signal_engine.weekend_swing.earnings import get_earnings_exclusions

WEEK_START = pd.Timestamp("2024-01-08")
WEEK_END = pd.Timestamp("2024-01-12")


class TestGetEarningsExclusions:
    def test_returns_symbols_from_calendar(self):
        client = MagicMock()
        client.get_earnings_calendar.return_value = [
            {"symbol": "AAPL", "date": "2024-01-09"},
            {"symbol": "msft", "date": "2024-01-10"},  # lowercase should be normalized
        ]

        result = get_earnings_exclusions(client, WEEK_START, WEEK_END)

        assert result == {"AAPL", "MSFT"}

    def test_passes_week_range_to_client(self):
        client = MagicMock()
        client.get_earnings_calendar.return_value = []

        get_earnings_exclusions(client, WEEK_START, WEEK_END)

        client.get_earnings_calendar.assert_called_once_with("2024-01-08", "2024-01-12", symbol="")

    def test_returns_empty_set_when_client_is_none(self):
        assert get_earnings_exclusions(None, WEEK_START, WEEK_END) == set()

    def test_ignores_entries_without_symbol(self):
        client = MagicMock()
        client.get_earnings_calendar.return_value = [{"date": "2024-01-09"}]

        result = get_earnings_exclusions(client, WEEK_START, WEEK_END)

        assert result == set()
