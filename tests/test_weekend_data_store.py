"""Tests for weekend_swing.data_store.IntradayBarStore incremental gap-filling."""

import pandas as pd
import pytest

from signal_engine.weekend_swing.data_store import IntradayBarStore


class FakeAlpacaClient:
    """Records every fetch call and returns one synthetic daily bar per day in range."""

    def __init__(self):
        self.calls = []

    def get_historical_bars(self, symbols, start, end, timeframe_str="1Day"):
        self.calls.append((tuple(symbols), start, end, timeframe_str))
        ticker = symbols[0]
        dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
        df = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
            index=dates,
        )
        return {ticker: df}


class TestIntradayBarStore:
    def test_first_call_fetches_full_range_and_persists(self, tmp_path):
        client = FakeAlpacaClient()
        store = IntradayBarStore(client, store_dir=tmp_path)

        bars = store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))

        assert len(client.calls) == 1
        assert not bars.empty
        assert (tmp_path / "AAPL_1Day.parquet").exists()

    def test_second_call_same_range_makes_zero_api_calls(self, tmp_path):
        client = FakeAlpacaClient()
        store = IntradayBarStore(client, store_dir=tmp_path)

        store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
        assert len(client.calls) == 1

        bars_second = store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))

        assert len(client.calls) == 1  # no new fetch
        assert not bars_second.empty

    def test_extending_end_only_fetches_the_new_edge(self, tmp_path):
        client = FakeAlpacaClient()
        store = IntradayBarStore(client, store_dir=tmp_path)

        store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
        assert len(client.calls) == 1

        bars = store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-20"))

        assert len(client.calls) == 2
        # The second fetch should only cover the missing tail, not re-fetch the whole range
        _, fetch_start, fetch_end, _ = client.calls[1]
        assert fetch_start >= pd.Timestamp("2024-01-10")
        assert bars.index.max() >= pd.Timestamp("2024-01-20")

    def test_extending_start_only_fetches_the_new_edge(self, tmp_path):
        client = FakeAlpacaClient()
        store = IntradayBarStore(client, store_dir=tmp_path)

        store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-20"))
        assert len(client.calls) == 1

        bars = store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-20"))

        assert len(client.calls) == 2
        _, fetch_start, fetch_end, _ = client.calls[1]
        assert fetch_end <= pd.Timestamp("2024-01-10")
        assert bars.index.min() <= pd.Timestamp("2024-01-01")

    def test_separate_tickers_and_timeframes_are_independent(self, tmp_path):
        client = FakeAlpacaClient()
        store = IntradayBarStore(client, store_dir=tmp_path)

        store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
        store.get_bars("AAPL", "5Min", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
        store.get_bars("MSFT", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))

        assert len(client.calls) == 3
        assert (tmp_path / "AAPL_1Day.parquet").exists()
        assert (tmp_path / "AAPL_5Min.parquet").exists()
        assert (tmp_path / "MSFT_1Day.parquet").exists()

    def test_returns_only_the_requested_slice(self, tmp_path):
        client = FakeAlpacaClient()
        store = IntradayBarStore(client, store_dir=tmp_path)

        store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-20"))
        narrow = store.get_bars("AAPL", "1Day", pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-10"))

        assert narrow.index.min() >= pd.Timestamp("2024-01-05")
        assert narrow.index.max() <= pd.Timestamp("2024-01-10")
        assert len(client.calls) == 1  # fully covered by the first fetch, no new call
