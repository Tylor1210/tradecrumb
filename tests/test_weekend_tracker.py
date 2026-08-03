"""Tests for weekend_swing.tracker.WeekendSwingTracker."""

from datetime import datetime

import pandas as pd
import pytest

from signal_engine.weekend_swing.tracker import WeekendSwingTracker
from signal_engine.weekend_swing.weekend_swing_signals import WeekendSwingSignal


def _signal(ticker="AAPL", strategy="prior_week_breakout", direction="long", gap_type=None) -> WeekendSwingSignal:
    return WeekendSwingSignal(
        ticker=ticker, date=pd.Timestamp("2024-01-08 09:30"), strategy=strategy, direction=direction,
        score=0.7, entry_price=100.0, stop_price=95.0, pivot_high=101.0, pivot_low=99.0,
        breakdown={"gap_type": gap_type} if gap_type else {},
    )


class TestScanResults:
    def test_save_and_get_scan_results(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        candidates = [
            {"ticker": "AAPL", "strategy": "prior_week_breakout", "pivot_high": 101, "pivot_low": 99,
             "current_price": 100, "distance_pct": 1.0, "earnings_excluded": False, "data_quality": ""},
            {"ticker": "SPY", "strategy": "gap_strength", "pivot_high": 500, "pivot_low": 490,
             "current_price": 495, "distance_pct": 0.5, "earnings_excluded": True, "data_quality": "proxy"},
        ]
        tracker.save_scan_results("2024-01-08", candidates)

        results = tracker.get_scan_results("2024-01-08")

        assert len(results) == 2
        assert results[0]["ticker"] == "SPY"  # sorted by distance_pct ascending
        assert results[0]["earnings_excluded"] == 1

    def test_get_latest_week_when_unspecified(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        tracker.save_scan_results("2024-01-08", [{"ticker": "A", "strategy": "s", "pivot_high": 1,
                                                    "pivot_low": 1, "current_price": 1, "distance_pct": 1}])
        tracker.save_scan_results("2024-01-15", [{"ticker": "B", "strategy": "s", "pivot_high": 1,
                                                    "pivot_low": 1, "current_price": 1, "distance_pct": 1}])
        results = tracker.get_scan_results()
        assert len(results) == 1
        assert results[0]["ticker"] == "B"

    def test_save_replaces_prior_snapshot_for_same_week(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        tracker.save_scan_results("2024-01-08", [{"ticker": "A", "strategy": "s", "pivot_high": 1,
                                                    "pivot_low": 1, "current_price": 1, "distance_pct": 1}])
        tracker.save_scan_results("2024-01-08", [{"ticker": "B", "strategy": "s", "pivot_high": 1,
                                                    "pivot_low": 1, "current_price": 1, "distance_pct": 1}])
        results = tracker.get_scan_results("2024-01-08")
        assert len(results) == 1
        assert results[0]["ticker"] == "B"

    def test_empty_store_returns_empty_list(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        assert tracker.get_scan_results() == []


class TestSignalLifecycle:
    def test_log_triggered_signal(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        signal_id = tracker.log_triggered_signal(_signal(), "2024-01-08")

        signals = tracker.get_signals(status="TRIGGERED")
        assert len(signals) == 1
        assert signals[0]["signal_id"] == signal_id
        assert signals[0]["ticker"] == "AAPL"

    def test_logging_same_trigger_twice_is_idempotent(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        id1 = tracker.log_triggered_signal(_signal(), "2024-01-08")
        id2 = tracker.log_triggered_signal(_signal(), "2024-01-08")

        assert id1 == id2
        assert len(tracker.get_signals()) == 1

    def test_different_gap_type_is_a_distinct_signal(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        tracker.log_triggered_signal(_signal(ticker="SPY", strategy="gap_strength", gap_type="gap_and_go"), "2024-01-08")
        tracker.log_triggered_signal(_signal(ticker="SPY", strategy="gap_strength", gap_type="gap_and_fail"), "2024-01-08")

        assert len(tracker.get_signals()) == 2

    def test_submit_moves_triggered_to_submitted(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        signal_id = tracker.log_triggered_signal(_signal(), "2024-01-08")

        ok = tracker.submit_signal(signal_id, qty=10, notional=1000.0, broker_trade_id="order-1", entry_price=100.5)

        assert ok is True
        assert tracker.get_signals(status="TRIGGERED") == []
        submitted = tracker.get_signals(status="SUBMITTED")
        assert len(submitted) == 1
        assert submitted[0]["qty"] == 10

    def test_cannot_submit_twice(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        signal_id = tracker.log_triggered_signal(_signal(), "2024-01-08")
        tracker.submit_signal(signal_id, qty=10, notional=1000.0, broker_trade_id="order-1", entry_price=100.5)

        ok_again = tracker.submit_signal(signal_id, qty=10, notional=1000.0, broker_trade_id="order-2", entry_price=101.0)

        assert ok_again is False

    def test_close_long_profit_when_price_rises(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        signal_id = tracker.log_triggered_signal(_signal(direction="long"), "2024-01-08")
        tracker.submit_signal(signal_id, qty=10, notional=1000.0, broker_trade_id="order-1", entry_price=100.0)

        outcome = tracker.close_signal(signal_id, exit_price=110.0, exit_reason="friday_close")

        assert outcome == pytest.approx(10.0)
        assert tracker.get_signals(status="CLOSED")[0]["exit_reason"] == "friday_close"

    def test_close_short_profit_when_price_falls(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        signal_id = tracker.log_triggered_signal(_signal(direction="short"), "2024-01-08")
        tracker.submit_signal(signal_id, qty=10, notional=1000.0, broker_trade_id="order-1", entry_price=100.0)

        outcome = tracker.close_signal(signal_id, exit_price=90.0, exit_reason="stopped_out")

        assert outcome == pytest.approx(10.0)

    def test_expire_marks_triggered_as_expired(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        signal_id = tracker.log_triggered_signal(_signal(), "2024-01-08")

        assert tracker.expire_signal(signal_id) is True
        assert tracker.get_signals(status="TRIGGERED") == []
        assert tracker.get_signals(status="EXPIRED")[0]["signal_id"] == signal_id


class TestHistoryAndSummary:
    def _closed_trade(self, tracker, ticker, strategy, direction, entry, exit_price, week_start="2024-01-08"):
        signal_id = tracker.log_triggered_signal(_signal(ticker=ticker, strategy=strategy, direction=direction), week_start)
        tracker.submit_signal(signal_id, qty=10, notional=1000.0, broker_trade_id=f"o-{signal_id[:6]}", entry_price=entry)
        tracker.close_signal(signal_id, exit_price=exit_price, exit_reason="friday_close")
        return signal_id

    def test_history_filters_by_strategy(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        self._closed_trade(tracker, "AAPL", "prior_week_breakout", "long", 100, 110)
        self._closed_trade(tracker, "SPY", "gap_strength", "long", 500, 510)

        history = tracker.get_history(strategy="gap_strength")
        assert len(history) == 1
        assert history[0]["ticker"] == "SPY"

    def test_summary_combined_and_by_strategy(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        self._closed_trade(tracker, "AAPL", "prior_week_breakout", "long", 100, 110)   # +10%
        self._closed_trade(tracker, "MSFT", "prior_week_breakout", "long", 100, 90)    # -10%
        self._closed_trade(tracker, "SPY", "gap_strength", "long", 500, 550)           # +10%

        summary = tracker.get_summary_stats()

        assert summary["combined"]["total_trades"] == 3
        assert summary["combined"]["win_rate_pct"] == pytest.approx(66.67, abs=0.01)
        assert summary["by_strategy"]["prior_week_breakout"]["total_trades"] == 2
        assert summary["by_strategy"]["gap_strength"]["total_trades"] == 1
        assert summary["by_strategy"]["gap_strength"]["win_rate_pct"] == 100.0

    def test_summary_combined_reflects_strategy_filter(self, tmp_path):
        """Regression test: the summary cards must match whatever the trade
        table is filtered to, not always show every strategy combined."""
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        self._closed_trade(tracker, "AAPL", "prior_week_breakout", "long", 100, 110)   # +10%
        self._closed_trade(tracker, "MSFT", "prior_week_breakout", "long", 100, 90)    # -10%
        self._closed_trade(tracker, "SPY", "gap_strength", "long", 500, 550)           # +10%

        summary = tracker.get_summary_stats(strategy="gap_strength")

        assert summary["combined"]["total_trades"] == 1
        assert summary["combined"]["win_rate_pct"] == 100.0

    def test_summary_with_no_closed_trades(self, tmp_path):
        tracker = WeekendSwingTracker(db_path=tmp_path / "t.db")
        summary = tracker.get_summary_stats()
        assert summary["combined"]["total_trades"] == 0
        assert summary["combined"]["win_rate_pct"] is None
