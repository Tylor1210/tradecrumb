"""Tests for TradeTracker outcome math, especially SHORT sign inversion."""

from datetime import datetime

import pytest

from signal_engine.paper_trading.tracker import TradeTracker


def _open_trade(tracker, ticker, entry_price, recommendation):
    return tracker.log_entry(
        ticker=ticker,
        signal_date=datetime.now(),
        entry_date=datetime.now(),
        entry_price=entry_price,
        qty=10,
        notional=entry_price * 10,
        recommendation=recommendation,
        composite_score=0.8 if recommendation == "BUY" else -0.8,
        signal_breakdown={},
        hold_period="intraday",
    )


class TestLogExit:
    def test_long_profit_when_price_rises(self, tmp_path):
        tracker = TradeTracker(db_path=tmp_path / "trades.db")
        trade_id = _open_trade(tracker, "ABC", 100.0, "BUY")

        outcome = tracker.log_exit(trade_id, datetime.now(), exit_price=110.0)

        assert outcome == pytest.approx(10.0)

    def test_long_loss_when_price_falls(self, tmp_path):
        tracker = TradeTracker(db_path=tmp_path / "trades.db")
        trade_id = _open_trade(tracker, "ABC", 100.0, "BUY")

        outcome = tracker.log_exit(trade_id, datetime.now(), exit_price=90.0)

        assert outcome == pytest.approx(-10.0)

    def test_short_profit_when_price_falls(self, tmp_path):
        """A SHORT wins when the price drops — outcome_pct must be inverted."""
        tracker = TradeTracker(db_path=tmp_path / "trades.db")
        trade_id = _open_trade(tracker, "XYZ", 100.0, "SHORT")

        outcome = tracker.log_exit(trade_id, datetime.now(), exit_price=90.0)

        assert outcome == pytest.approx(10.0)

    def test_short_loss_when_price_rises(self, tmp_path):
        tracker = TradeTracker(db_path=tmp_path / "trades.db")
        trade_id = _open_trade(tracker, "XYZ", 100.0, "SHORT")

        outcome = tracker.log_exit(trade_id, datetime.now(), exit_price=110.0)

        assert outcome == pytest.approx(-10.0)

    def test_unknown_trade_id_returns_none(self, tmp_path):
        tracker = TradeTracker(db_path=tmp_path / "trades.db")
        assert tracker.log_exit("does-not-exist", datetime.now(), exit_price=100.0) is None
