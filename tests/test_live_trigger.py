"""
Tests for LiveDayTradeTrigger entry sizing logic (signal_engine.paper_trading.live_trigger).

Broker and tracker are mocked so these run with no network calls and no
real Alpaca credentials — they verify the BUY/SHORT decision + sizing math,
not the SDK integration itself (that's covered by manual testing against
the live paper account).
"""

from datetime import datetime
from unittest.mock import MagicMock

from signal_engine.paper_trading.live_trigger import LiveDayTradeTrigger
from signal_engine.signals.composite import SignalReport


def _report(ticker: str, score: float) -> SignalReport:
    return SignalReport(
        ticker=ticker,
        date=datetime.now(),
        composite_score=score,
        recommendation="BUY" if score > 0 else "SHORT",
        confidence="MEDIUM",
    )


def _make_trigger(pt_config: dict = None) -> LiveDayTradeTrigger:
    """Construct without running __init__ so no real broker/tracker/db is touched."""
    pt_config = pt_config or {}
    trigger = LiveDayTradeTrigger.__new__(LiveDayTradeTrigger)
    trigger.pt_config = pt_config
    trigger.sig_config = {}
    trigger.wl_config = {}
    trigger.trade_size = float(pt_config.get("fixed_trade_size", 1000.0))
    trigger.execution_threshold = float(pt_config.get("execution_threshold", 0.55))
    trigger.max_positions = int(pt_config.get("max_positions", 10))
    trigger.tracker = MagicMock()
    trigger.broker = MagicMock()
    trigger.tracker.get_open_trades.return_value = []
    trigger.tracker.log_entry.return_value = "trade-id-1"
    return trigger


class TestEnterNewTradesBuy:
    def test_places_fixed_notional_buy(self, monkeypatch):
        trigger = _make_trigger()
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("AAPL", 0.7)], {"AAPL": 185.0}))
        trigger.broker.submit_market_buy.return_value = {"id": "order-1", "filled_avg_price": 185.0, "qty": 5.405}

        entered = trigger.enter_new_trades(datetime.now())

        assert len(entered) == 1
        assert entered[0]["side"] == "BUY"
        trigger.broker.submit_market_buy.assert_called_once_with("AAPL", notional=1000.0)
        trigger.broker.submit_market_short.assert_not_called()


class TestEnterNewTradesShort:
    def test_places_qty_based_short(self, monkeypatch):
        trigger = _make_trigger()
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("XYZ", -0.7)], {"XYZ": 50.0}))
        trigger.broker.submit_market_short.return_value = {"id": "order-2", "filled_avg_price": None, "qty": None}

        entered = trigger.enter_new_trades(datetime.now())

        assert len(entered) == 1
        assert entered[0]["side"] == "SHORT"
        # $1000 trade size / $50 share price = 20 whole shares (shorts can't use notional orders)
        trigger.broker.submit_market_short.assert_called_once_with("XYZ", qty=20)

        log_kwargs = trigger.tracker.log_entry.call_args.kwargs
        assert log_kwargs["recommendation"] == "SHORT"
        assert log_kwargs["entry_price"] == 50.0  # falls back to reference price when fill price is None

    def test_skips_short_when_trade_size_cant_buy_one_share(self, monkeypatch):
        trigger = _make_trigger()
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("EXP", -0.9)], {"EXP": 5000.0}))

        entered = trigger.enter_new_trades(datetime.now())

        assert entered == []
        trigger.broker.submit_market_short.assert_not_called()

    def test_skips_short_when_no_reference_price(self, monkeypatch):
        trigger = _make_trigger()
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("NOPRICE", -0.9)], {}))

        entered = trigger.enter_new_trades(datetime.now())

        assert entered == []
        trigger.broker.submit_market_short.assert_not_called()


class TestEnterNewTradesLimits:
    def test_respects_open_position_slots(self, monkeypatch):
        trigger = _make_trigger({"max_positions": 1})
        trigger.tracker.get_open_trades.return_value = [{"ticker": "ALREADY_OPEN"}]
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("AAPL", 0.9)], {"AAPL": 100.0}))

        entered = trigger.enter_new_trades(datetime.now())

        assert entered == []
        trigger.broker.submit_market_buy.assert_not_called()

    def test_skips_tickers_already_open(self, monkeypatch):
        trigger = _make_trigger()
        trigger.tracker.get_open_trades.return_value = [{"ticker": "AAPL"}]
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("AAPL", 0.9)], {"AAPL": 100.0}))

        entered = trigger.enter_new_trades(datetime.now())

        assert entered == []
        trigger.broker.submit_market_buy.assert_not_called()

    def test_below_threshold_signals_are_ignored(self, monkeypatch):
        trigger = _make_trigger()
        monkeypatch.setattr(trigger, "_compute_signals", lambda now: ([_report("FLAT", 0.1)], {"FLAT": 100.0}))

        entered = trigger.enter_new_trades(datetime.now())

        assert entered == []
