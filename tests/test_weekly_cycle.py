"""Weekly cycle: phase selection, week helpers, tracker migration/skip, review what-if + findings."""

import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from signal_engine.weekend_swing import weekly_cycle
from signal_engine.weekend_swing.review import _findings
from signal_engine.weekend_swing.tracker import WeekendSwingTracker
from signal_engine.weekend_swing.weekly_cycle import auto_phases, current_week_start, previous_week_start


def et(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def test_week_helpers():
    assert current_week_start(et("2026-11-21 10:00")) == pd.Timestamp("2026-11-23", tz="UTC")  # Saturday -> upcoming Monday
    assert current_week_start(et("2026-11-24 10:00")) == pd.Timestamp("2026-11-23", tz="UTC")  # Tuesday
    assert previous_week_start(et("2026-11-21 10:00")) == pd.Timestamp("2026-11-16", tz="UTC")  # Saturday -> week just ended
    assert previous_week_start(et("2026-11-20 17:00")) == pd.Timestamp("2026-11-16", tz="UTC")


def test_auto_phases():
    assert auto_phases(et("2026-11-21 10:00")) == ["scan", "review"]       # Saturday
    assert auto_phases(et("2026-11-22 10:00")) == ["scan"]                 # Sunday
    assert auto_phases(et("2026-11-23 10:00")) == ["trigger", "manage"]    # Monday session
    assert auto_phases(et("2026-11-24 10:00")) == ["manage"]               # Tuesday session
    assert auto_phases(et("2026-11-24 20:00")) == []                       # after hours
    assert auto_phases(et("2026-11-27 16:30")) == ["review"]               # Friday after close


def _signal(ticker="AAPL", direction="long"):
    return SimpleNamespace(ticker=ticker, strategy="prior_week_breakout", direction=direction, entry_price=100.0,
                           stop_price=98.0, date=pd.Timestamp("2026-11-23 10:00"), breakdown={})


def test_tracker_stores_regime_and_skip(tmp_path):
    t = WeekendSwingTracker(tmp_path / "w.db")
    sid = t.log_triggered_signal(_signal(), "2026-11-23", regime={"pressure": 0.3, "fear_greed": 40})
    assert "regime_json" in t.get_signal(sid) and '"pressure": 0.3' in t.get_signal(sid)["regime_json"]
    assert t.skip_signal(sid, "regime policy: test")
    row = t.get_signal(sid)
    assert row["status"] == "SKIPPED" and row["skip_reason"] == "regime policy: test"
    assert not t.skip_signal(sid, "again")  # only TRIGGERED can be skipped
    assert [r["signal_id"] for r in t.get_week_signals("2026-11-23")] == [sid]


def test_tracker_migrates_old_db(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:  # the pre-regime schema
        c.execute("CREATE TABLE weekend_signals (id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT UNIQUE NOT NULL, "
                  "week_start TEXT NOT NULL, ticker TEXT NOT NULL, strategy TEXT NOT NULL, gap_type TEXT, direction TEXT NOT NULL, "
                  "trigger_price REAL NOT NULL, stop_price REAL NOT NULL, triggered_at TEXT NOT NULL, breakdown_json TEXT, "
                  "status TEXT DEFAULT 'TRIGGERED', qty REAL, notional REAL, broker_trade_id TEXT, entry_price REAL, "
                  "submitted_at TEXT, exit_date TEXT, exit_price REAL, exit_reason TEXT, outcome_pct REAL, closed_at TEXT, "
                  "created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    t = WeekendSwingTracker(path)
    assert t.log_triggered_signal(_signal(), "2026-11-23")


def _rv(**kw):
    base = {"n_scan_candidates": 5, "signals": {"SUBMITTED": 0, "CLOSED": 6}, "skipped_whatif": {"n": 0, "ev_pct": None},
            "taken": {"n": 6, "ev_pct": -1.0}, "missed_movers": []}
    base.update(kw)
    return base


def _closed(exit_reason, out, recovered=None, mfe=0.0):
    return {"exit_reason": exit_reason, "outcome_pct": out, "excursion": {"recovered_after_stop": recovered, "mfe_pct": mfe}}


def test_findings_flag_tight_stops_and_giveback_and_negative_strategy():
    closed = [_closed("stopped_out", -2, recovered=True, mfe=2.5)] * 5 + [_closed("friday_close", 1)]
    f = _findings(_rv(), closed, [], {"prior_week_breakout": {"n": 12, "ev_pct": -0.7}, "ALL": {"n": 12, "ev_pct": -0.7}})
    text = " ".join(f)
    assert "stops look too tight" in text and "up 2%+" in text and "consider removing" in text


def test_findings_small_sample_does_not_recommend_disabling():
    f = _findings(_rv(), [], [], {"prior_week_breakout": {"n": 3, "ev_pct": -2.0}, "ALL": {"n": 3, "ev_pct": -2.0}})
    assert not any("consider removing" in x for x in f)


def test_findings_missed_movers_and_skip_cost():
    rv = _rv(missed_movers=[{"ticker": "NVDA", "move_pct": 9.0, "was_shortlisted": False}],
             skipped_whatif={"n": 4, "ev_pct": 1.5})
    f = " ".join(_findings(rv, [], [], {"ALL": {"n": 0, "ev_pct": None}}))
    assert "NVDA" in f and "skip rules cost money" in f


# ── Late-trigger guard: Monday signals must never be ordered at later prices ──

def _cycle(tmp_path, monkeypatch, signal_time="2026-11-23 10:00"):
    """A cycle holding one un-submitted signal for the week of Nov 23; records orders instead of sending them."""
    monkeypatch.setattr(weekly_cycle, "RegimeStore", lambda: None)
    tracker = WeekendSwingTracker(tmp_path / "w.db")
    tracker.save_scan_results("2026-11-23", [{
        "ticker": "AAPL", "strategy": "prior_week_breakout", "pivot_high": 101.0, "pivot_low": 95.0,
        "current_price": 100.0, "distance_pct": 1.0,
    }])
    signal = _signal()
    signal.date = et(signal_time)  # bar timestamps are tz-aware UTC
    sid = tracker.log_triggered_signal(signal, "2026-11-23")
    cyc = weekly_cycle.WeeklyCycle({}, {}, None, None, None, tracker, log=lambda msg: None)
    cyc.regime_snapshot = lambda *a, **k: None
    orders = []
    cyc._submit = lambda s, notional, mult: orders.append(s["ticker"]) or True
    return cyc, tracker, sid, orders


@pytest.mark.parametrize("when", [
    "2026-11-24 10:00",  # Tuesday
    "2026-11-27 10:00",  # Friday
    "2026-11-23 09:00",  # Monday before the open
    "2026-11-23 16:15",  # Monday after the close (a DAY order would queue for Tuesday)
])
def test_trigger_outside_monday_session_sends_no_orders(tmp_path, monkeypatch, when):
    cyc, tracker, sid, orders = _cycle(tmp_path, monkeypatch)
    assert cyc.trigger(et(when)) == {"new": 0}
    assert orders == []
    assert tracker.get_signal(sid)["status"] == "TRIGGERED"  # left for the review's what-if scoring


def test_trigger_skips_stale_signal_on_monday(tmp_path, monkeypatch):
    cyc, tracker, sid, orders = _cycle(tmp_path, monkeypatch)  # fired 10:00, checked 15:00
    cyc.trigger(et("2026-11-23 15:00"))
    row = tracker.get_signal(sid)
    assert orders == [] and row["status"] == "SKIPPED" and row["skip_reason"].startswith("stale")


def test_trigger_submits_fresh_signal_on_monday(tmp_path, monkeypatch):
    cyc, _, _, orders = _cycle(tmp_path, monkeypatch)
    cyc.trigger(et("2026-11-23 10:30"))  # 15-min bar stamped 10:00 closed at 10:15
    assert orders == ["AAPL"]
