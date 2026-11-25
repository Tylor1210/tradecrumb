"""
Weekly paper-trading cycle: the automation that runs the weekend-swing lane
end to end, every week, on the Alpaca PAPER account only.

  scan     (Sat/Sun)      regime snapshot + weekend shortlist saved
  trigger  (Mon, polled)  check shortlist for triggers, apply regime policy, auto-submit paper orders
  manage   (Mon-Fri, polled) enforce stops, close everything at Friday 15:45 ET, expire stale signals
  review   (Fri eve/Sat)  weekly review report (what worked, what we skipped, what we're missing)

Called by: cli.py (`weekly-cycle`), meant to be run from cron (see CRON_LINES).
Depends on: monitor, tracker, review, market_regime, paper_trading.broker.

CONSTRAINT: paper only. The broker class hardcodes paper=True; nothing here
may construct any other trading client.

WHY stops are enforced here: Alpaca market orders are sent without a stop leg
and the old flow never watched the stop, so a "stopped out" trade just kept
running. `manage` polls and closes at market when the stop is breached, so the
live paper results match what the backtest assumes (stop-or-Friday-close).
GOTCHA: polling means the fill can be worse than the stop price; the recorded
exit is the broker's fill, not the stop level.
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from signal_engine.market_regime.policy import RegimePolicy
from signal_engine.market_regime.provider import RegimeProvider
from signal_engine.market_regime.store import RegimeStore
from signal_engine.weekend_swing.monitor import check_candidate_trigger

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

CRON_LINES = """\
# signal-engine weekly paper-trading cycle (times are machine-local; assumes US Eastern)
# Replace /path/to/tradecrumb with the repo root.
0 10 * * 6     cd /path/to/tradecrumb && uv run signal-engine weekly-cycle --phase scan   >> data/weekly_cycle.log 2>&1
*/15 9-15 * * 1   cd /path/to/tradecrumb && uv run signal-engine weekly-cycle --phase trigger >> data/weekly_cycle.log 2>&1
*/15 9-15 * * 1-5 cd /path/to/tradecrumb && uv run signal-engine weekly-cycle --phase manage  >> data/weekly_cycle.log 2>&1
30 16 * * 5    cd /path/to/tradecrumb && uv run signal-engine weekly-cycle --phase review  >> data/weekly_cycle.log 2>&1
"""

DEFAULT_CYCLE_CFG = {
    "auto_submit": True,
    # Strategy 2 is excluded by default: 95% of its 107 backtested trades were stopped out
    # (stop = far edge of a tiny range). Its signals are still logged and scored as what-ifs.
    "auto_submit_strategies": ["prior_week_breakout", "gap_strength"],
    "max_new_positions_per_week": 10,
    "friday_close_time_et": "15:45",
}


def current_week_start(now: pd.Timestamp) -> pd.Timestamp:
    """Monday (UTC midnight) of the week containing `now`; weekend dates map to the *upcoming* Monday."""
    d = now.tz_convert(ET).normalize().tz_localize(None)
    if d.weekday() >= 5:
        d += pd.Timedelta(days=7 - d.weekday())
    else:
        d -= pd.Timedelta(days=d.weekday())
    return pd.Timestamp(d.date(), tz="UTC")


def previous_week_start(now: pd.Timestamp) -> pd.Timestamp:
    """Monday of the week that is ending/just ended (Sat/Sun -> the Monday just gone)."""
    d = now.tz_convert(ET).normalize().tz_localize(None)
    d -= pd.Timedelta(days=d.weekday())
    return pd.Timestamp(d.date(), tz="UTC")


def auto_phases(now: pd.Timestamp) -> list:
    """Which phases make sense right now (ET). Idempotent, so over-selecting is harmless."""
    et = now.tz_convert(ET)
    wd, hm = et.weekday(), et.hour * 60 + et.minute
    market_open = 9 * 60 + 30 <= hm <= 16 * 60
    if wd >= 5:
        return ["scan"] + (["review"] if wd == 5 else [])
    phases = []
    if wd == 0 and market_open:
        phases.append("trigger")
    if market_open:
        phases.append("manage")
    if wd == 4 and hm > 16 * 60:
        phases.append("review")
    return phases


class WeeklyCycle:
    def __init__(self, ws_cfg, rg_cfg, alpaca, finnhub, bar_store, tracker, dry_run=False, log=print) -> None:
        self.ws_cfg, self.rg_cfg = ws_cfg, rg_cfg
        self.alpaca, self.finnhub, self.bar_store = alpaca, finnhub, bar_store
        self.tracker = tracker
        self.dry_run = dry_run
        self.log = log
        self.cfg = {**DEFAULT_CYCLE_CFG, **ws_cfg.get("weekly_cycle", {})}
        self.policy = RegimePolicy(rg_cfg)
        self.regime_store = RegimeStore()
        self._broker = None

    @property
    def broker(self):
        if self._broker is None:
            from signal_engine.paper_trading.broker import PaperTradingBroker
            self._broker = PaperTradingBroker()
        return self._broker

    # ── regime ──────────────────────────────────────────────────────────────

    def regime_snapshot(self, when: pd.Timestamp, breadth: Optional[list] = None):
        """Regime using only bars before `when`, saved to the regime history."""
        from signal_engine.cli import _cached_universe  # same breadth universe as `regime` / the study

        provider = RegimeProvider(self.rg_cfg, self.bar_store, breadth or _cached_universe(self.ws_cfg))
        regime = provider.as_of(when)
        if regime is not None:
            self.regime_store.save(regime)
        return regime

    # ── phases ──────────────────────────────────────────────────────────────

    def scan(self, now: pd.Timestamp) -> dict:
        from signal_engine.weekend_swing.scan import build_weekend_watchlist

        week_start = current_week_start(now)
        regime = self.regime_snapshot(now + pd.Timedelta(days=1))
        wl = build_weekend_watchlist(self.ws_cfg, self.alpaca, self.finnhub, self.bar_store, week_start)
        self.tracker.save_scan_results(week_start.date().isoformat(), wl["candidates"])
        self.log(f"scan: week {week_start.date()} — {len(wl['candidates'])} candidates from {wl['universe_size']} tickers")
        if regime:
            for line in regime.summary:
                self.log(f"  regime: {line}")
        return {"candidates": len(wl["candidates"])}

    def trigger(self, now: pd.Timestamp) -> dict:
        from signal_engine.weekend_swing.weekend_swing_signals import (
            GapStrengthStrategy, OvernightRangeBreakoutStrategy, PriorWeekBreakoutStrategy,
        )

        week_start = current_week_start(now)
        ws_str = week_start.date().isoformat()
        if now < week_start:
            self.log("trigger: before Monday — nothing to check")
            return {"new": 0}

        candidates = self.tracker.get_scan_results(ws_str)
        if not candidates:
            self.log("trigger: no saved scan for this week — run `weekly-cycle --phase scan` first")
            return {"new": 0}

        from signal_engine.weekend_swing.earnings import get_earnings_exclusions
        earnings = get_earnings_exclusions(self.finnhub, week_start, week_start + pd.Timedelta(days=4))
        strategies = {
            "prior_week_breakout": PriorWeekBreakoutStrategy(self.ws_cfg),
            "overnight_range_breakout": OvernightRangeBreakoutStrategy(self.ws_cfg),
            "gap_strength": GapStrengthStrategy(self.ws_cfg),
        }
        existing = {(s["ticker"], s["strategy"]) for s in self.tracker.get_week_signals(ws_str)}
        regime = self.regime_snapshot(week_start)  # point-in-time: only bars before Monday
        regime_dict = regime.to_dict() if regime else None

        new_ids = []
        for c in candidates:
            if (c["ticker"], c["strategy"]) in existing:
                continue
            try:
                signal = check_candidate_trigger(c, strategies, self.bar_store, week_start, earnings, now)
            except Exception as exc:
                logger.warning("trigger check failed for %s/%s: %s", c["ticker"], c["strategy"], exc)
                continue
            if signal is None:
                continue
            sid = self.tracker.log_triggered_signal(signal, ws_str, regime=regime_dict)
            existing.add((c["ticker"], c["strategy"]))
            new_ids.append(sid)
            self.log(f"trigger: {signal.ticker} {signal.strategy} {signal.direction.upper()} "
                     f"@ {signal.entry_price:.2f} stop {signal.stop_price:.2f}")

        submitted = self._auto_submit(ws_str) if self.cfg["auto_submit"] else 0
        return {"new": len(new_ids), "submitted": submitted}

    def _auto_submit(self, ws_str: str) -> int:
        trade_size = self.ws_cfg.get("execution", {}).get("fixed_trade_size", 1000.0)
        week = self.tracker.get_week_signals(ws_str)
        taken = sum(1 for s in week if s["status"] in ("SUBMITTED", "CLOSED"))
        n = 0
        for s in week:
            if s["status"] != "TRIGGERED":
                continue
            if s["strategy"] not in self.cfg["auto_submit_strategies"]:
                self.tracker.skip_signal(s["signal_id"], "strategy not enabled for auto-submit")
                continue
            if taken >= self.cfg["max_new_positions_per_week"]:
                self.tracker.skip_signal(s["signal_id"], "weekly position cap reached")
                continue
            import json as _json
            decision = self.policy.evaluate(s["direction"], _json.loads(s["regime_json"]) if s.get("regime_json") else None)
            if not decision.allow:
                self.tracker.skip_signal(s["signal_id"], f"regime policy: {decision.reason}")
                self.log(f"skip: {s['ticker']} {s['direction']} — {decision.reason}")
                continue
            if self._submit(s, trade_size * decision.size_mult, decision.size_mult):
                taken += 1
                n += 1
        return n

    def _submit(self, s: dict, notional: float, size_mult: float) -> bool:
        price = s["trigger_price"]
        if self.dry_run:
            self.log(f"[dry-run] would submit {s['direction']} {s['ticker']} ${notional:.0f}")
            return False
        try:
            if s["direction"] == "long":
                order = self.broker.submit_market_buy(s["ticker"], notional=notional)
                entry = order.get("filled_avg_price") or price
                qty = order.get("qty") or (notional / entry if entry else 0)
            else:
                qty = int(notional // price) if price else 0
                if qty < 1:
                    self.tracker.skip_signal(s["signal_id"], f"${notional:.0f} too small for 1 share at ${price:.2f}")
                    return False
                order = self.broker.submit_market_short(s["ticker"], qty=qty)
                entry = order.get("filled_avg_price") or price
                qty = order.get("qty") or qty
            self.tracker.submit_signal(
                s["signal_id"], qty=float(qty or 0), notional=round(float(qty or 0) * float(entry or 0), 2),
                broker_trade_id=order.get("id", ""), entry_price=float(entry or 0), size_mult=size_mult,
            )
            self.log(f"submitted: {s['direction']} {s['ticker']} ${notional:.0f} (size x{size_mult})")
            return True
        except Exception as exc:
            logger.error("submit failed for %s: %s", s["ticker"], exc)
            self.log(f"submit FAILED {s['ticker']}: {exc}")
            return False

    def manage(self, now: pd.Timestamp) -> dict:
        """Close positions whose stop was breached; close all at Friday 15:45 ET; expire stale signals."""
        et = now.tz_convert(ET)
        hh, mm = (int(x) for x in self.cfg["friday_close_time_et"].split(":"))
        friday_close = et.weekday() == 4 and et.hour * 60 + et.minute >= hh * 60 + mm
        closed = 0
        for s in self.tracker.get_signals(status="SUBMITTED"):
            reason, px = None, None
            if friday_close:
                reason = "friday_close"
            else:
                reason, px = self._stop_breached(s, now)
            if reason is None:
                continue
            closed += self._close(s, reason, px)

        ws_str = current_week_start(now).date().isoformat()
        if et.weekday() == 4 and et.hour >= 16:
            for s in self.tracker.get_week_signals(ws_str):
                if s["status"] == "TRIGGERED":
                    self.tracker.expire_signal(s["signal_id"])
        if closed:
            self.log(f"manage: closed {closed} position(s)")
        return {"closed": closed}

    def _stop_breached(self, s: dict, now: pd.Timestamp):
        # submitted_at is stored naive in machine-local time (datetime.now()).
        start = pd.Timestamp(datetime.fromisoformat(s["submitted_at"]).astimezone(ZoneInfo("UTC")))
        bars = self.bar_store.get_bars(s["ticker"], "5Min", start, now)
        if bars.empty:
            return None, None
        if s["direction"] == "long" and float(bars["low"].min()) <= s["stop_price"]:
            return "stopped_out", None
        if s["direction"] == "short" and float(bars["high"].max()) >= s["stop_price"]:
            return "stopped_out", None
        return None, None

    def _close(self, s: dict, reason: str, px: Optional[float]) -> int:
        if self.dry_run:
            self.log(f"[dry-run] would close {s['ticker']} ({reason})")
            return 0
        try:
            order = self.broker.close_position(s["ticker"])
            exit_price = order.get("filled_avg_price")
            if not exit_price:
                latest = self.alpaca.get_latest_bar(s["ticker"])
                exit_price = latest["close"] if latest else s["entry_price"]
            out = self.tracker.close_signal(s["signal_id"], exit_price=float(exit_price), exit_reason=reason)
            self.log(f"closed: {s['ticker']} {reason} {out:+.2f}%")
            return 1
        except Exception as exc:
            # Typically "position not found": entry hasn't filled yet. Retry next poll.
            logger.warning("close failed for %s: %s", s["ticker"], exc)
            return 0

    def review(self, now: pd.Timestamp) -> dict:
        from signal_engine.weekend_swing.review import build_review, save_review

        week_start = previous_week_start(now)
        review = build_review(self.tracker, self.bar_store, self.regime_store, week_start, self.rg_cfg, self.ws_cfg)
        path = save_review(review)
        self.log(f"review: written to {path}")
        for f in review["findings"]:
            self.log(f"  - {f}")
        return {"path": str(path)}

    def run(self, phase: str, now: Optional[pd.Timestamp] = None) -> dict:
        now = now or pd.Timestamp.now(tz="UTC")
        phases = auto_phases(now) if phase == "auto" else [phase]
        if not phases:
            self.log("nothing to do at this time")
        return {p: getattr(self, p)(now) for p in phases}
