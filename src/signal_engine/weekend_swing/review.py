"""
Weekly review: what the paper trades did, what we skipped, and what we're
missing — written to reports/weekly/<monday>.md and .json.

Called by: weekly_cycle.WeeklyCycle.review, dashboard/server.py.
Depends on: tracker rows, IntradayBarStore, backtest._simulate_exit (to score
signals we did NOT trade as what-ifs), regime_study.

WHY what-if scoring: a loss you took teaches little by itself. Scoring the
signals we skipped (strategy disabled, regime policy, weekly cap) with the same
exit rules tells us whether each skip cost or saved money, which is the only
way to ever justify turning a filter on or off.

CONSTRAINT: findings are suggestions from small samples. MIN_TRADES gates
every "do X" finding so one bad week cannot flip a strategy off.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from signal_engine.market_regime.policy import RegimePolicy
from signal_engine.weekend_swing.backtest import _return_pct, _simulate_exit
from signal_engine.weekend_swing.levels import is_data_sane, monday_session_bars
from signal_engine.weekend_swing.regime_study import run_study

logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/weekly")
MIN_TRADES = 10


def _stats(rets: list) -> dict:
    if not rets:
        return {"n": 0, "win_rate_pct": None, "ev_pct": None, "total_pct": 0.0}
    return {"n": len(rets), "win_rate_pct": round(sum(r > 0 for r in rets) / len(rets) * 100, 1),
            "ev_pct": round(sum(rets) / len(rets), 3), "total_pct": round(sum(rets), 2)}


def _utc(ts: Optional[str]) -> Optional[pd.Timestamp]:
    if not ts:
        return None
    t = pd.Timestamp(datetime.fromisoformat(ts))
    return t.tz_convert("UTC") if t.tzinfo else pd.Timestamp(t.to_pydatetime().astimezone()).tz_convert("UTC")


def _excursions(row: dict, bar_store) -> Optional[dict]:
    """MFE/MAE in % from entry while the position was open, and whether a stopped-out trade recovered."""
    entry, start, end = row.get("entry_price"), _utc(row.get("submitted_at")), _utc(row.get("closed_at"))
    if not entry or start is None or end is None:
        return None
    bars = bar_store.get_bars(row["ticker"], "15Min", start, end)
    if bars.empty:
        return None
    sign = 1 if row["direction"] == "long" else -1
    fav = (bars["high"].max() if sign == 1 else bars["low"].min())
    adv = (bars["low"].min() if sign == 1 else bars["high"].max())
    mfe = max(0.0, sign * (float(fav) - entry) / entry * 100)
    mae = min(0.0, sign * (float(adv) - entry) / entry * 100)
    out = {"mfe_pct": round(mfe, 2), "mae_pct": round(mae, 2), "recovered_after_stop": None}
    if row.get("exit_reason") == "stopped_out":
        week_end = pd.Timestamp(row["week_start"], tz="UTC") + pd.Timedelta(days=5)
        later = bar_store.get_bars(row["ticker"], "1Day", end.normalize(), week_end)
        if not later.empty:
            out["recovered_after_stop"] = bool(sign * (float(later.iloc[-1]["close"]) - entry) > 0)
    return out


def _whatif(row: dict, bar_store) -> Optional[float]:
    """Return % the signal would have made had it been traded at trigger_price with the standard exits."""
    week_start = pd.Timestamp(row["week_start"], tz="UTC")
    week_end = week_start + pd.Timedelta(days=4)
    trig = pd.Timestamp(row["triggered_at"])
    trig = trig.tz_convert("UTC") if trig.tzinfo else trig.tz_localize("UTC")
    monday = monday_session_bars(
        bar_store.get_bars(row["ticker"], "5Min", week_start, week_start + pd.Timedelta(days=1)), week_start
    )
    daily = bar_store.get_bars(row["ticker"], "1Day", week_start - pd.Timedelta(days=5), week_end + pd.Timedelta(days=1))
    if monday.empty or daily.empty:
        return None
    exit_price, _, _ = _simulate_exit(row["direction"], row["trigger_price"], row["stop_price"], trig,
                                      monday, daily, week_start, week_end)
    return round(_return_pct(row["direction"], row["trigger_price"], exit_price), 3)


def _missed_movers(tickers: list, touched: set, week_start: pd.Timestamp, bar_store, max_move: float = 50.0, top: int = 5) -> list:
    out = []
    for t in tickers:
        try:
            # WHY the sanity check: unadjusted splits / bad prints (e.g. ZBAO) otherwise top this list every week.
            hist = bar_store.get_bars(t, "1Day", week_start - pd.Timedelta(days=90), week_start + pd.Timedelta(days=5))
            if not is_data_sane(hist, max_move):
                continue
            d = hist.loc[hist.index >= week_start]
        except Exception:
            continue
        if len(d) < 3:
            continue
        move = (float(d.iloc[-1]["close"]) / float(d.iloc[0]["open"]) - 1) * 100
        if abs(move) > max_move:
            continue  # same data-sanity reasoning as above
        out.append({"ticker": t, "move_pct": round(move, 2), "was_shortlisted": t in touched})
    out.sort(key=lambda x: abs(x["move_pct"]), reverse=True)
    return out[:top]


def build_review(tracker, bar_store, regime_store, week_start: pd.Timestamp, rg_cfg: dict, ws_cfg: dict) -> dict:
    ws = week_start.date().isoformat()
    rows = tracker.get_week_signals(ws)
    closed = [r for r in rows if r["status"] == "CLOSED"]
    for r in closed:
        r["excursion"] = _excursions(r, bar_store)

    shadow_rows = [r for r in rows if r["status"] in ("SKIPPED", "EXPIRED", "TRIGGERED")]
    for r in shadow_rows:
        r["whatif_pct"] = _whatif(r, bar_store)

    taken = [r["outcome_pct"] for r in closed if r["outcome_pct"] is not None]
    skipped_ret = [r["whatif_pct"] for r in shadow_rows if r.get("whatif_pct") is not None]

    by_strategy_week = {}
    for r in closed:
        by_strategy_week.setdefault(r["strategy"], []).append(r["outcome_pct"])

    all_closed = tracker.get_history()
    cum = {}
    for r in all_closed:
        cum.setdefault(r["strategy"], []).append(r["outcome_pct"])
    cumulative = {k: _stats(v) for k, v in cum.items()}
    cumulative["ALL"] = _stats([r["outcome_pct"] for r in all_closed])

    study_input = [
        {"direction": r["direction"], "return_pct": r["outcome_pct"], "entry_time": r["submitted_at"],
         "regime": json.loads(r["regime_json"]) if r.get("regime_json") else None}
        for r in all_closed if r.get("outcome_pct") is not None
    ]
    pol_cfg = {**rg_cfg, "policy": {**rg_cfg.get("policy", {}), "enabled": True}}
    study = run_study(study_input, RegimePolicy(pol_cfg)) if study_input else None

    touched = {r["ticker"] for r in rows} | {c["ticker"] for c in tracker.get_scan_results(ws)}
    from signal_engine.cli import _cached_universe
    movers = _missed_movers(_cached_universe(ws_cfg), touched, week_start, bar_store,
                            ws_cfg.get("universe", {}).get("max_single_day_move_pct", 50.0))

    snaps = regime_store.history(limit=3)
    review = {
        "week_start": ws,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signals": {s: sum(r["status"] == s for r in rows) for s in ("TRIGGERED", "SUBMITTED", "CLOSED", "SKIPPED", "EXPIRED")},
        "taken": _stats(taken),
        "taken_by_strategy": {k: _stats(v) for k, v in by_strategy_week.items()},
        "skipped_whatif": _stats(skipped_ret),
        "closed_trades": [
            {k: r.get(k) for k in ("ticker", "strategy", "direction", "entry_price", "exit_price",
                                    "exit_reason", "outcome_pct", "excursion")} for r in closed
        ],
        "skipped_trades": [
            {k: r.get(k) for k in ("ticker", "strategy", "direction", "status", "skip_reason", "whatif_pct")}
            for r in shadow_rows
        ],
        "missed_movers": movers,
        "cumulative": cumulative,
        "regime_study": study,
        "regime_history": snaps,
        "n_scan_candidates": len(tracker.get_scan_results(ws)),
    }
    review["findings"] = _findings(review, closed, shadow_rows, cumulative)
    review["markdown"] = _markdown(review)
    return review


def _findings(rv: dict, closed: list, shadow_rows: list, cumulative: dict) -> list:
    out = []
    if rv["n_scan_candidates"] == 0:
        out.append("No weekend scan was saved for this week — the scan phase did not run, so nothing could trigger.")
    elif not rv["signals"]["SUBMITTED"] and not rv["signals"]["CLOSED"]:
        out.append(f"{rv['n_scan_candidates']} shortlisted but no trades taken — check trigger/skip reasons below.")

    for strat, st in cumulative.items():
        if strat != "ALL" and st["n"] >= MIN_TRADES and st["ev_pct"] is not None and st["ev_pct"] < 0:
            out.append(f"{strat}: {st['n']} closed paper trades, EV {st['ev_pct']:+.2f}%/trade — "
                       "negative; consider removing it from auto_submit_strategies.")

    stops = [r for r in closed if r["exit_reason"] == "stopped_out"]
    if len(closed) >= 5 and len(stops) / len(closed) > 0.7:
        rec = [r for r in stops if (r.get("excursion") or {}).get("recovered_after_stop")]
        if len(stops) >= 5 and len(rec) / len(stops) >= 0.3:
            out.append(f"{len(stops)}/{len(closed)} trades stopped out and {len(rec)} of those were back in profit by "
                       "Friday — stops look too tight for this week's volatility.")
        else:
            out.append(f"{len(stops)}/{len(closed)} trades stopped out and most did not recover — entries were poor, not stops.")

    given_back = [r for r in closed if (r.get("excursion") or {}).get("mfe_pct", 0) >= 2 and (r["outcome_pct"] or 0) <= 0]
    if given_back:
        out.append(f"{len(given_back)} trade(s) were up 2%+ at some point and still closed at a loss — "
                   "a profit-protect/trailing exit is the likely missing piece.")

    sk = rv["skipped_whatif"]
    tk = rv["taken"]
    if sk["n"] >= 3 and sk["ev_pct"] is not None and tk["n"] and tk["ev_pct"] is not None and sk["ev_pct"] > tk["ev_pct"] + 0.5:
        out.append(f"Skipped signals (EV {sk['ev_pct']:+.2f}%) beat the ones we took (EV {tk['ev_pct']:+.2f}%) — "
                   "the skip rules cost money this week.")

    unseen = [m for m in rv["missed_movers"] if not m["was_shortlisted"] and abs(m["move_pct"]) >= 5]
    if unseen:
        out.append("Big movers never shortlisted: " + ", ".join(f"{m['ticker']} {m['move_pct']:+.1f}%" for m in unseen)
                   + " — the pivot-proximity filter or universe may be too narrow.")
    if not out:
        out.append("No actionable findings this week (sample too small or nothing stood out).")
    return out


def _markdown(rv: dict) -> str:
    L = [f"# Weekly review — week of {rv['week_start']}", ""]
    s = rv["signals"]
    L.append(f"Shortlisted {rv['n_scan_candidates']} | triggered signals: "
             + ", ".join(f"{k.lower()} {v}" for k, v in s.items() if v))
    t = rv["taken"]
    if t["n"]:
        L.append(f"Closed trades: {t['n']} | win rate {t['win_rate_pct']}% | EV {t['ev_pct']:+.2f}%/trade | total {t['total_pct']:+.2f}%")
    L += ["", "## Findings"] + [f"- {f}" for f in rv["findings"]]
    if rv["closed_trades"]:
        L += ["", "## Trades taken", "| Ticker | Strategy | Dir | Result % | Exit | MFE % | MAE % |", "|---|---|---|---|---|---|---|"]
        for c in rv["closed_trades"]:
            ex = c.get("excursion") or {}
            L.append(f"| {c['ticker']} | {c['strategy']} | {c['direction']} | {c['outcome_pct']:+.2f} | {c['exit_reason']} "
                     f"| {ex.get('mfe_pct', '')} | {ex.get('mae_pct', '')} |")
    if rv["skipped_trades"]:
        L += ["", "## Signals not traded (what-if)", "| Ticker | Strategy | Dir | Why | What-if % |", "|---|---|---|---|---|"]
        for c in rv["skipped_trades"]:
            w = c["whatif_pct"]
            L.append(f"| {c['ticker']} | {c['strategy']} | {c['direction']} | {c.get('skip_reason') or c['status']} "
                     f"| {'n/a' if w is None else f'{w:+.2f}'} |")
    if rv["missed_movers"]:
        L += ["", "## Biggest movers in the universe", "| Ticker | Move % | Shortlisted |", "|---|---|---|"]
        L += [f"| {m['ticker']} | {m['move_pct']:+.2f} | {'yes' if m['was_shortlisted'] else 'no'} |" for m in rv["missed_movers"]]
    L += ["", "## Cumulative (all closed paper trades)", "| Strategy | N | Win % | EV %/trade | Total % |", "|---|---|---|---|---|"]
    for k, v in rv["cumulative"].items():
        if v["n"]:
            L.append(f"| {k} | {v['n']} | {v['win_rate_pct']} | {v['ev_pct']:+.3f} | {v['total_pct']:+.2f} |")
    if rv["regime_history"]:
        L += ["", "## Regime (latest snapshots)"] + [
            f"- {h['as_of']}: fear/greed {h['fear_greed']}, pressure {h['pressure']:+.2f} ({h['label']})" for h in rv["regime_history"]]
    return "\n".join(L) + "\n"


def save_review(review: dict, out_dir: Path = REPORT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{review['week_start']}.md"
    md.write_text(review["markdown"])
    (out_dir / f"{review['week_start']}.json").write_text(
        json.dumps({k: v for k, v in review.items() if k != "markdown"}, indent=2, default=str))
    return md
