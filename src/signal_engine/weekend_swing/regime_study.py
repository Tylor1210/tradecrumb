"""
Regime study: slices backtest/paper trades by the market regime they entered in
and measures whether a RegimePolicy would have improved results.

Called by: cli.py (`weekend-regime-study`), weekend_swing/review.py.
Depends on: market_regime.policy. Works on any list of dicts with
{direction, return_pct, regime(dict|None), entry_time}, so it serves both
backtest trades and closed paper trades.

WHY split-half: with a few hundred trades over 12 months it is easy to find a
"filter" that only fits noise. Every comparison is also reported on each half
of the date range; a rule that only helps one half should not be enabled.
"""

from statistics import mean
from typing import Callable, Optional

from signal_engine.market_regime.policy import RegimePolicy


def _stats(returns: list, weights: Optional[list] = None) -> dict:
    if not returns:
        return {"n": 0, "win_rate_pct": None, "ev_pct": None, "total_pct": 0.0}
    w = weights or [1.0] * len(returns)
    wsum = sum(w) or 1.0
    wins = sum(1 for r in returns if r > 0)
    return {
        "n": len(returns),
        "win_rate_pct": round(wins / len(returns) * 100, 1),
        "ev_pct": round(sum(r * x for r, x in zip(returns, w)) / wsum, 3),  # size-weighted EV per $ deployed
        "total_pct": round(sum(r * x for r, x in zip(returns, w)), 2),     # P&L in units of one full-size trade
    }


def _group(trades: list, key: Callable) -> dict:
    out: dict = {}
    for t in trades:
        k = key(t)
        if k is not None:
            out.setdefault(k, []).append(t["return_pct"])
    return {k: _stats(v) for k, v in sorted(out.items())}


def _aligned(t: dict) -> Optional[str]:
    reg = t.get("regime")
    if not reg:
        return None
    p = reg["pressure"]
    if abs(p) < 0.15:
        return "neutral_pressure"
    return "with_pressure" if (p > 0) == (t["direction"] == "long") else "against_pressure"


def _fg_bucket(t: dict) -> Optional[str]:
    reg = t.get("regime")
    if not reg or reg.get("fear_greed") is None:
        return None
    g = reg["fear_greed"]
    return "fear(<40)" if g < 40 else "greed(>60)" if g > 60 else "neutral(40-60)"


def evaluate_policy(trades: list, policy: RegimePolicy) -> dict:
    """Baseline vs policy-applied results (skips removed, reduced trades size-weighted)."""
    base = _stats([t["return_pct"] for t in trades])
    kept, weights, skipped = [], [], []
    for t in trades:
        d = policy.evaluate(t["direction"], t.get("regime"))
        if d.allow and d.size_mult > 0:
            kept.append(t["return_pct"])
            weights.append(d.size_mult)
        else:
            skipped.append(t["return_pct"])
    return {"baseline": base, "with_policy": _stats(kept, weights), "skipped": _stats(skipped)}


def run_study(trades: list, policy: RegimePolicy) -> dict:
    """trades: dicts as described in the module docstring. Trades with no regime tag are ignored."""
    tagged = [t for t in trades if t.get("regime")]
    tagged.sort(key=lambda t: str(t["entry_time"]))
    mid = len(tagged) // 2
    return {
        "n_trades": len(trades),
        "n_tagged": len(tagged),
        "by_pressure_label": _group(tagged, lambda t: f"{t['regime']['pressure_label']}/{t['direction']}"),
        "by_alignment": _group(tagged, _aligned),
        "by_fear_greed": _group(tagged, lambda t: f"{_fg_bucket(t)}/{t['direction']}" if _fg_bucket(t) else None),
        "policy": evaluate_policy(tagged, policy),
        "policy_first_half": evaluate_policy(tagged[:mid], policy),
        "policy_second_half": evaluate_policy(tagged[mid:], policy),
        "mean_return_pct": round(mean(t["return_pct"] for t in tagged), 3) if tagged else None,
    }


def trades_to_dicts(trades: list) -> list:
    """WeekendSwingTrade objects -> plain dicts for run_study()."""
    return [
        {"ticker": t.ticker, "strategy": t.strategy, "gap_type": t.gap_type, "direction": t.direction,
         "return_pct": t.return_pct, "entry_time": str(t.entry_time), "regime": t.regime}
        for t in trades
    ]
