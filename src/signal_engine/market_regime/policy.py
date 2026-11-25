"""
Regime policy: turns a MarketRegime into allow/skip + a position-size multiplier
for one trade direction.

Called by: weekend_swing monitor/auto-submit, weekend_swing/review.py (to
re-apply the policy to history), the regime study. Depends on: nothing.

CONSTRAINT: disabled by default. It only gets switched on after the regime
study shows it helps out-of-sample; an unvalidated filter is just a way to
throw away trades.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeDecision:
    allow: bool
    size_mult: float
    reason: str


class RegimePolicy:
    def __init__(self, cfg: dict) -> None:
        p = cfg.get("policy", {})
        self.enabled = bool(p.get("enabled", False))
        self.counter_pressure = float(p.get("counter_trend_pressure", 0.25))
        self.counter_action = p.get("counter_trend_action", "reduce")
        self.counter_mult = float(p.get("counter_trend_size_mult", 0.5))
        self.crowded_greed = float(p.get("crowded_greed_level", 80))
        self.crowded_fear = float(p.get("crowded_fear_level", 20))
        self.crowded_mult = float(p.get("crowded_size_mult", 1.0))

    def evaluate(self, direction: str, regime) -> RegimeDecision:
        """`regime` may be a MarketRegime or its dict form (as stored in breakdown_json)."""
        if not self.enabled:
            return RegimeDecision(True, 1.0, "regime policy disabled")
        if regime is None:
            return RegimeDecision(True, 1.0, "no regime data")

        g = (lambda k: regime.get(k)) if isinstance(regime, dict) else (lambda k: getattr(regime, k))
        pressure, level = g("pressure"), g("fear_greed")
        mult, notes = 1.0, []

        counter = (direction == "long" and pressure <= -self.counter_pressure) or (
            direction == "short" and pressure >= self.counter_pressure
        )
        if counter:
            if self.counter_action == "skip":
                return RegimeDecision(False, 0.0, f"{direction} against {pressure:+.2f} pressure")
            mult *= self.counter_mult
            notes.append(f"{direction} against {pressure:+.2f} pressure")

        if level is not None:
            if direction == "long" and level >= self.crowded_greed:
                mult *= self.crowded_mult
                notes.append(f"crowded long (greed {level:.0f})")
            elif direction == "short" and level <= self.crowded_fear:
                mult *= self.crowded_mult
                notes.append(f"crowded short (fear {level:.0f})")

        return RegimeDecision(True, round(mult, 3), "; ".join(notes) or "aligned")
