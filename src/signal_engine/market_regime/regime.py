"""
Market regime: Fear/Greed index (0-100) + directional pressure (-1..+1).

What it does: turns daily bars for SPY, bond/credit ETFs, sector ETFs and a
breadth universe into (a) a CNN-style fear/greed level and (b) a signed
buying/selling-pressure score with the drivers behind it, so the bot can tell
which way the pressure is coming from and how broad it is.

Called by: market_regime/provider.py (point-in-time, live + backtest),
cli.py (`regime`), weekend_swing monitor/review, dashboard/server.py.
Depends on: pandas/numpy only. Pure functions over DataFrames, no I/O.

CONSTRAINT: every function here sees only bars strictly before `as_of`.
Backtests call this per historical Monday; one bar of look-ahead would
silently inflate every regime study built on it.

GOTCHA: VIX is not available on Alpaca's stock feed. Volatility fear is the
SPY 10d/60d realized-vol ratio, which moves with VIX but is not VIX.

GOTCHA: IEX-feed volume is a fraction of consolidated volume. Money-flow and
up/down-volume use *ratios*, so the scale cancels; do not compare absolute
volumes to other sources.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_LABELS = [(25, "Extreme Fear"), (45, "Fear"), (55, "Neutral"), (75, "Greed"), (101, "Extreme Greed")]


@dataclass
class MarketRegime:
    as_of: str                       # date of the last bar used (ISO)
    fear_greed: Optional[float]      # 0-100, None if too few components were available
    fear_greed_label: str
    pressure: float                  # -1 (selling) .. +1 (buying)
    pressure_label: str              # BULLISH | BEARISH | NEUTRAL
    components: dict = field(default_factory=dict)         # fear/greed pieces: {name: {score, raw}}
    pressure_components: dict = field(default_factory=dict)  # {name: value in [-1,1]}
    sectors: list = field(default_factory=list)            # [{symbol, flow}] sorted strongest buying first
    breadth_ad: Optional[float] = None                     # share of breadth universe up over ad_days
    summary: list = field(default_factory=list)            # plain-English "where is pressure coming from"

    def to_dict(self) -> dict:
        return asdict(self)


# ── primitives ──────────────────────────────────────────────────────────────

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _scale(value: float, lo: float, hi: float) -> float:
    """Linear map lo->0, hi->100, clipped. lo > hi gives an inverted scale."""
    if hi == lo:
        return 50.0
    return _clip((value - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def _ret_pct(df: Optional[pd.DataFrame], n: int) -> Optional[float]:
    if df is None or len(df) <= n:
        return None
    c = df["close"]
    prev = float(c.iloc[-1 - n])
    return (float(c.iloc[-1]) / prev - 1.0) * 100.0 if prev else None


def _cmf(df: Optional[pd.DataFrame], n: int) -> Optional[float]:
    """Chaikin money flow over the last n bars, in [-1, 1]. Close near the high on volume = buying."""
    if df is None or len(df) < n:
        return None
    w = df.iloc[-n:]
    rng = (w["high"] - w["low"]).replace(0, np.nan)
    clv = (((w["close"] - w["low"]) - (w["high"] - w["close"])) / rng).fillna(0.0)
    vol = w["volume"].astype(float)
    total = float(vol.sum())
    return float((clv * vol).sum() / total) if total > 0 else None


def _updown_volume_share(df: Optional[pd.DataFrame], n: int) -> Optional[float]:
    """Share of the last n bars' volume that traded on up-closes (vs the prior close)."""
    if df is None or len(df) < n + 1:
        return None
    w = df.iloc[-(n + 1):]
    up = w["close"].diff().iloc[1:] > 0
    vol = w["volume"].iloc[1:].astype(float)
    total = float(vol.sum())
    return float(vol[up].sum() / total) if total > 0 else None


def _realized_vol(df: Optional[pd.DataFrame], n: int) -> Optional[float]:
    if df is None or len(df) < n + 1:
        return None
    r = np.log(df["close"].astype(float)).diff().iloc[-n:]
    v = float(r.std())
    return v if not math.isnan(v) else None


def _above_sma(df: Optional[pd.DataFrame], n: int) -> Optional[bool]:
    if df is None or len(df) < n:
        return None
    return bool(df["close"].iloc[-1] > df["close"].iloc[-n:].mean())


def _truncate(df: Optional[pd.DataFrame], as_of: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if as_of is None:
        return df
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    out = df.loc[df.index < as_of]
    return out if not out.empty else None


def _label(level: float, labels) -> str:
    for upper, name in labels:
        if level < upper:
            return name
    return labels[-1][1]


# ── main entry ──────────────────────────────────────────────────────────────

def compute_regime(
    daily: dict,
    cfg: dict,
    breadth_symbols: Optional[list] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> Optional[MarketRegime]:
    """
    daily: {symbol: daily OHLCV DataFrame (UTC DatetimeIndex)}.
    breadth_symbols: the stock universe used for breadth / universe money flow.
    as_of: only bars strictly before this timestamp are used (None = all bars).
    Returns None if the benchmark has no data.
    """
    sym = cfg.get("symbols", {})
    fg = cfg.get("fear_greed", {})
    pr = cfg.get("pressure", {})

    bars = {s: _truncate(df, as_of) for s, df in daily.items()}
    spy = bars.get(sym.get("benchmark", "SPY"))
    if spy is None or len(spy) < 20:
        return None

    labels = [tuple(x) for x in fg.get("labels", DEFAULT_LABELS)]
    breadth_universe = [bars[s] for s in (breadth_symbols or []) if bars.get(s) is not None]

    # ── Fear / Greed components (each 0-100) ────────────────────────────────
    comps: dict = {}

    sma_days = fg.get("momentum_sma_days", 125)
    if len(spy) >= sma_days:
        dev = (float(spy["close"].iloc[-1]) / float(spy["close"].iloc[-sma_days:].mean()) - 1) * 100
        r = fg.get("momentum_range_pct", 6.0)
        comps["momentum"] = {"score": _scale(dev, -r, r), "raw": round(dev, 2)}

    flags = [_above_sma(df, fg.get("breadth_sma_days", 50)) for df in breadth_universe]
    flags = [f for f in flags if f is not None]
    if len(flags) >= fg.get("min_breadth_symbols", 5):
        pct = sum(flags) / len(flags) * 100
        comps["breadth_strength"] = {"score": pct, "raw": round(pct, 1)}

    v_short = _realized_vol(spy, fg.get("vol_short_days", 10))
    v_long = _realized_vol(spy, fg.get("vol_long_days", 60))
    if v_short is not None and v_long:
        ratio = v_short / v_long
        comps["volatility"] = {
            "score": _scale(ratio, fg.get("vol_ratio_fear", 1.8), fg.get("vol_ratio_greed", 0.6)),
            "raw": round(ratio, 3),
        }

    n = fg.get("safe_haven_days", 20)
    spy_r, tlt_r = _ret_pct(spy, n), _ret_pct(bars.get(sym.get("bonds_long", "TLT")), n)
    if spy_r is not None and tlt_r is not None:
        r = fg.get("safe_haven_range_pct", 6.0)
        comps["safe_haven"] = {"score": _scale(spy_r - tlt_r, -r, r), "raw": round(spy_r - tlt_r, 2)}

    n = fg.get("junk_bond_days", 20)
    hyg_r = _ret_pct(bars.get(sym.get("junk", "HYG")), n)
    lqd_r = _ret_pct(bars.get(sym.get("investment_grade", "LQD")), n)
    if hyg_r is not None and lqd_r is not None:
        r = fg.get("junk_bond_range_pct", 1.5)
        comps["junk_bond"] = {"score": _scale(hyg_r - lqd_r, -r, r), "raw": round(hyg_r - lqd_r, 2)}

    n = fg.get("rotation_days", 20)
    off = [x for x in (_ret_pct(bars.get(s), n) for s in sym.get("offensive", [])) if x is not None]
    dfn = [x for x in (_ret_pct(bars.get(s), n) for s in sym.get("defensive", [])) if x is not None]
    if off and dfn:
        spread = sum(off) / len(off) - sum(dfn) / len(dfn)
        r = fg.get("rotation_range_pct", 5.0)
        comps["risk_rotation"] = {"score": _scale(spread, -r, r), "raw": round(spread, 2)}

    if len(comps) >= fg.get("min_components", 3):
        level = round(sum(c["score"] for c in comps.values()) / len(comps), 1)
        level_label = _label(level, labels)
    else:
        level, level_label = None, "Unavailable"

    # ── Directional pressure (each -1..+1) ──────────────────────────────────
    flow_days = pr.get("flow_days", 10)
    gain = pr.get("flow_gain", 3.0)
    pcomps: dict = {}

    spy_cmf = _cmf(spy, flow_days)
    if spy_cmf is not None:
        pcomps["index_flow"] = _clip(spy_cmf * gain, -1, 1)

    cmfs = [x for x in (_cmf(df, flow_days) for df in breadth_universe) if x is not None]
    if len(cmfs) >= fg.get("min_breadth_symbols", 5):
        pcomps["universe_flow"] = _clip(float(np.mean(cmfs)) * gain, -1, 1)

    ad_days = pr.get("ad_days", 5)
    rets = [x for x in (_ret_pct(df, ad_days) for df in breadth_universe) if x is not None]
    breadth_ad = None
    if len(rets) >= fg.get("min_breadth_symbols", 5):
        breadth_ad = sum(1 for x in rets if x > 0) / len(rets)
        pcomps["advance_decline"] = (breadth_ad - 0.5) * 2

    udv = _updown_volume_share(spy, flow_days)
    if udv is not None:
        pcomps["updown_volume"] = (udv - 0.5) * 2

    trend = _ret_pct(spy, pr.get("trend_days", 5))
    if trend is not None:
        t = pr.get("trend_range_pct", 3.0)
        pcomps["short_term_trend"] = _clip(trend / t, -1, 1)

    pressure = round(float(np.mean(list(pcomps.values()))), 3) if pcomps else 0.0
    bull_t, bear_t = pr.get("bullish_threshold", 0.15), pr.get("bearish_threshold", -0.15)
    p_label = "BULLISH" if pressure >= bull_t else "BEARISH" if pressure <= bear_t else "NEUTRAL"

    # Sector flow: which parts of the market the buying/selling is concentrated in.
    sectors = []
    for s in sym.get("sectors", []):
        f = _cmf(bars.get(s), flow_days)
        if f is not None:
            sectors.append({"symbol": s, "flow": round(f, 3)})
    sectors.sort(key=lambda x: x["flow"], reverse=True)

    summary = _summarize(pressure, p_label, pcomps, sectors, breadth_ad, pr, level, level_label)

    return MarketRegime(
        as_of=str(spy.index[-1].date()),
        fear_greed=level,
        fear_greed_label=level_label,
        pressure=pressure,
        pressure_label=p_label,
        components={k: {"score": round(v["score"], 1), "raw": v["raw"]} for k, v in comps.items()},
        pressure_components={k: round(v, 3) for k, v in pcomps.items()},
        sectors=sectors,
        breadth_ad=round(breadth_ad, 3) if breadth_ad is not None else None,
        summary=summary,
    )


def _summarize(pressure, p_label, pcomps, sectors, breadth_ad, pr_cfg, level, level_label) -> list:
    lines = []
    if level is not None:
        lines.append(f"Fear/Greed {level:.0f} ({level_label}); pressure {pressure:+.2f} ({p_label}).")
    else:
        lines.append(f"Pressure {pressure:+.2f} ({p_label}); fear/greed unavailable (too little data).")

    if pcomps:
        top = max(pcomps.items(), key=lambda kv: abs(kv[1]))
        lines.append(f"Biggest driver: {top[0].replace('_', ' ')} ({top[1]:+.2f}).")

    if len(sectors) >= 4:
        buyers = [s for s in sectors[:3] if s["flow"] > 0]
        sellers = [s for s in sectors[-3:] if s["flow"] < 0]
        if buyers:
            lines.append("Buying concentrated in " + ", ".join(f"{s['symbol']} ({s['flow']:+.2f})" for s in buyers) + ".")
        if sellers:
            lines.append("Selling concentrated in " + ", ".join(f"{s['symbol']} ({s['flow']:+.2f})" for s in reversed(sellers)) + ".")

    narrow = pr_cfg.get("narrow_breadth_ad", 0.45)
    if breadth_ad is not None:
        if pcomps.get("index_flow", 0) > 0.1 and breadth_ad < narrow:
            lines.append(f"Narrow rally: index flow positive but only {breadth_ad:.0%} of stocks are up — fragile.")
        elif pcomps.get("index_flow", 0) < -0.1 and breadth_ad > 1 - narrow:
            lines.append(f"Index selling but {breadth_ad:.0%} of stocks are up — selling is concentrated in a few names.")
    return lines
