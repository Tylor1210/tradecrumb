"""
LEAPS candidate screener.

Identifies long-horizon (LEAPS, 1yr+ expiration) options candidates
based on fundamental/macro catalysts rather than short-term news spikes.

Scoring dimensions:
  1. Relative strength vs SPY over 180 days
  2. Earnings trend (consecutive positive EPS growth quarters via Finnhub)
  3. Macro theme alignment (AI, defense, GLP-1, etc.)
  4. Long-term price trend (% above SMA-200)

Each dimension contributes a score in [0, 1]. The composite LEAPS score
is a weighted average.

Options chain data (IV, strikes, Greeks) is stubbed — see options_stub.py.
The screener is fully functional for candidate ranking without options data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from math import erf, exp, log, sqrt
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Risk-free rate assumption (~4.5% — approximate US 1yr T-bill yield)
_RISK_FREE_RATE = 0.045


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf — no scipy needed."""
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def _black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.

    Args:
        S:     Current stock price
        K:     Strike price
        T:     Time to expiry in years
        r:     Risk-free rate (annualized)
        sigma: Implied / historical volatility (annualized)

    Returns:
        Call option price per share (multiply by 100 for 1 contract)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return max(0.0, call_price)


@dataclass
class LEAPSCandidate:
    """
    LEAPS candidate for a single ticker.

    leaps_score in [0, 1] — higher is stronger candidate.
    options_data_available reflects whether live chain data is connected.
    """

    ticker: str
    scored_at: datetime
    leaps_score: float              # 0 to 1 (1 = strongest candidate)
    catalysts: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)

    # Suggested options parameters (informational — not executable without chain data)
    suggested_strike: Optional[float] = None  # ATM or slight OTM
    suggested_expiry_min_dte: int = 365        # minimum days to expiration (LEAPS = 1yr+)
    current_price: Optional[float] = None
    options_data_available: bool = False       # True when options_stub is connected
    direction: str = "CALL"                    # Always CALL — this screener is bullish/long only
    direction_rationale: str = "Long call: stock expected to rise over 1yr+ horizon"

    # ── Price target & return estimates ──────────────────────────────────────
    hist_volatility: Optional[float] = None      # annualized historical volatility (e.g. 0.28 = 28%)
    target_price_1yr: Optional[float] = None     # conservative 1yr price target
    target_upside_pct: Optional[float] = None    # % gain from current → target
    est_call_premium: Optional[float] = None     # Black-Scholes call price per share
    est_cost_per_contract: Optional[float] = None  # premium * 100 (1 contract = 100 shares)
    est_return_on_1k: Optional[float] = None     # dollar P&L if $1,000 invested & target hit
    est_return_pct_on_1k: Optional[float] = None # % return on that $1,000


class LEAPSScreener:
    """
    Screens a watchlist for LEAPS options candidates.

    Scoring is based on long-horizon catalysts only — not short-term signals.
    """

    # Component weights (must sum to 1.0)
    _W_RELATIVE_STRENGTH = 0.30
    _W_EARNINGS_TREND = 0.30
    _W_MACRO_THEMES = 0.20
    _W_LONG_TREND = 0.20

    def __init__(self, config: dict, finnhub_client=None) -> None:
        """
        Args:
            config:          leaps section from config/signals.yaml
            finnhub_client:  FinnhubNewsClient instance (optional — for earnings data)
        """
        self.config = config
        self.finnhub = finnhub_client
        self.earnings_growth_quarters = config.get("earnings_growth_quarters", 3)
        self.rs_window_days = config.get("relative_strength_window_days", 180)
        self.active_macro_themes = set(config.get("active_macro_themes", []))

    def screen(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        watchlist_entry: dict,
    ) -> LEAPSCandidate:
        """
        Screen a single ticker for LEAPS candidacy.

        Args:
            ticker:          Ticker symbol
            price_data:      OHLCV DataFrame (DatetimeIndex, daily bars)
            spy_data:        SPY OHLCV DataFrame (same index type)
            watchlist_entry: Dict from watchlist.yaml for this ticker

        Returns:
            LEAPSCandidate with score, catalyst explanation, and return estimates
        """
        catalysts: list[str] = []
        breakdown: dict = {}
        score_components: list[float] = []

        current_price = float(price_data["close"].iloc[-1]) if not price_data.empty else None

        # ── 1. Relative Strength vs SPY ──────────────────────────────────
        rs_result = self._score_relative_strength(price_data, spy_data)
        breakdown["relative_strength"] = rs_result
        score_components.append(self._W_RELATIVE_STRENGTH * rs_result["score"])
        if rs_result["score"] >= 0.6:
            rs_pct = rs_result.get("outperformance_pct", 0)
            catalysts.append(f"Strong RS vs SPY: {rs_pct:+.1f}% over {self.rs_window_days}d")

        # ── 2. Earnings Trend ────────────────────────────────────────────
        earnings_result = self._score_earnings_trend(ticker)
        breakdown["earnings_trend"] = earnings_result
        score_components.append(self._W_EARNINGS_TREND * earnings_result["score"])
        if earnings_result["score"] >= 0.5:
            catalysts.append(
                f"Positive earnings trend: {earnings_result.get('quarters_positive', 0)} consecutive quarters"
            )

        # ── 3. Macro Theme Alignment ─────────────────────────────────────
        theme_result = self._score_macro_themes(watchlist_entry)
        breakdown["macro_theme_alignment"] = theme_result
        score_components.append(self._W_MACRO_THEMES * theme_result["score"])
        for theme in theme_result.get("matched_themes", []):
            catalysts.append(f"Active macro theme: {theme.replace('_', ' ').title()}")

        # ── 4. Long-Term Price Trend (SMA-200) ───────────────────────────
        trend_result = self._score_long_trend(price_data)
        breakdown["long_term_trend"] = trend_result
        score_components.append(self._W_LONG_TREND * trend_result["score"])
        pct_above = trend_result.get("pct_above_sma200")
        if pct_above and pct_above > 10:
            catalysts.append(f"Strong long-term trend: {pct_above:+.1f}% above SMA-200")

        # ── Composite LEAPS score ─────────────────────────────────────────
        leaps_score = sum(score_components)
        leaps_score = float(max(0.0, min(1.0, leaps_score)))

        # Suggested parameters (ATM + slight OTM for leverage)
        suggested_strike = None
        if current_price:
            suggested_strike = round(current_price * 1.05, 2)  # ~5% OTM call

        # ── Return estimates ──────────────────────────────────────────────
        hist_vol = self._calc_historical_volatility(price_data)
        target_price, target_upside_pct = self._estimate_target_price(
            price_data, rs_result, current_price
        )
        est_premium, est_cost_per_contract, est_return_on_1k, est_return_pct_on_1k = (
            self._estimate_return_on_1k(
                current_price, suggested_strike, hist_vol, target_price
            )
        )

        breakdown["return_estimates"] = {
            "hist_volatility_pct": round(hist_vol * 100, 1) if hist_vol else None,
            "target_price_1yr": round(target_price, 2) if target_price else None,
            "target_upside_pct": round(target_upside_pct, 1) if target_upside_pct else None,
            "est_call_premium_per_share": round(est_premium, 2) if est_premium else None,
            "est_cost_per_contract": round(est_cost_per_contract, 2) if est_cost_per_contract else None,
            "investment": 1000.0,
            "est_return_on_1k_dollars": round(est_return_on_1k, 2) if est_return_on_1k is not None else None,
            "est_return_pct_on_1k": round(est_return_pct_on_1k, 1) if est_return_pct_on_1k is not None else None,
            "note": "Estimates use Black-Scholes with historical vol. Actual results vary.",
        }

        return LEAPSCandidate(
            ticker=ticker,
            scored_at=datetime.now(),
            leaps_score=round(leaps_score, 4),
            catalysts=catalysts if catalysts else ["No strong catalysts identified"],
            breakdown=breakdown,
            suggested_strike=suggested_strike,
            suggested_expiry_min_dte=365,
            current_price=current_price,
            options_data_available=False,
            hist_volatility=round(hist_vol, 4) if hist_vol else None,
            target_price_1yr=round(target_price, 2) if target_price else None,
            target_upside_pct=round(target_upside_pct, 1) if target_upside_pct else None,
            est_call_premium=round(est_premium, 2) if est_premium else None,
            est_cost_per_contract=round(est_cost_per_contract, 2) if est_cost_per_contract else None,
            est_return_on_1k=round(est_return_on_1k, 2) if est_return_on_1k is not None else None,
            est_return_pct_on_1k=round(est_return_pct_on_1k, 1) if est_return_pct_on_1k is not None else None,
        )

    # ─────────────────────────────────────────────────────────────
    # Return Estimation Methods
    # ─────────────────────────────────────────────────────────────

    def _calc_historical_volatility(self, price_data: pd.DataFrame) -> Optional[float]:
        """
        Annualized historical volatility from daily log returns.
        Uses up to 252 trading days (1 year). Returns None if insufficient data.
        """
        if len(price_data) < 30:
            return None
        close = price_data["close"].astype(float).iloc[-252:]
        log_returns = np.log(close / close.shift(1)).dropna()
        if len(log_returns) < 10:
            return None
        return float(log_returns.std() * sqrt(252))

    def _estimate_target_price(
        self,
        price_data: pd.DataFrame,
        rs_result: dict,
        current_price: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Estimate a conservative 1-year target price.

        Method: Annualize the 180-day price return, then apply a 60%
        confidence discount (momentum doesn't always sustain). Cap at +80%
        and floor at +2% (we only screen bullish candidates).

        Returns:
            (target_price, target_upside_pct) or (None, None) if insufficient data.
        """
        if current_price is None or current_price <= 0:
            return None, None

        ticker_return_180d = rs_result.get("ticker_return_pct")
        if ticker_return_180d is None:
            return None, None

        # Annualize 180d return and apply 60% confidence discount
        ticker_return_dec = ticker_return_180d / 100.0
        annualized_return = ((1 + ticker_return_dec) ** (365.0 / 180.0)) - 1
        conservative_return = annualized_return * 0.60

        # Cap at +80% upside, floor at +2% (never target a drop for long calls)
        conservative_return = max(0.02, min(0.80, conservative_return))

        target_price = current_price * (1 + conservative_return)
        target_upside_pct = conservative_return * 100

        return target_price, target_upside_pct

    def _estimate_return_on_1k(
        self,
        current_price: Optional[float],
        strike: Optional[float],
        hist_vol: Optional[float],
        target_price: Optional[float],
        investment: float = 1000.0,
        T: float = 1.0,  # 1 year to expiry
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Estimate the P&L on a $1,000 investment in LEAPS calls if target is hit.

        Steps:
          1. Price the call today using Black-Scholes (entry cost per share)
          2. Calculate how many contracts $1,000 buys (fractional for estimation)
          3. At target price, intrinsic value = max(0, target - strike)
          4. P&L = (intrinsic_at_target * 100 * contracts) - investment

        Returns:
            (premium_per_share, cost_per_contract, dollar_return, pct_return)
            All None if inputs are insufficient.
        """
        if not all([current_price, strike, hist_vol, target_price]):
            return None, None, None, None
        if current_price <= 0 or strike <= 0 or hist_vol <= 0:
            return None, None, None, None

        # Use at least 15% vol floor (even low-vol stocks have uncertainty over 1yr)
        vol = max(0.15, hist_vol)

        # Entry: Black-Scholes call price today
        premium_per_share = _black_scholes_call(
            S=current_price, K=strike, T=T, r=_RISK_FREE_RATE, sigma=vol
        )
        if premium_per_share <= 0:
            return None, None, None, None

        cost_per_contract = premium_per_share * 100  # 1 contract = 100 shares

        # Number of contracts $1k buys (fractional for estimation purposes)
        num_contracts = investment / cost_per_contract

        # At expiry at target price: intrinsic value of call
        intrinsic_at_target = max(0.0, target_price - strike)
        proceeds = intrinsic_at_target * 100 * num_contracts

        dollar_return = proceeds - investment
        pct_return = (dollar_return / investment) * 100

        return premium_per_share, cost_per_contract, dollar_return, pct_return

    # ─────────────────────────────────────────────────────────────
    # Scoring Methods
    # ─────────────────────────────────────────────────────────────

    def _score_relative_strength(
        self,
        price_data: pd.DataFrame,
        spy_data: pd.DataFrame,
    ) -> dict:
        """Relative return vs SPY over the RS window."""
        window = self.rs_window_days

        if len(price_data) < window or len(spy_data) < window:
            return {"score": 0.0, "reason": "insufficient_data"}

        ticker_return = (
            float(price_data["close"].iloc[-1]) - float(price_data["close"].iloc[-window])
        ) / float(price_data["close"].iloc[-window])

        spy_return = (
            float(spy_data["close"].iloc[-1]) - float(spy_data["close"].iloc[-window])
        ) / float(spy_data["close"].iloc[-window])

        outperformance = (ticker_return - spy_return) * 100  # in %

        # Normalize: +20% outperformance → score 1.0; -20% → score 0.0
        score = min(1.0, max(0.0, (outperformance + 20) / 40))

        return {
            "score": round(score, 4),
            "outperformance_pct": round(outperformance, 2),
            "ticker_return_pct": round(ticker_return * 100, 2),
            "spy_return_pct": round(spy_return * 100, 2),
            "window_days": window,
        }

    def _score_earnings_trend(self, ticker: str) -> dict:
        """Check for consecutive quarters of EPS growth using Finnhub."""
        if not self.finnhub:
            return {"score": 0.0, "reason": "finnhub_not_configured"}

        try:
            financials = self.finnhub.get_basic_financials(ticker, "all")
            series = financials.get("series", {})
            quarterly = series.get("quarterly", {})
            eps_series = quarterly.get("eps", [])

            if len(eps_series) < self.earnings_growth_quarters:
                return {
                    "score": 0.0,
                    "reason": f"insufficient_eps_data ({len(eps_series)} quarters)",
                }

            # Most recent quarters first
            recent = eps_series[: self.earnings_growth_quarters]
            values = [float(e.get("v", 0)) for e in recent]

            # Check consecutive growth (most recent > previous)
            is_growing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
            quarters_positive = sum(1 for v in values if v > 0)

            if is_growing and quarters_positive >= self.earnings_growth_quarters:
                score = 1.0
                signal = "consecutive_growth_all_positive"
            elif is_growing:
                score = 0.7
                signal = "consecutive_growth_some_negative"
            elif quarters_positive >= 2:
                score = 0.4
                signal = "mostly_positive_no_consecutive_growth"
            else:
                score = 0.1
                signal = "weak_earnings"

            return {
                "score": score,
                "signal": signal,
                "quarters_positive": quarters_positive,
                "is_growing": is_growing,
                "eps_values": [round(v, 2) for v in values],
            }

        except Exception as exc:
            logger.debug("Earnings trend fetch failed for %s: %s", ticker, exc)
            return {"score": 0.0, "reason": f"api_error: {exc}"}

    def _score_macro_themes(self, watchlist_entry: dict) -> dict:
        """Score alignment with currently active macro themes."""
        ticker_themes = set(watchlist_entry.get("themes", []))
        matched = ticker_themes & self.active_macro_themes

        if not matched:
            return {"score": 0.0, "matched_themes": [], "ticker_themes": list(ticker_themes)}

        # More matches = higher score, max at 3 matches
        score = min(1.0, len(matched) * 0.35)

        return {
            "score": round(score, 4),
            "matched_themes": list(matched),
            "ticker_themes": list(ticker_themes),
            "active_macro_themes": list(self.active_macro_themes),
        }

    def _score_long_trend(self, price_data: pd.DataFrame) -> dict:
        """Price trend: % above or below SMA-200."""
        if len(price_data) < 200:
            return {"score": 0.0, "reason": "insufficient_data_for_sma200"}

        close = price_data["close"].astype(float)
        sma200 = float(close.rolling(200).mean().iloc[-1])
        current = float(close.iloc[-1])

        if sma200 == 0:
            return {"score": 0.0, "reason": "sma200_zero"}

        pct_above = (current - sma200) / sma200 * 100

        # Normalize: +20% above SMA-200 = score 1.0; -20% = score 0.0
        score = min(1.0, max(0.0, (pct_above + 20) / 40))

        return {
            "score": round(score, 4),
            "pct_above_sma200": round(pct_above, 2),
            "current_price": round(current, 2),
            "sma200": round(sma200, 2),
        }
