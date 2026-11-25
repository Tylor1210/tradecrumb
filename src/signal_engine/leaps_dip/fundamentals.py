"""
Point-in-time revenue growth check for the dip strategy.

Uses Finnhub's "financials as reported" (SEC filings, annual). Each filing
carries the date it was filed, so on a signal date we only look at annual
reports filed on or before that day. That avoids look-ahead: a 10-K filed
in February can't help a signal in January.

The video's rule is "revenue growing year over year". The check passes when
the latest annual revenue known on that date is higher than the year before.
The video also wants revenue "forecast to keep growing"; historical analyst
forecasts aren't available point in time, so that part is not tested.

Depends on: finnhub client (optional), pandas.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# XBRL revenue concepts, most common first. Finnhub prefixes them, e.g.
# "us-gaap_Revenues", so matching is done on the part after the prefix.
REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",
    "TotalRevenuesAndOtherIncome",
]

REVENUE_COLUMNS = ["filed_date", "fiscal_year", "revenue"]


def _concept_name(concept: str) -> str:
    for sep in ("_", ":"):
        if sep in concept:
            concept = concept.split(sep)[-1]
    return concept


def parse_financials_reported(payload: dict) -> pd.DataFrame:
    """
    Turn a Finnhub /stock/financials-reported response into one row per
    annual filing: filed_date, fiscal_year, revenue. Filings without a
    recognizable revenue line are dropped.
    """
    rows = []
    for item in (payload or {}).get("data", []) or []:
        filed = item.get("filedDate") or item.get("acceptedDate")
        year = item.get("year")
        ic = ((item.get("report") or {}).get("ic")) or []
        if not filed or year is None or not ic:
            continue
        by_concept = {}
        for line in ic:
            name = _concept_name(str(line.get("concept", "")))
            value = line.get("value")
            if isinstance(value, (int, float)) and name not in by_concept:
                by_concept[name] = float(value)
        revenue = next((by_concept[c] for c in REVENUE_CONCEPTS if c in by_concept), None)
        if revenue is None or revenue <= 0:
            continue
        rows.append({
            "filed_date": pd.Timestamp(str(filed)[:10]),
            "fiscal_year": int(year),
            "revenue": revenue,
        })
    if not rows:
        return pd.DataFrame(columns=REVENUE_COLUMNS)
    df = pd.DataFrame(rows).sort_values(["fiscal_year", "filed_date"])
    # Keep the first filing for each fiscal year (later amendments would be look-ahead)
    df = df.drop_duplicates("fiscal_year", keep="first").reset_index(drop=True)
    return df[REVENUE_COLUMNS]


def revenue_growing_as_of(history: Optional[pd.DataFrame], date: pd.Timestamp) -> Optional[bool]:
    """
    True if the latest annual revenue filed on or before `date` beats the
    prior year's. None when fewer than two years were known on that date.
    """
    if history is None or history.empty:
        return None
    day = pd.Timestamp(date)
    if day.tzinfo is not None:
        day = day.tz_convert(None)
    known = history[history["filed_date"] <= day.normalize()]
    known = known.sort_values("fiscal_year")
    if len(known) < 2:
        return None
    latest, prior = known.iloc[-1], known.iloc[-2]
    if latest["fiscal_year"] - prior["fiscal_year"] != 1:
        return None
    return bool(latest["revenue"] > prior["revenue"])


def fetch_revenue_history(finnhub_raw_client, ticker: str, cache=None) -> pd.DataFrame:
    """
    Annual revenue history for one ticker, cached for 30 days.
    finnhub_raw_client is a finnhub.Client (FinnhubNewsClient._client).
    """
    key = f"leaps_dip:revenue:{ticker}"
    if cache is not None:
        hit = cache.get(key, ttl_hours=24 * 30)
        if hit is not None:
            hit["filed_date"] = pd.to_datetime(hit["filed_date"])
            return hit
    try:
        payload = finnhub_raw_client.financials_reported(symbol=ticker, freq="annual")
    except Exception as exc:  # network, rate limit, premium-only response
        logger.warning("Finnhub financials-reported failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=REVENUE_COLUMNS)
    df = parse_financials_reported(payload)
    if cache is not None and not df.empty:
        cache.set(key, df)
    return df
