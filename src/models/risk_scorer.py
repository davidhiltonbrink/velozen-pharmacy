"""
Velozen AI — Risk Scoring Module

Adds numeric risk scores (0–100) to the output of compute_recommendations().

Two independent scores per SKU:

  understock_score  — probability-weighted severity of running out of stock.
                      Driven by days_of_supply vs lead time + safety buffer,
                      modulated by forecast uncertainty (MAPE) and historical
                      bias direction.

  overstock_score   — severity of having too much stock relative to demand,
                      with expiry pressure as the primary amplifier.
                      A drug sitting at 10× target supply with an expiry date
                      arriving before it can be dispensed scores near 100.

  risk_label        — human-readable composite:
                        STOCKOUT_CRITICAL  understock >= 75
                        STOCKOUT_WARNING   understock >= 40
                        OVERSTOCK_CRITICAL overstock  >= 75
                        OVERSTOCK_WARNING  overstock  >= 40
                        OK                 both < 40

Score thresholds
----------------
  0–39   Low       no meaningful action required
  40–74  Warning   monitor / flag for review
  75–100 Critical  act immediately
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Label thresholds
CRITICAL_THRESHOLD = 75
WARNING_THRESHOLD  = 40


def _understock_score(
    dos: float,
    target: int,
    lead_time: int,
    min_days: int,
    mape_pct: float,
    bias: float,
    weekly_demand: float,
) -> float:
    """
    0  → well stocked (dos >= target)
    100 → stockout imminent or active

    Segments
    --------
    [target, warn_threshold]  →  0–40  low concern
    [warn_threshold, min_days] → 40–75  medium concern (order now or risk stockout)
    [min_days, 0]              → 75–100 critical
    """
    if dos >= target:
        return 0.0

    warn_threshold = lead_time + min_days

    if dos >= warn_threshold:
        span = max(target - warn_threshold, 1)
        base = 40.0 * (target - dos) / span
    elif dos >= min_days:
        span = max(warn_threshold - min_days, 1)
        base = 40.0 + 35.0 * (warn_threshold - dos) / span
    else:
        span = max(min_days, 1)
        base = 75.0 + 25.0 * (min_days - max(dos, 0)) / span

    # Forecast uncertainty adds up to +10 pts (MAPE 50% → +10)
    mape_adj = min(10.0, mape_pct / 5.0)

    # Bias: negative bias means model under-forecasts → real demand is higher → riskier
    # Positive bias: model over-forecasts → real demand is lower → less risky
    bias_adj = 0.0
    if weekly_demand > 0:
        bias_pct = -bias / weekly_demand      # negative bias → positive adjustment
        bias_adj = float(np.clip(bias_pct * 10.0, -5.0, 5.0))

    return float(np.clip(base + mape_adj + bias_adj, 0.0, 100.0))


def _overstock_score(
    dos: float,
    target: int,
    days_until_expiry: float | None,
) -> float:
    """
    0  → not overstocked (dos <= target)
    100 → severely overstocked with near-certain expiry waste

    Two additive components (each capped at 50):
    1. Overstock magnitude — how many multiples of the target is dos?
       1× = 0 pts, 2× = 25 pts, 4× = 50 pts
    2. Expiry pressure — what fraction of on-hand will expire unused?
       waste_fraction = max(0, dos − days_until_expiry) / dos
    """
    if dos >= 999 or dos <= target:
        return 0.0

    # Component 1: overstock magnitude
    overstock_ratio = dos / max(target, 1)
    magnitude = float(np.clip((overstock_ratio - 1.0) / 3.0 * 50.0, 0.0, 50.0))

    # Component 2: expiry pressure
    expiry_component = 0.0
    if days_until_expiry is not None and not (isinstance(days_until_expiry, float) and np.isnan(days_until_expiry)):
        due = float(days_until_expiry)
        if due <= 0:
            expiry_component = 50.0                   # already expired
        elif due < dos:
            waste_fraction = (dos - due) / dos
            expiry_component = waste_fraction * 50.0

    return float(np.clip(magnitude + expiry_component, 0.0, 100.0))


def _risk_label(understock: float, overstock: float) -> str:
    if understock >= CRITICAL_THRESHOLD:
        return "STOCKOUT_CRITICAL"
    if overstock >= CRITICAL_THRESHOLD:
        return "OVERSTOCK_CRITICAL"
    if understock >= WARNING_THRESHOLD:
        return "STOCKOUT_WARNING"
    if overstock >= WARNING_THRESHOLD:
        return "OVERSTOCK_WARNING"
    return "OK"


def compute_risk_scores(
    recs: pd.DataFrame,
    target_supply_days: int = 30,
    lead_time_days: int     = 5,
    min_supply_days: int    = 7,
) -> pd.DataFrame:
    """
    Augment a compute_recommendations() DataFrame with risk scores.

    Parameters
    ----------
    recs
        Output of compute_recommendations() — must include:
        days_of_supply, days_until_expiry, mape_pct, bias,
        forecast_weekly_demand
    target_supply_days, lead_time_days, min_supply_days
        Must match the values used when generating recs.

    Returns
    -------
    recs with three new columns appended:
        understock_score  float  0–100
        overstock_score   float  0–100
        risk_label        str
    """
    out = recs.copy()

    out["understock_score"] = [
        _understock_score(
            dos=row["days_of_supply"],
            target=target_supply_days,
            lead_time=lead_time_days,
            min_days=min_supply_days,
            mape_pct=row["mape_pct"],
            bias=row["bias"],
            weekly_demand=row["forecast_weekly_demand"],
        )
        for _, row in out.iterrows()
    ]

    out["overstock_score"] = [
        _overstock_score(
            dos=row["days_of_supply"],
            target=target_supply_days,
            days_until_expiry=row.get("days_until_expiry"),
        )
        for _, row in out.iterrows()
    ]

    out["risk_label"] = [
        _risk_label(u, o)
        for u, o in zip(out["understock_score"], out["overstock_score"])
    ]

    return out
