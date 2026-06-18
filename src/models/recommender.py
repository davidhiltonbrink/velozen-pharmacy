"""
Velozen AI — Recommendation Engine

Computes per-SKU order recommendations from inventory, forecasts, and
model accuracy metrics. Designed for human-in-the-loop review: output
is a ranked table the pharmacist inspects before placing any order.

Core logic
----------
1. Average forecast demand (weekly) per NDC over the forecast period
2. Days of supply  = on_hand / daily_demand
3. Safety stock    = lead_time_demand × (1 + MAPE/100)
4. Bias correction = −bias × 0.5  (partial correction for systematic over/under forecast)
5. Target stock    = daily_demand × target_days + safety_stock
6. Order qty       = max(0, target − on_hand + bias_adj), rounded to pack size
7. Risk level      = HIGH / MEDIUM / LOW based on days of supply vs thresholds
8. Expiration risk = current stock likely to expire before it is dispensed
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# Defaults (all overridable by callers)
DEFAULT_TARGET_SUPPLY_DAYS = 30
DEFAULT_LEAD_TIME_DAYS     = 5
DEFAULT_MIN_SUPPLY_DAYS    = 7   # below this → HIGH risk regardless of order
INFINITE_SUPPLY            = 999  # sentinel when daily demand is zero


def _normalize_ndc(series: pd.Series) -> pd.Series:
    """Strip hyphens and zero-pad to 11 digits for consistent joining."""
    return series.astype(str).str.replace("-", "", regex=False).str.zfill(11)


def _smallest_pack_size(pack_str: Optional[str]) -> int:
    """Return smallest pack size from a pipe-delimited string, or 1."""
    if not pack_str or str(pack_str) in ("nan", "None", ""):
        return 1
    try:
        return min(int(s) for s in str(pack_str).split("|") if s.strip().isdigit())
    except (ValueError, AttributeError):
        return 1


def _round_to_pack(qty: float, pack_size: int) -> int:
    if qty <= 0:
        return 0
    if pack_size <= 1:
        return int(np.ceil(qty))
    return int(np.ceil(qty / pack_size)) * pack_size


def _risk_level(row: pd.Series, lead_time_days: int, min_supply_days: int,
                target_supply_days: int) -> str:
    dos = row["days_of_supply"]
    if dos >= INFINITE_SUPPLY:
        return "LOW"
    if row["stockout_flag"] or dos < min_supply_days:
        return "HIGH"
    if dos < (lead_time_days + min_supply_days):
        # stock runs out before an order placed today would arrive
        return "HIGH"
    if dos < target_supply_days * 0.5:
        return "MEDIUM"
    return "LOW"


def _make_reason(row: pd.Series, target_supply_days: int) -> str:
    parts = []
    dos = row["days_of_supply"]
    if dos >= INFINITE_SUPPLY:
        parts.append("No demand forecast — verify stock level manually.")
    else:
        parts.append(f"~{dos:.0f} days of supply on hand.")
    if row["recommended_order_qty"] > 0:
        parts.append(
            f"Order {int(row['recommended_order_qty'])} units to reach "
            f"{target_supply_days}-day target."
        )
    if row["expiration_risk"]:
        parts.append("Warning: current stock may expire before it is dispensed.")
    if row["high_error_flag"]:
        parts.append(f"Low forecast confidence (MAPE {row['mape_pct']:.0f}%).")
    if abs(row["bias"]) > 2:
        direction = "over" if row["bias"] > 0 else "under"
        parts.append(
            f"Model tends to {direction}-forecast by {abs(row['bias']):.1f} units/wk "
            f"— order adjusted accordingly."
        )
    return " ".join(parts)


def compute_recommendations(
    inventory_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    drugs_df: pd.DataFrame,
    target_supply_days: int = DEFAULT_TARGET_SUPPLY_DAYS,
    lead_time_days: int     = DEFAULT_LEAD_TIME_DAYS,
    min_supply_days: int    = DEFAULT_MIN_SUPPLY_DAYS,
    as_of_date: date        = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per SKU, sorted by urgency.

    Parameters
    ----------
    inventory_df
        One row per NDC — the most recent inventory snapshot.
        Required columns: ndc, on_hand_units, dispensed_past_week,
                          expiry_date, stockout_flag
    forecasts_df
        One or more rows per NDC (weekly forecasts).
        Required columns: ndc, yhat
    metrics_df
        One row per NDC — per-SKU model accuracy.
        Required columns: ndc, mape_pct, bias, high_error_flag
    drugs_df
        One row per NDC — drug catalog.
        Required columns: ndc, drug_name, category, pack_sizes, unit_cost_usd
    target_supply_days
        Target days of inventory to maintain after ordering.
    lead_time_days
        Days from order placement to stock arriving on shelf.
    min_supply_days
        Days of supply below which stockout risk is HIGH.
    as_of_date
        Reference date for expiration calculations (defaults to today).
    """
    if as_of_date is None:
        as_of_date = date.today()
    today = pd.Timestamp(as_of_date)

    # Normalize NDC formats across all inputs
    for df in (inventory_df, forecasts_df, metrics_df, drugs_df):
        if "ndc" in df.columns:
            df["ndc"] = _normalize_ndc(df["ndc"])

    # Average weekly forecast demand per NDC
    avg_demand = (
        forecasts_df.groupby("ndc")["yhat"]
        .mean()
        .reset_index()
        .rename(columns={"yhat": "forecast_weekly_demand"})
    )

    # Build base table from drug catalog
    rec = (
        drugs_df[["ndc", "drug_name", "category", "pack_sizes", "unit_cost_usd"]]
        .copy()
        .merge(
            inventory_df[["ndc", "on_hand_units", "dispensed_past_week",
                           "expiry_date", "stockout_flag"]],
            on="ndc", how="left",
        )
        .merge(avg_demand, on="ndc", how="left")
        .merge(metrics_df[["ndc", "mape_pct", "bias", "high_error_flag"]], on="ndc", how="left")
    )

    # Fill missing values with safe defaults
    rec["on_hand_units"]          = pd.to_numeric(rec["on_hand_units"], errors="coerce").fillna(0)
    rec["dispensed_past_week"]    = pd.to_numeric(rec["dispensed_past_week"], errors="coerce").fillna(0)
    rec["forecast_weekly_demand"] = rec["forecast_weekly_demand"].fillna(rec["dispensed_past_week"])
    rec["mape_pct"]               = pd.to_numeric(rec["mape_pct"], errors="coerce").fillna(25.0)
    rec["bias"]                   = pd.to_numeric(rec["bias"], errors="coerce").fillna(0.0)
    rec["high_error_flag"]        = rec["high_error_flag"].fillna(False).astype(bool)
    rec["stockout_flag"]          = rec["stockout_flag"].fillna(False).astype(bool)
    rec["unit_cost_usd"]          = pd.to_numeric(rec["unit_cost_usd"], errors="coerce").fillna(0.0)

    daily_demand = rec["forecast_weekly_demand"] / 7

    # Days of supply
    rec["days_of_supply"] = np.where(
        daily_demand > 0,
        (rec["on_hand_units"] / daily_demand).round(1),
        float(INFINITE_SUPPLY),
    )

    # Safety stock: cover lead time + uncertainty buffer
    lead_demand      = daily_demand * lead_time_days
    uncertainty_buf  = lead_demand * (rec["mape_pct"] / 100)
    safety_stock     = lead_demand + uncertainty_buf

    # Bias correction: if model over-forecasts (+bias), reduce order slightly
    bias_adj = (-rec["bias"] * 0.5).clip(lower=-rec["on_hand_units"])

    # Target and raw order quantity
    target_stock = daily_demand * target_supply_days + safety_stock
    raw_order    = (target_stock - rec["on_hand_units"] + bias_adj).clip(lower=0)

    # Zero out if stock is already sufficient
    raw_order = np.where(rec["days_of_supply"] >= target_supply_days, 0, raw_order)

    # Round to pack size
    rec["pack_size"] = rec["pack_sizes"].apply(_smallest_pack_size)
    rec["recommended_order_qty"] = [
        _round_to_pack(q, p) for q, p in zip(raw_order, rec["pack_size"])
    ]

    # Order value
    rec["order_value_usd"] = (rec["recommended_order_qty"] * rec["unit_cost_usd"]).round(2)

    # Risk level
    rec["stockout_risk"] = rec.apply(
        _risk_level,
        axis=1,
        lead_time_days=lead_time_days,
        min_supply_days=min_supply_days,
        target_supply_days=target_supply_days,
    )

    # Expiration risk
    rec["expiry_date"]      = pd.to_datetime(rec["expiry_date"], errors="coerce")
    rec["days_until_expiry"] = (rec["expiry_date"] - today).dt.days
    rec["expiration_risk"]   = (
        rec["days_until_expiry"].notna() &
        (rec["days_until_expiry"] < rec["days_of_supply"])
    )

    # Human-readable reasoning
    rec["reasoning"] = rec.apply(_make_reason, axis=1, target_supply_days=target_supply_days)

    # Sort: HIGH first, then by days_of_supply ascending within each group
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rec["_risk_sort"] = rec["stockout_risk"].map(risk_order)
    rec = rec.sort_values(["_risk_sort", "days_of_supply"]).reset_index(drop=True)

    return rec[[
        "ndc", "drug_name", "category",
        "on_hand_units", "forecast_weekly_demand",
        "days_of_supply", "stockout_risk", "expiration_risk",
        "recommended_order_qty", "order_value_usd",
        "mape_pct", "bias", "high_error_flag",
        "reasoning",
    ]]
