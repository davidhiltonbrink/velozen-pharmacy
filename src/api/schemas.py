"""
Velozen AI — API request/response schemas.

Pydantic models define the exact JSON contract for every endpoint.
FastAPI uses these for automatic validation and /docs generation.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound (PMS → Velozen)
# ---------------------------------------------------------------------------

class InventorySnapshotIn(BaseModel):
    """One drug's current inventory state, as reported by the PMS."""
    ndc:                  str            = Field(..., description="11-digit NDC (hyphens optional)")
    on_hand_units:        float          = Field(..., ge=0)
    dispensed_past_week:  float          = Field(0.0, ge=0)
    expiry_date:          Optional[date] = None
    stockout_flag:        bool           = False
    inventory_value_usd:  Optional[float] = None


# ---------------------------------------------------------------------------
# Outbound (Velozen → PMS / client)
# ---------------------------------------------------------------------------

class RecommendationOut(BaseModel):
    ndc:                    str
    drug_name:              str
    category:               str
    on_hand_units:          float
    forecast_weekly_demand: float
    days_of_supply:         float
    stockout_risk:          str   = Field(..., description="HIGH | MEDIUM | LOW")
    expiration_risk:        bool
    recommended_order_qty:  int
    order_value_usd:        float
    mape_pct:               float = Field(..., description="Forecast MAPE % — higher = less confident")
    reasoning:              str


class RiskScoreOut(BaseModel):
    ndc:              str
    drug_name:        str
    category:         str
    days_of_supply:   float
    understock_score: float = Field(..., ge=0, le=100)
    overstock_score:  float = Field(..., ge=0, le=100)
    risk_label:       str   = Field(..., description="STOCKOUT_CRITICAL | STOCKOUT_WARNING | OVERSTOCK_CRITICAL | OVERSTOCK_WARNING | OK")


class ForecastOut(BaseModel):
    ndc:           str
    drug_name:     str
    category:      str
    forecast_week: date
    yhat:          float = Field(..., description="Predicted weekly fill count")
    y:             Optional[float] = Field(None, description="Actual fill count (if evaluation period)")


# ---------------------------------------------------------------------------
# Envelope wrappers
# ---------------------------------------------------------------------------

class RecommendationsResponse(BaseModel):
    pharmacy_id:    int
    as_of_date:     Optional[date]
    target_supply_days: int
    lead_time_days:     int
    items:          list[RecommendationOut]


class RiskScoresResponse(BaseModel):
    pharmacy_id: int
    as_of_date:  Optional[date]
    items:       list[RiskScoreOut]


class ForecastsResponse(BaseModel):
    pharmacy_id: int
    ndc:         Optional[str]
    items:       list[ForecastOut]


class InventoryPushResponse(BaseModel):
    pharmacy_id:    int
    rows_written:   int
    snapshot_date:  date
