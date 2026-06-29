"""
Velozen AI — REST API

Run locally:
    uvicorn src.api.main:app --reload

Endpoints
---------
    GET  /v1/health
    GET  /v1/pharmacies/{pharmacy_id}/recommendations
    GET  /v1/pharmacies/{pharmacy_id}/risk-scores
    GET  /v1/pharmacies/{pharmacy_id}/forecasts
    POST /v1/pharmacies/{pharmacy_id}/inventory

Interactive docs (Swagger UI):
    http://localhost:8000/docs
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.auth import require_api_key
from api.schemas import (
    ForecastOut,
    ForecastsResponse,
    InventoryPushResponse,
    InventorySnapshotIn,
    RecommendationOut,
    RecommendationsResponse,
    RiskScoreOut,
    RiskScoresResponse,
)
from db.connection import get_session
from db.models import InventorySnapshot, Pharmacy
from models.recommender import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_TARGET_SUPPLY_DAYS,
    compute_recommendations,
)
from models.risk_scorer import compute_risk_scores

app = FastAPI(
    title="Velozen AI API",
    description="Pharmacy demand forecasting, ordering recommendations, and inventory risk scores.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific PMS domains before production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pharmacy_or_404(pharmacy_id: int) -> Pharmacy:
    with get_session() as session:
        pharmacy = session.get(Pharmacy, pharmacy_id)
    if pharmacy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Pharmacy {pharmacy_id} not found.")
    return pharmacy


def _load_rec_inputs(pharmacy_id: int):
    """Load inventory, forecasts, and metrics from DB for one pharmacy."""
    with get_session() as session:
        # Latest inventory snapshot per NDC
        inventory_df = pd.read_sql(
            "SELECT DISTINCT ON (ndc) ndc, snapshot_date, "
            "on_hand_units::float, dispensed_past_week::float, "
            "expiry_date, stockout_flag "
            "FROM inventory_snapshots WHERE pharmacy_id = %(pid)s "
            "ORDER BY ndc, snapshot_date DESC",
            session.bind,
            params={"pid": pharmacy_id},
            parse_dates=["snapshot_date", "expiry_date"],
        )
        forecasts_df = pd.read_sql(
            "SELECT ndc, yhat::float AS yhat FROM forecasts WHERE pharmacy_id = %(pid)s",
            session.bind,
            params={"pid": pharmacy_id},
        )
        metrics_df = pd.read_sql(
            "SELECT ndc, mape_pct::float, mae::float, rmse::float, "
            "bias::float, high_error_flag FROM model_eval_metrics",
            session.bind,
        )
        drugs_df = pd.read_sql(
            "SELECT ndc, drug_name, category, pack_sizes, "
            "unit_cost_usd::float FROM drugs",
            session.bind,
        )

    as_of = None
    if not inventory_df.empty and "snapshot_date" in inventory_df.columns:
        as_of = pd.to_datetime(inventory_df["snapshot_date"].max()).date()

    return inventory_df, forecasts_df, metrics_df, drugs_df, as_of


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.get(
    "/v1/pharmacies/{pharmacy_id}/recommendations",
    response_model=RecommendationsResponse,
    tags=["Recommendations"],
    dependencies=[Depends(require_api_key)],
)
def get_recommendations(
    pharmacy_id: int,
    target_supply_days: int = DEFAULT_TARGET_SUPPLY_DAYS,
    lead_time_days:     int = DEFAULT_LEAD_TIME_DAYS,
):
    """
    Ordered list of SKU-level recommendations for the pharmacy.
    Sorted by urgency: HIGH stockout risk first, then by days of supply ascending.
    """
    _get_pharmacy_or_404(pharmacy_id)
    inventory_df, forecasts_df, metrics_df, drugs_df, as_of = _load_rec_inputs(pharmacy_id)

    if inventory_df.empty:
        return RecommendationsResponse(
            pharmacy_id=pharmacy_id, as_of_date=as_of,
            target_supply_days=target_supply_days, lead_time_days=lead_time_days,
            items=[],
        )

    recs = compute_recommendations(
        inventory_df=inventory_df,
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        drugs_df=drugs_df,
        target_supply_days=target_supply_days,
        lead_time_days=lead_time_days,
        as_of_date=as_of,
    )

    items = [
        RecommendationOut(
            ndc=row.ndc,
            drug_name=row.drug_name,
            category=row.category,
            on_hand_units=row.on_hand_units,
            forecast_weekly_demand=round(row.forecast_weekly_demand, 2),
            days_of_supply=row.days_of_supply,
            stockout_risk=row.stockout_risk,
            expiration_risk=bool(row.expiration_risk),
            recommended_order_qty=int(row.recommended_order_qty),
            order_value_usd=row.order_value_usd,
            mape_pct=round(row.mape_pct, 1),
            reasoning=row.reasoning,
        )
        for row in recs.itertuples()
    ]

    return RecommendationsResponse(
        pharmacy_id=pharmacy_id,
        as_of_date=as_of,
        target_supply_days=target_supply_days,
        lead_time_days=lead_time_days,
        items=items,
    )


@app.get(
    "/v1/pharmacies/{pharmacy_id}/risk-scores",
    response_model=RiskScoresResponse,
    tags=["Risk"],
    dependencies=[Depends(require_api_key)],
)
def get_risk_scores(
    pharmacy_id: int,
    target_supply_days: int = DEFAULT_TARGET_SUPPLY_DAYS,
    lead_time_days:     int = DEFAULT_LEAD_TIME_DAYS,
):
    """
    Numeric understock (0–100) and overstock (0–100) scores for every SKU.
    """
    _get_pharmacy_or_404(pharmacy_id)
    inventory_df, forecasts_df, metrics_df, drugs_df, as_of = _load_rec_inputs(pharmacy_id)

    if inventory_df.empty:
        return RiskScoresResponse(pharmacy_id=pharmacy_id, as_of_date=as_of, items=[])

    recs = compute_recommendations(
        inventory_df=inventory_df,
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        drugs_df=drugs_df,
        target_supply_days=target_supply_days,
        lead_time_days=lead_time_days,
        as_of_date=as_of,
    )
    scored = compute_risk_scores(recs, target_supply_days=target_supply_days,
                                 lead_time_days=lead_time_days)

    items = [
        RiskScoreOut(
            ndc=row.ndc,
            drug_name=row.drug_name,
            category=row.category,
            days_of_supply=row.days_of_supply,
            understock_score=round(row.understock_score, 1),
            overstock_score=round(row.overstock_score, 1),
            risk_label=row.risk_label,
        )
        for row in scored.itertuples()
    ]

    return RiskScoresResponse(pharmacy_id=pharmacy_id, as_of_date=as_of, items=items)


@app.get(
    "/v1/pharmacies/{pharmacy_id}/forecasts",
    response_model=ForecastsResponse,
    tags=["Forecasts"],
    dependencies=[Depends(require_api_key)],
)
def get_forecasts(
    pharmacy_id: int,
    ndc: Optional[str] = None,
):
    """
    Demand forecasts for the pharmacy. Filter to a single NDC with ?ndc=...
    """
    _get_pharmacy_or_404(pharmacy_id)

    query = (
        "SELECT f.ndc, f.drug_name, f.category, "
        "f.forecast_week, f.yhat::float AS yhat, f.y::float AS y "
        "FROM forecasts f WHERE f.pharmacy_id = %(pid)s"
    )
    params: dict = {"pid": pharmacy_id}
    if ndc:
        query += " AND f.ndc = %(ndc)s"
        params["ndc"] = ndc.replace("-", "").zfill(11)

    with get_session() as session:
        df = pd.read_sql(query, session.bind, params=params,
                         parse_dates=["forecast_week"])

    items = [
        ForecastOut(
            ndc=row.ndc,
            drug_name=row.drug_name,
            category=row.category,
            forecast_week=row.forecast_week.date(),
            yhat=round(row.yhat, 2),
            y=round(row.y, 2) if row.y is not None and not pd.isna(row.y) else None,
        )
        for row in df.itertuples()
    ]

    return ForecastsResponse(pharmacy_id=pharmacy_id, ndc=ndc, items=items)


@app.post(
    "/v1/pharmacies/{pharmacy_id}/inventory",
    response_model=InventoryPushResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Inventory"],
    dependencies=[Depends(require_api_key)],
)
def push_inventory(
    pharmacy_id: int,
    snapshots: list[InventorySnapshotIn],
):
    """
    Accept a batch of current inventory snapshots from the PMS.
    Each call is treated as a new snapshot (today's date). Existing rows
    for the same pharmacy + NDC + date are replaced.
    """
    _get_pharmacy_or_404(pharmacy_id)

    today = date.today()

    with get_session() as session:
        # Remove any existing snapshots for this pharmacy + date to allow re-push
        session.query(InventorySnapshot).filter_by(
            pharmacy_id=pharmacy_id, snapshot_date=today
        ).delete()

        rows = []
        for snap in snapshots:
            ndc = snap.ndc.replace("-", "").zfill(11)
            rows.append(InventorySnapshot(
                pharmacy_id=pharmacy_id,
                ndc=ndc,
                snapshot_date=today,
                on_hand_units=snap.on_hand_units,
                dispensed_past_week=snap.dispensed_past_week,
                expiry_date=snap.expiry_date,
                stockout_flag=snap.stockout_flag,
                inventory_value_usd=snap.inventory_value_usd,
            ))

        session.bulk_save_objects(rows)

    return InventoryPushResponse(
        pharmacy_id=pharmacy_id,
        rows_written=len(rows),
        snapshot_date=today,
    )
