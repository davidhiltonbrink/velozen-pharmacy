"""
Velozen AI — database seeder.

Loads existing synthetic CSVs into PostgreSQL. Safe to re-run:
tables are truncated before re-inserting, so you always get a
clean, consistent state.

Usage
-----
    python src/db/seed.py

What gets loaded
----------------
    sku_catalog.csv         → drugs
    forecasts.csv           → forecasts  (+ a default pharmacy row)
    eval_metrics.csv        → model_eval_metrics
    inventory_snapshots.csv → inventory_snapshots  (if file exists)
    dispensing_records.csv  → dispensing_records   (if file exists; large file)
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# Allow running as: python src/db/seed.py from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.connection import engine, get_session
from db.models import Base, Drug, Pharmacy, DispensingRecord, InventorySnapshot, Forecast, ModelEvalMetric

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")

MODEL_VERSION = "lgbm-v1-synthetic+synthea+cms"
DEFAULT_PHARMACY = dict(name="Aberdeen Family Pharmacy", city="Aberdeen", state="SD", zip_code="57401")


def _truncate(session, *models):
    for model in reversed(models):
        session.query(model).delete()
    session.commit()


def seed_drugs(session) -> int:
    path = os.path.join(DATA_DIR, "sku_catalog.csv")
    df = pd.read_csv(path, dtype=str)
    df["ndc"] = df["ndc"].str.replace("-", "").str.zfill(11)

    rows = []
    for _, r in df.iterrows():
        rows.append(Drug(
            ndc=r["ndc"],
            drug_name=r["drug_name"],
            category=r["category"],
            avg_daily_rx_fills=float(r.get("avg_daily_rx_fills", 0) or 0),
            pack_sizes=str(r.get("pack_sizes", "") or ""),
            unit_cost_usd=float(r.get("unit_cost_usd", 0) or 0),
            shelf_life_days=int(float(r.get("shelf_life_days", 0) or 0)),
        ))
    session.bulk_save_objects(rows)
    return len(rows)


def seed_pharmacy(session) -> int:
    existing = session.query(Pharmacy).filter_by(name=DEFAULT_PHARMACY["name"]).first()
    if not existing:
        session.add(Pharmacy(**DEFAULT_PHARMACY))
    return 1


def seed_forecasts(session, pharmacy_id: int) -> int:
    path = os.path.join(DATA_DIR, "forecasts.csv")
    if not os.path.isfile(path):
        print("  forecasts.csv not found — skipping")
        return 0
    df = pd.read_csv(path)
    df["ndc"] = df["ndc"].astype(str).str.zfill(11)
    df["ds"]  = pd.to_datetime(df["ds"]).dt.date

    rows = []
    for _, r in df.iterrows():
        rows.append(Forecast(
            pharmacy_id=pharmacy_id,
            ndc=r["ndc"],
            drug_name=r.get("drug_name"),
            category=r.get("category"),
            forecast_week=r["ds"],
            y=float(r["y"]) if pd.notna(r.get("y")) else None,
            yhat=float(r["yhat"]),
            model_version=MODEL_VERSION,
        ))
    session.bulk_save_objects(rows)
    return len(rows)


def seed_eval_metrics(session) -> int:
    path = os.path.join(DATA_DIR, "eval_metrics.csv")
    if not os.path.isfile(path):
        print("  eval_metrics.csv not found — skipping")
        return 0
    df = pd.read_csv(path)
    df["ndc"] = df["ndc"].astype(str).str.zfill(11)

    rows = []
    for _, r in df.iterrows():
        rows.append(ModelEvalMetric(
            model_version=MODEL_VERSION,
            ndc=r["ndc"],
            drug_name=r.get("drug_name"),
            category=r.get("category"),
            test_weeks=int(r.get("test_weeks", 0) or 0),
            mae=float(r.get("mae", 0) or 0),
            rmse=float(r.get("rmse", 0) or 0),
            mape_pct=float(r.get("mape_pct", 0) or 0),
            bias=float(r.get("bias", 0) or 0),
            high_error_flag=bool(r.get("high_error_flag", False)),
        ))
    session.bulk_save_objects(rows)
    return len(rows)


def seed_inventory(session, pharmacy_id: int) -> int:
    path = os.path.join(DATA_DIR, "inventory_snapshots.csv")
    if not os.path.isfile(path):
        print("  inventory_snapshots.csv not found (gitignored) — skipping")
        return 0

    df = pd.read_csv(path)
    df["ndc"] = df["ndc"].astype(str).str.replace("-", "").str.zfill(11)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date

    rows = []
    for _, r in df.iterrows():
        rows.append(InventorySnapshot(
            pharmacy_id=pharmacy_id,
            ndc=r["ndc"],
            snapshot_date=r["snapshot_date"],
            on_hand_units=float(r.get("on_hand_units", 0) or 0),
            dispensed_past_week=float(r.get("dispensed_past_week", 0) or 0),
            reorder_qty=float(r.get("reorder_qty", 0) or 0),
            expired_units=float(r.get("expired_units", 0) or 0),
            stockout_flag=bool(r.get("stockout_flag", False)),
            expiry_date=r.get("expiry_date"),
            inventory_value_usd=float(r.get("inventory_value_usd", 0) or 0),
        ))
    session.bulk_save_objects(rows)
    return len(rows)


def seed_dispensing(session, pharmacy_id: int) -> int:
    path = os.path.join(DATA_DIR, "dispensing_records.csv")
    if not os.path.isfile(path):
        print("  dispensing_records.csv not found (gitignored) — skipping")
        return 0

    print("  Loading dispensing_records.csv (large file — may take a moment)...")
    df = pd.read_csv(path)
    df["ndc"] = df["ndc"].astype(str).str.replace("-", "").str.zfill(11)

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if not date_col:
        print("  No date column found in dispensing_records.csv — skipping")
        return 0
    df["dispense_date"] = pd.to_datetime(df[date_col]).dt.date

    qty_col = next((c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower() or "fills" in c.lower() or "units" in c.lower()), None)
    if not qty_col:
        print("  No quantity column found in dispensing_records.csv — skipping")
        return 0

    rows = []
    for _, r in df.iterrows():
        rows.append(DispensingRecord(
            pharmacy_id=pharmacy_id,
            ndc=r["ndc"],
            dispense_date=r["dispense_date"],
            quantity=float(r[qty_col]),
            source="synthetic",
        ))
    session.bulk_save_objects(rows)
    return len(rows)


def run():
    print("Creating tables (if not exist)...")
    Base.metadata.create_all(engine)

    with get_session() as session:
        print("\nTruncating existing seed data...")
        _truncate(session, DispensingRecord, InventorySnapshot, Forecast, ModelEvalMetric, Drug, Pharmacy)

    with get_session() as session:
        print("\nSeeding drugs (sku_catalog)...")
        n = seed_drugs(session)
        print(f"  {n} drugs inserted")

    with get_session() as session:
        print("\nSeeding default pharmacy...")
        seed_pharmacy(session)

    with get_session() as session:
        pharmacy = session.query(Pharmacy).filter_by(name=DEFAULT_PHARMACY["name"]).first()
        pharmacy_id = pharmacy.id

    with get_session() as session:
        print("\nSeeding forecasts...")
        n = seed_forecasts(session, pharmacy_id)
        print(f"  {n} forecast rows inserted")

    with get_session() as session:
        print("\nSeeding eval metrics...")
        n = seed_eval_metrics(session)
        print(f"  {n} eval metric rows inserted")

    with get_session() as session:
        print("\nSeeding inventory snapshots...")
        n = seed_inventory(session, pharmacy_id)
        print(f"  {n} inventory rows inserted")

    with get_session() as session:
        print("\nSeeding dispensing records...")
        n = seed_dispensing(session, pharmacy_id)
        print(f"  {n} dispensing rows inserted")

    print("\nDone. Database is ready.")


if __name__ == "__main__":
    run()
