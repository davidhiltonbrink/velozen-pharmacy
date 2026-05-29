"""
Forecasting pipeline for Velozen AI — offline train/evaluate run.

Build sequence step 3: Core forecasting model pipeline.

Split strategy:
  Train  : 2024-01-01 to 2024-12-31  (52 weeks)
  Test   : 2025-01-01 to 2025-06-30  (26 weeks)

Outputs (written to data/synthetic/):
  forecasts.csv       - yhat for every SKU across the test window
  eval_metrics.csv    - per-SKU MAE, RMSE, MAPE, bias; flagged if MAPE > 25%

Usage:
  python src/models/train_and_evaluate.py
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.forecaster import GlobalForecaster, build_features, FEATURE_COLS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")
DISP_FILE    = os.path.join(DATA_DIR, "dispensing_records.csv")
CATALOG_FILE = os.path.join(DATA_DIR, "sku_catalog.csv")

TRAIN_END  = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END   = pd.Timestamp("2025-06-30")

HIGH_ERROR_MAPE = 25.0


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_weekly(disp_file: str) -> pd.DataFrame:
    df = pd.read_csv(disp_file, parse_dates=["date"])
    # Normalize to Monday of each week so dates align with pd.date_range(freq="7D")
    df["ds"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    return (
        df.groupby(["ds", "ndc", "drug_name", "category"])
        .agg(y=("rx_fill_count", "sum"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    nonzero = actual > 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any() else float("nan")
    )
    bias = float(np.mean(predicted - actual))
    return {
        "mae":      round(mae, 2),
        "rmse":     round(rmse, 2),
        "mape_pct": round(mape, 2),
        "bias":     round(bias, 2),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    print("Loading weekly dispensing data...")
    weekly  = load_weekly(DISP_FILE)
    catalog = pd.read_csv(CATALOG_FILE)

    train = weekly[weekly["ds"] <= TRAIN_END].copy()
    test  = weekly[(weekly["ds"] >= TEST_START) & (weekly["ds"] <= TEST_END)].copy()

    n_skus = train["ndc"].nunique()
    print(f"  {n_skus} SKUs | {train['ds'].nunique()} train weeks | {test['ds'].nunique()} test weeks")

    # ------------------------------------------------------------------
    # Train global model
    # ------------------------------------------------------------------
    print("\nTraining global LightGBM model...")
    forecaster = GlobalForecaster()
    forecaster.fit(train, catalog)
    print("  Done.")

    # ------------------------------------------------------------------
    # Recursive multi-step forecast over test window
    # ------------------------------------------------------------------
    horizon = test["ds"].nunique()
    print(f"\nForecasting {horizon} weeks ahead (recursive)...")
    forecasts = forecaster.predict(horizon_weeks=horizon)

    # Align forecast dates to test dates (both use W-MON anchor)
    forecasts["ds"] = pd.to_datetime(forecasts["ds"])
    test["ds"]      = pd.to_datetime(test["ds"])
    merged = test.merge(forecasts, on=["ds", "ndc", "drug_name"], how="inner")
    merged["yhat"] = merged["yhat"].clip(lower=0).round(1)
    print(f"  {len(merged):,} forecast rows matched to actuals.")

    # ------------------------------------------------------------------
    # Per-SKU metrics
    # ------------------------------------------------------------------
    print("\nComputing per-SKU metrics...")
    metrics_rows = []
    for ndc, grp in merged.groupby("ndc"):
        drug_name = grp["drug_name"].iloc[0]
        category  = grp["category"].iloc[0]
        m = compute_metrics(grp["y"].values, grp["yhat"].values)
        metrics_rows.append({
            "ndc":            ndc,
            "drug_name":      drug_name,
            "category":       category,
            "test_weeks":     len(grp),
            "high_error_flag": m["mape_pct"] > HIGH_ERROR_MAPE,
            **m,
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mape_pct", ascending=False)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    fi = forecaster.feature_importance()

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    fcst_path    = os.path.join(DATA_DIR, "forecasts.csv")
    metrics_path = os.path.join(DATA_DIR, "eval_metrics.csv")
    fi_path      = os.path.join(DATA_DIR, "feature_importance.csv")

    merged.to_csv(fcst_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    fi.to_csv(fi_path, index=False)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  SKUs evaluated     : {len(metrics_df)}")
    print(f"  Median MAPE        : {metrics_df['mape_pct'].median():.1f}%")
    print(f"  Mean MAPE          : {metrics_df['mape_pct'].mean():.1f}%")
    best_idx  = metrics_df["mape_pct"].idxmin()
    worst_idx = metrics_df["mape_pct"].idxmax()
    print(f"  Best MAPE          : {metrics_df.loc[best_idx,  'mape_pct']:.1f}%"
          f"  ({metrics_df.loc[best_idx,  'drug_name']})")
    print(f"  Worst MAPE         : {metrics_df.loc[worst_idx, 'mape_pct']:.1f}%"
          f"  ({metrics_df.loc[worst_idx, 'drug_name']})")
    print(f"  High-error SKUs    : {metrics_df['high_error_flag'].sum()} (>{HIGH_ERROR_MAPE}% MAPE)")

    print(f"\n  By category:")
    for cat, grp in metrics_df.groupby("category"):
        print(f"    {cat:10s}  median MAPE={grp['mape_pct'].median():.1f}%  n={len(grp)}")

    print(f"\n  Top 5 features by gain:")
    for _, row in fi.head(5).iterrows():
        print(f"    {row['feature']:25s}  {row['importance']:,.0f}")

    print(f"\n  Outputs:")
    print(f"    {fcst_path}")
    print(f"    {metrics_path}")
    print(f"    {fi_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
