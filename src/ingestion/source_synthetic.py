"""
Velozen AI — Synthetic data source adapter.

Reads the existing synthetic dispensing_records.csv + sku_catalog.csv and
returns a normalized DataFrame conforming to STANDARD_SCHEMA.
"""

from __future__ import annotations

import pandas as pd

from ingestion.normalize import standardize_ndc, to_monday, validate

POPULATION = 5_000  # approximate patient population the synthetic data represents


def load(
    disp_file: str,
    catalog_file: str,
    source: str = "synthetic",
    weight: float = 1.0,
) -> pd.DataFrame:
    """
    Load synthetic dispensing records and return a normalized weekly DataFrame.

    Parameters
    ----------
    disp_file    : path to dispensing_records.csv
    catalog_file : path to sku_catalog.csv
    source       : source tag written to the 'source' column
    weight       : base sample weight for this source (can be overridden in build_training_set)
    """
    raw     = pd.read_csv(disp_file, parse_dates=["date"])
    catalog = pd.read_csv(catalog_file)

    # Normalize NDC format
    raw["ndc"]  = standardize_ndc(raw["ndc"])
    catalog["ndc"] = standardize_ndc(catalog["ndc"])

    # Snap daily dates to Monday of each week
    raw["ds"] = to_monday(raw["date"])

    # Aggregate daily fills to weekly
    weekly = (
        raw.groupby(["ds", "ndc", "drug_name", "category"])
        .agg(fills=("rx_fill_count", "sum"))
        .reset_index()
    )

    # Attach metadata columns
    weekly["source"]     = source
    weekly["population"] = POPULATION
    weekly["weight"]     = weight

    return validate(weekly, source_label=source)
