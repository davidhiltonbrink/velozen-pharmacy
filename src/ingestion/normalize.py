"""
Velozen AI — Data normalization layer.

All data sources (synthetic, Synthea, CMS, PMS exports) must pass through
this module before reaching the model. Every source adapter returns a DataFrame
conforming to STANDARD_SCHEMA; build_training_set() combines them.

Standard schema
---------------
ds          : datetime  — week start date, always a Monday
ndc         : str       — 11-digit NDC (no hyphens)
drug_name   : str       — canonical drug name
category    : str       — "chronic" | "seasonal" | "other"
fills       : float     — 30-day-supply-equivalent fills for this week
source      : str       — origin tag ("synthetic", "synthea", "cms_medicaid", "pms_<name>")
population  : int       — patient population this data represents (for scale awareness)
weight      : float     — training sample weight; 1.0 = neutral, >1.0 = upweight this source
"""

from __future__ import annotations

import re
import pandas as pd

REQUIRED_COLS  = ["ds", "ndc", "drug_name", "category", "fills", "source", "population", "weight"]
VALID_CATEGORIES = {"chronic", "seasonal", "other"}


# ---------------------------------------------------------------------------
# NDC helpers
# ---------------------------------------------------------------------------

_NDC_STRIP = re.compile(r"[^0-9]")

def standardize_ndc(series: pd.Series) -> pd.Series:
    """
    Coerce NDC values to plain 11-digit strings (no hyphens, zero-padded).
    Handles common formats: 5-4-2, 5-3-2, 10-digit unformatted, etc.
    Returns the series unchanged if values are already 11 digits.
    """
    stripped = series.astype(str).str.replace(_NDC_STRIP, "", regex=True)
    # Pad to 11 digits (some sources omit leading zeros)
    return stripped.str.zfill(11)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def to_monday(series: pd.Series) -> pd.Series:
    """Snap any date to the Monday of its ISO week."""
    dates = pd.to_datetime(series)
    return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
    """
    Assert that df conforms to STANDARD_SCHEMA.
    Raises ValueError on any violation so bad data never silently reaches the model.
    Returns df unchanged on success.
    """
    tag = f"[{source_label}] " if source_label else ""

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{tag}Missing columns: {missing}")

    if df["ds"].isnull().any():
        raise ValueError(f"{tag}Null values in 'ds'")
    if df["ndc"].isnull().any() or (df["ndc"].str.len() != 11).any():
        raise ValueError(f"{tag}'ndc' must be non-null 11-digit strings")
    if df["fills"].isnull().any() or (df["fills"] < 0).any():
        raise ValueError(f"{tag}'fills' must be non-null and >= 0")

    bad_cats = set(df["category"].unique()) - VALID_CATEGORIES
    if bad_cats:
        raise ValueError(f"{tag}Invalid categories: {bad_cats}. Must be one of {VALID_CATEGORIES}")

    if (df["population"] <= 0).any():
        raise ValueError(f"{tag}'population' must be > 0")
    if (df["weight"] <= 0).any():
        raise ValueError(f"{tag}'weight' must be > 0")

    return df


# ---------------------------------------------------------------------------
# Combining sources
# ---------------------------------------------------------------------------

def build_training_set(
    sources: list[pd.DataFrame],
    source_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Concatenate normalized DataFrames from multiple sources into one training set.

    source_weights : optional dict mapping source tag → weight multiplier.
                     e.g. {"pms_pionerrx": 3.0, "synthetic": 0.5}
                     Overrides the per-row weight column for matching sources.

    Returns a DataFrame with REQUIRED_COLS ready to feed into the model.
    The 'weight' column is passed to LightGBM as sample_weight so higher-quality
    sources have proportionally more influence on the trained model.
    """
    if not sources:
        raise ValueError("No source DataFrames provided.")

    validated = []
    for df in sources:
        if source_weights:
            for src_tag, w in source_weights.items():
                mask = df["source"] == src_tag
                df.loc[mask, "weight"] = w
        validated.append(df[REQUIRED_COLS])

    combined = pd.concat(validated, ignore_index=True)
    combined["ds"] = pd.to_datetime(combined["ds"])
    combined = combined.sort_values(["source", "ndc", "ds"]).reset_index(drop=True)
    return combined
