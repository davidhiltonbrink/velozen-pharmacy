"""
Velozen AI — CMS Medicaid Drug Utilization source adapter.

Transforms CMS State Drug Utilization Data (SDUD) into normalized weekly
dispensing records conforming to STANDARD_SCHEMA.

Downloading CMS data
--------------------
Data is available at data.medicaid.gov. A downloader script is provided in
tools/download_cms_sd.py, or download manually:

  Dataset IDs (South Dakota):
    2024: 61729e5a-7aa8-448c-8903-ba3e0cd0ea3c
    2023: d890d3a9-6b00-43fd-8b31-fcba4c8e2909
    2022: 200c2cba-e58d-4a95-aa60-14b99736808d

  API: https://data.medicaid.gov/api/1/datastore/query/{id}/0
       ?conditions[0][property]=state&conditions[0][value]=SD&conditions[0][operator]=%3D

  Save combined CSV to: data/cms/sd_drug_utilization_2022_2024.csv

CMS CSV columns (actual)
------------------------
  utilization_type          : "FFSU" (fee-for-service) or "MCO" (managed care)
  state                     : two-letter state code
  ndc                       : 11-digit NDC (no hyphens)
  labeler_code              : first 5 digits of NDC
  product_code              : digits 6-9 of NDC
  package_size              : last 2 digits of NDC
  year                      : calendar year (string)
  quarter                   : 1–4 (string)
  suppression_used          : "true" if data suppressed (small counts — drop these)
  product_name              : drug name (may have trailing spaces; brand or generic)
  units_reimbursed          : total units dispensed (string float)
  number_of_prescriptions   : total Rx claims (string int)
  total_amount_reimbursed   : dollars (unused here)

Known limitations
-----------------
- Data is quarterly; this adapter distributes fills evenly across the 13 weeks
  of each quarter to produce weekly estimates.
- CMS data covers Medicaid patients only — roughly 20-25% of rural SD pharmacy
  volume. Weight is set lower than real PMS data to reflect partial coverage.
- product_name is not standardized — brand names, generics, and abbreviations
  are all mixed. We use keyword matching against the SKU catalog (same approach
  as source_synthea.py) to map to catalog NDCs.
- Suppressed rows (suppression_used == "true") are dropped — counts unreliable.

SD Medicaid enrollment (approximate, for population column)
------------------------------------------------------------
  2022: ~133,000 enrollees
  2023: ~150,000 enrollees
  2024: ~145,000 enrollees
"""

from __future__ import annotations

import re
import pandas as pd

from ingestion.normalize import standardize_ndc, to_monday, validate
from ingestion.source_synthea import _build_catalog_index, _match_description, _infer_category

WEEKS_PER_QUARTER = 13

_QUARTER_MONTH_START = {1: 1, 2: 4, 3: 7, 4: 10}

_SD_MEDICAID_POPULATION = {
    2022: 133_000,
    2023: 150_000,
    2024: 145_000,
}
_DEFAULT_POPULATION = 140_000


def load(
    utilization_csv: str,
    catalog_file: str,
    state: str = "SD",
    utilization_types: list[str] | None = None,
    catalog_only: bool = True,
    scale_to_population: int = 5_000,
    source: str = "cms_medicaid",
    weight: float = 0.6,
) -> pd.DataFrame:
    """
    Load CMS Medicaid Drug Utilization data and return normalized weekly records.

    Parameters
    ----------
    utilization_csv      : path to downloaded CMS state drug utilization CSV
    catalog_file         : path to VeloZen sku_catalog.csv (for NDC mapping)
    state                : two-letter state code to filter (default "SD")
    utilization_types    : ["FFSU", "MCO"] by default; pass ["FFSU"] for FFSU only
    catalog_only         : if True, drop drugs that didn't match a catalog NDC
    scale_to_population  : proportionally rescale fills from statewide Medicaid
                           volume (~140k SD enrollees) down to a single-pharmacy
                           patient base. Preserves demand patterns (which drugs
                           are popular, seasonal trends) while keeping fill
                           volumes in the same range as synthetic/Synthea sources.
                           Default 5,000 matches the synthetic data patient base.
    source               : source tag for the 'source' column
    weight               : base sample weight
    """
    if utilization_types is None:
        utilization_types = ["FFSU", "MCO"]

    df = pd.read_csv(utilization_csv, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Filter to target state and utilization types
    df = df[df["state"].str.upper() == state.upper()]
    df = df[df["utilization_type"].isin(utilization_types)]

    # Drop suppressed rows — counts are unreliable
    df = df[df["suppression_used"].str.lower() != "true"]

    # Parse numeric fields
    df["year"]    = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df["number_of_prescriptions"] = pd.to_numeric(
        df["number_of_prescriptions"], errors="coerce"
    ).fillna(0)

    df = df[df["number_of_prescriptions"] > 0].copy()

    # Standardize NDC
    df["ndc"] = standardize_ndc(df["ndc"])
    df["product_name"] = df["product_name"].str.strip()

    # Map drugs to catalog NDCs via keyword matching
    catalog = pd.read_csv(catalog_file)
    catalog["ndc"] = standardize_ndc(catalog["ndc"])
    catalog_index  = _build_catalog_index(catalog)

    unique_drugs = df[["ndc", "product_name"]].drop_duplicates("ndc")
    ndc_map: dict[str, dict] = {}
    for _, row in unique_drugs.iterrows():
        match = _match_description(str(row["product_name"]), catalog_index)
        if match:
            ndc_map[row["ndc"]] = {
                "ndc":       match["ndc"],
                "drug_name": match["drug_name"],
                "category":  match["category"],
                "catalog_hit": True,
            }
        else:
            ndc_map[row["ndc"]] = {
                "ndc":       row["ndc"],
                "drug_name": row["product_name"][:60],
                "category":  _infer_category(row["product_name"]),
                "catalog_hit": False,
            }

    df["mapped_ndc"]      = df["ndc"].map(lambda n: ndc_map[n]["ndc"])
    df["drug_name"]       = df["ndc"].map(lambda n: ndc_map[n]["drug_name"])
    df["category"]        = df["ndc"].map(lambda n: ndc_map[n]["category"])
    df["catalog_hit"]     = df["ndc"].map(lambda n: ndc_map[n]["catalog_hit"])

    if catalog_only:
        df = df[df["catalog_hit"]].copy()

    if df.empty:
        raise ValueError(f"No rows remain after filtering. Check state='{state}' and catalog_only={catalog_only}.")

    # Distribute quarterly prescriptions across WEEKS_PER_QUARTER weeks
    df["fills_per_week"] = df["number_of_prescriptions"] / WEEKS_PER_QUARTER

    # Vectorized week expansion — repeat each row WEEKS_PER_QUARTER times,
    # then compute the ds for each week offset.
    df["quarter_start"] = df.apply(
        lambda r: pd.Timestamp(year=r["year"], month=_QUARTER_MONTH_START[r["quarter"]], day=1),
        axis=1,
    )

    repeat_idx   = df.index.repeat(WEEKS_PER_QUARTER)
    week_offsets = list(range(WEEKS_PER_QUARTER)) * len(df)

    expanded = df.loc[repeat_idx, ["mapped_ndc", "drug_name", "category",
                                   "fills_per_week", "quarter_start", "year"]].copy()
    expanded = expanded.reset_index(drop=True)
    expanded["ds"] = expanded["quarter_start"] + pd.to_timedelta(
        [i * 7 for i in week_offsets], unit="D"
    )
    expanded["ds"] = to_monday(expanded["ds"])

    expanded = expanded.rename(columns={"mapped_ndc": "ndc"})

    weekly = (
        expanded
        .groupby(["ds", "ndc", "drug_name", "category"])
        .agg(fills=("fills_per_week", "sum"))
        .reset_index()
        .rename(columns={"fills_per_week": "fills"})
    )

    weekly["ds"]     = pd.to_datetime(weekly["ds"])
    weekly["source"] = source
    weekly["population"] = weekly["ds"].dt.year.map(
        lambda y: _SD_MEDICAID_POPULATION.get(y, _DEFAULT_POPULATION)
    )
    weekly["weight"] = weight

    # Scale fills proportionally from statewide Medicaid volume to single-pharmacy level.
    if scale_to_population:
        scale_factors = weekly["ds"].dt.year.map(
            lambda y: scale_to_population / _SD_MEDICAID_POPULATION.get(y, _DEFAULT_POPULATION)
        )
        weekly["fills"] = weekly["fills"] * scale_factors

    return validate(weekly, source_label=source)
