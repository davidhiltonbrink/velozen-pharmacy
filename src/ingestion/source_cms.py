"""
Velozen AI — CMS Medicaid Drug Utilization source adapter.

Transforms CMS Medicaid Drug Utilization data into normalized weekly dispensing
records conforming to STANDARD_SCHEMA.

Obtaining CMS data
------------------
1. Navigate to: https://data.medicaid.gov/datasets?theme=Drug%20Utilization
2. Download "State Drug Utilization Data" for the target year(s).
   Direct link pattern: https://data.medicaid.gov/api/1/datastore/query/<dataset-id>
3. Filter for SD (state = "SD") to reduce file size before download, or filter
   in this adapter using the `state` parameter.

CMS Medicaid Drug Utilization CSV columns (relevant subset)
------------------------------------------------------------
  utilization_type          : "FFSU" (fee-for-service) or "MCO" (managed care)
  state                     : two-letter state code
  labeler_code              : first 5 digits of NDC
  product_code              : digits 6-9 of NDC
  package_size_code         : last 2 digits of NDC
  year                      : calendar year
  quarter                   : 1–4
  product_name              : drug name (brand or generic, varies)
  units_reimbursed          : total units dispensed
  number_of_prescriptions   : total Rx claims
  suppression_used          : "true" if data suppressed for privacy (small counts)

Known transform challenges
--------------------------
- Data is quarterly; this adapter distributes fills evenly across the 13 weeks
  of each quarter to produce weekly estimates.
- NDC must be assembled from labeler_code + product_code + package_size_code
  and zero-padded to 11 digits.
- CMS data is at the state Medicaid population level — not a single pharmacy.
  The `population` column is set to the SD Medicaid enrollment for the year,
  which is used downstream to scale fills to a single-pharmacy level if needed.
- `suppression_used = "true"` rows contain unreliable counts and are dropped.
- CMS data covers Medicaid patients only (~20% of rural pharmacy volume).
  Weight is set lower than real PMS data to reflect this partial coverage.

SD Medicaid enrollment reference (approximate)
-----------------------------------------------
  2022: ~133,000 enrollees
  2023: ~150,000 enrollees
  2024: ~145,000 enrollees
"""

from __future__ import annotations

import pandas as pd

from ingestion.normalize import standardize_ndc, to_monday, validate

# Approximate SD Medicaid enrollment by year — update as needed
_SD_MEDICAID_POPULATION = {
    2022: 133_000,
    2023: 150_000,
    2024: 145_000,
}
_DEFAULT_POPULATION = 140_000

_QUARTER_TO_MONTH_START = {1: 1, 2: 4, 3: 7, 4: 10}
WEEKS_PER_QUARTER = 13


def load(
    utilization_csv: str,
    state: str = "SD",
    years: list[int] | None = None,
    source: str = "cms_medicaid",
    weight: float = 0.6,
) -> pd.DataFrame:
    """
    Transform CMS Medicaid Drug Utilization data into normalized weekly dispensing records.

    Parameters
    ----------
    utilization_csv : path to downloaded CMS state drug utilization CSV
    state           : two-letter state code to filter (default "SD")
    years           : list of years to include; None = all years in file
    source          : source tag for the 'source' column
    weight          : base sample weight (lower than real PMS data — Medicaid
                      only, quarterly estimates distributed to weekly)
    """
    raise NotImplementedError(
        "CMS Medicaid adapter not yet implemented.\n"
        "Steps to enable:\n"
        "  1. Download CMS State Drug Utilization Data from data.medicaid.gov.\n"
        "  2. Pass the CSV path to this function.\n"
        "  3. Implement the transform below (see module docstring for approach).\n"
    )

    # --- Implementation outline (fill in when CMS data is downloaded) ---
    #
    # df = pd.read_csv(utilization_csv, dtype=str)
    # df.columns = df.columns.str.lower().str.replace(" ", "_")
    #
    # # Filter
    # df = df[df["state"] == state]
    # df = df[df["suppression_used"].str.lower() != "true"]
    # if years:
    #     df = df[df["year"].astype(int).isin(years)]
    #
    # # Assemble 11-digit NDC
    # df["ndc"] = standardize_ndc(df["labeler_code"] + df["product_code"] + df["package_size_code"])
    #
    # # Distribute quarterly prescriptions across 13 weeks
    # df["year"]    = df["year"].astype(int)
    # df["quarter"] = df["quarter"].astype(int)
    # df["number_of_prescriptions"] = pd.to_numeric(df["number_of_prescriptions"], errors="coerce").fillna(0)
    # df["fills_per_week"] = df["number_of_prescriptions"] / WEEKS_PER_QUARTER
    #
    # rows = []
    # for _, row in df.iterrows():
    #     month_start = _QUARTER_TO_MONTH_START[row["quarter"]]
    #     qstart = pd.Timestamp(year=row["year"], month=month_start, day=1)
    #     weeks  = pd.date_range(qstart, periods=WEEKS_PER_QUARTER, freq="7D")
    #     for w in weeks:
    #         rows.append({
    #             "ds":        to_monday(pd.Series([w]))[0],
    #             "ndc":       row["ndc"],
    #             "drug_name": row["product_name"],
    #             "fills":     row["fills_per_week"],
    #         })
    #
    # weekly = pd.DataFrame(rows)
    # weekly = _infer_category(weekly)
    # weekly["source"]     = source
    # weekly["population"] = weekly.apply(
    #     lambda r: _SD_MEDICAID_POPULATION.get(r["ds"].year, _DEFAULT_POPULATION), axis=1
    # )
    # weekly["weight"] = weight
    # return validate(weekly, source_label=source)


def _infer_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer drug category from drug name when source data doesn't provide it.
    Extend keyword lists as needed.
    """
    chronic_keywords  = ["metformin", "lisinopril", "atorvastatin", "amlodipine",
                         "omeprazole", "metoprolol", "losartan", "simvastatin",
                         "levothyroxine", "gabapentin", "sertraline", "fluoxetine"]
    seasonal_keywords = ["amoxicillin", "azithromycin", "albuterol", "oseltamivir",
                         "cetirizine", "loratadine", "fluticasone", "prednisone"]

    name = df["drug_name"].str.lower()
    df["category"] = "other"
    df.loc[name.str.contains("|".join(seasonal_keywords), na=False), "category"] = "seasonal"
    df.loc[name.str.contains("|".join(chronic_keywords),  na=False), "category"] = "chronic"
    return df
