"""
Velozen AI — Synthea source adapter.

Transforms Synthea's synthetic patient output into normalized weekly dispensing
records conforming to STANDARD_SCHEMA.

Generating Synthea data
-----------------------
1. Download Synthea: https://github.com/synthetichealth/synthea
2. Run for a South Dakota town, e.g. Aberdeen (pop ~28,000):
     java -jar synthea-with-dependencies.jar -p 5000 "South Dakota" "Aberdeen"
   or for a smaller rural town:
     java -jar synthea-with-dependencies.jar -p 800 "South Dakota" "Huron"
3. Point this adapter at the output/csv/ directory.

Relevant Synthea CSV files
--------------------------
medications.csv columns:
  START, STOP, PATIENT, PAYER, PAYER_COVERAGE, DISPENSES, TOTALCOST,
  REASONCODE, REASONDESCRIPTION, CODE (RxNorm), DESCRIPTION, BASE_COST

patients.csv columns:
  Id, BIRTHDATE, DEATHDATE, SSN, DRIVERS, PASSPORT, PREFIX, FIRST, LAST,
  SUFFIX, MAIDEN, MARITAL, RACE, ETHNICITY, GENDER, BIRTHPLACE, ADDRESS,
  CITY, STATE, COUNTY, FIPS, ZIP, LAT, LON, HEALTHCARE_EXPENSES,
  HEALTHCARE_COVERAGE, INCOME

Known transform challenges
--------------------------
- Synthea uses RxNorm codes (CODE column), not NDC codes. A RxNorm → NDC
  mapping table is required. The RxNorm API (rxnav.nlm.nih.gov) or the
  full RxNorm release file can provide this mapping.
- DISPENSES counts whole prescription fills. Each fill is assumed to be a
  30-day supply for this adapter (standard US retail pharmacy default).
- START/STOP define the prescription period. Dispenses are distributed evenly
  across weeks within that window.
- Synthea models prescribing patterns (what doctors order), not actual
  pharmacy dispensing. Adherence gaps are not modeled — adjust weight
  downward relative to real PMS data to account for this.
"""

from __future__ import annotations

import pandas as pd

from ingestion.normalize import standardize_ndc, to_monday, validate, VALID_CATEGORIES


def load(
    medications_csv: str,
    patients_csv: str,
    rxnorm_to_ndc_csv: str,
    population_size: int,
    state: str = "SD",
    source: str = "synthea",
    weight: float = 0.8,
) -> pd.DataFrame:
    """
    Transform Synthea medications.csv into normalized weekly dispensing records.

    Parameters
    ----------
    medications_csv    : path to Synthea medications.csv
    patients_csv       : path to Synthea patients.csv
    rxnorm_to_ndc_csv  : path to RxNorm→NDC mapping table
                         Expected columns: rxnorm_code, ndc, drug_name, category
    population_size    : number of patients simulated (used in 'population' column)
    state              : two-letter state code to filter patients (default "SD")
    source             : source tag for the 'source' column
    weight             : base sample weight (lower than real PMS data; Synthea
                         doesn't model non-adherence or partial fills)
    """
    raise NotImplementedError(
        "Synthea adapter not yet implemented.\n"
        "Steps to enable:\n"
        "  1. Run Synthea to generate output/csv/ for target SD population.\n"
        "  2. Download RxNorm full release or use rxnav.nlm.nih.gov API to build\n"
        "     rxnorm_to_ndc_csv mapping table.\n"
        "  3. Implement the transform below (see module docstring for approach).\n"
    )

    # --- Implementation outline (fill in when Synthea data is available) ---
    #
    # meds     = pd.read_csv(medications_csv, parse_dates=["START", "STOP"])
    # patients = pd.read_csv(patients_csv)
    # mapping  = pd.read_csv(rxnorm_to_ndc_csv, dtype={"rxnorm_code": str, "ndc": str})
    #
    # # Filter to target state
    # sd_patients = patients[patients["STATE"] == state]["Id"]
    # meds = meds[meds["PATIENT"].isin(sd_patients)]
    #
    # # Map RxNorm → NDC
    # meds = meds.merge(mapping, left_on="CODE", right_on="rxnorm_code", how="inner")
    # meds["ndc"] = standardize_ndc(meds["ndc"])
    #
    # # Distribute dispenses across weeks in the prescription window
    # rows = []
    # for _, row in meds.iterrows():
    #     weeks = pd.date_range(row["START"], row["STOP"], freq="7D")
    #     fills_per_week = row["DISPENSES"] / max(len(weeks), 1)
    #     for w in weeks:
    #         rows.append({"ds": to_monday(pd.Series([w]))[0], "ndc": row["ndc"],
    #                      "drug_name": row["drug_name"], "category": row["category"],
    #                      "fills": fills_per_week})
    #
    # weekly = (
    #     pd.DataFrame(rows)
    #     .groupby(["ds", "ndc", "drug_name", "category"])
    #     .agg(fills=("fills", "sum"))
    #     .reset_index()
    # )
    # weekly["source"]     = source
    # weekly["population"] = population_size
    # weekly["weight"]     = weight
    # return validate(weekly, source_label=source)
