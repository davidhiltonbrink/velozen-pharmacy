"""
Velozen AI — Synthea source adapter.

Transforms Synthea's synthetic patient CSV output into normalized weekly
dispensing records conforming to STANDARD_SCHEMA.

Generating Synthea data
-----------------------
1. Download Synthea: https://github.com/synthetichealth/synthea/releases
2. Run for a South Dakota town:
     java -jar tools/synthea-with-dependencies.jar -p 2000 \\
       --exporter.csv.export=true --exporter.fhir.export=false \\
       --exporter.baseDirectory=data/synthea \\
       "South Dakota" "Aberdeen"
3. Pass data/synthea/csv/ to this adapter.

NDC mapping strategy
--------------------
Synthea uses RxNorm codes (CODE column), not NDC codes. Rather than requiring
an API call or full RxNorm release download, this adapter uses keyword matching
against the VeloZen SKU catalog to map Synthea drug descriptions to catalog NDCs.

  - Match found  → use catalog NDC + category (consistent across all sources)
  - No match     → use RxNorm code zero-padded to 11 digits as synthetic NDC,
                   infer category from drug name keywords

This keeps identifiers consistent across synthetic and Synthea sources for drugs
we already model, while still capturing new drugs Synthea introduces.
"""

from __future__ import annotations

import re
import pandas as pd

from ingestion.normalize import standardize_ndc, to_monday, validate

# Training window end — ongoing prescriptions (no STOP date) are capped here
_DEFAULT_HISTORY_END = pd.Timestamp("2025-12-31")

_CHRONIC_KEYWORDS = [
    "metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
    "losartan", "gabapentin", "levothyroxine", "metoprolol", "sertraline",
    "escitalopram", "bupropion", "hydrochlorothiazide", "furosemide",
    "carvedilol", "pantoprazole", "montelukast", "glipizide", "tamsulosin",
    "alendronate", "simvastatin", "rosuvastatin", "ezetimibe", "clopidogrel",
    "warfarin", "rivaroxaban", "apixaban", "spironolactone", "memantine",
    "donepezil", "insulin", "topiramate", "quetiapine", "aripiprazole",
    "duloxetine", "venlafaxine", "latanoprost", "timolol", "famotidine",
    "lactulose", "testosterone", "progesterone", "estradiol",
]
_SEASONAL_KEYWORDS = [
    "amoxicillin", "azithromycin", "albuterol", "oseltamivir", "cetirizine",
    "loratadine", "fexofenadine", "fluticasone", "prednisone", "doxycycline",
    "cephalexin", "trimethoprim", "sulfamethoxazole", "ciprofloxacin",
    "metronidazole", "clindamycin", "nitrofurantoin", "diphenhydramine",
    "chlorpheniramine",
]


_STOP_TOKENS = {
    "mg", "mcg", "ml", "tablet", "capsule", "solution", "oral", "injection",
    "inhaler", "spray", "cream", "gel", "susp", "tabs", "with", "plus",
    "extended", "release", "delayed", "immediate", "sustained", "chewable",
    "powder", "liquid", "syrup", "drop", "patch", "film", "pack", "dose",
}


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase alpha-only tokens of length >= 4."""
    tokens = re.sub(r"[^a-z ]", " ", text.lower()).split()
    return {t for t in tokens if len(t) >= 4 and t not in _STOP_TOKENS}


def _build_catalog_index(catalog: pd.DataFrame) -> list[dict]:
    """
    Build a keyword lookup list from the SKU catalog.
    Each entry maps a set of meaningful name tokens → (ndc, drug_name, category).
    Uses token-level matching (not substring) to avoid false positives on dose
    fragments like 'er', '05', 'susp' matching unrelated drugs.
    """
    index = []
    for _, row in catalog.iterrows():
        keywords = _tokenize(row["drug_name"])
        if not keywords:
            continue
        index.append({
            "keywords": keywords,
            "ndc":       row["ndc"],
            "drug_name": row["drug_name"],
            "category":  row["category"],
        })
    return index


def _match_description(description: str, catalog_index: list[dict]) -> dict | None:
    """
    Return the best catalog match for a Synthea drug description, or None.
    Scoring: token-level intersection between catalog keywords and description tokens.
    Requires at least 1 matching token.
    """
    desc_tokens = _tokenize(description)
    best_score, best_entry = 0, None
    for entry in catalog_index:
        score = len(entry["keywords"] & desc_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry if best_score >= 1 else None


def _infer_category(description: str) -> str:
    desc = description.lower()
    if any(kw in desc for kw in _CHRONIC_KEYWORDS):
        return "chronic"
    if any(kw in desc for kw in _SEASONAL_KEYWORDS):
        return "seasonal"
    return "other"


def _rxnorm_to_synthetic_ndc(code: int) -> str:
    """Pad RxNorm code to 11 digits for use as a synthetic NDC."""
    return str(int(code)).zfill(11)


_DEFAULT_HISTORY_START = pd.Timestamp("2020-01-01")


def load(
    synthea_csv_dir: str,
    catalog_file: str,
    history_start: pd.Timestamp = _DEFAULT_HISTORY_START,
    history_end: pd.Timestamp = _DEFAULT_HISTORY_END,
    catalog_only: bool = True,
    source: str = "synthea",
    weight: float = 0.8,
) -> pd.DataFrame:
    """
    Load Synthea medications.csv and return normalized weekly dispensing records.

    Parameters
    ----------
    synthea_csv_dir : path to Synthea output/csv/ directory
    catalog_file    : path to VeloZen sku_catalog.csv (for NDC mapping)
    history_start   : exclude prescription activity before this date
    history_end     : ongoing prescriptions (no STOP) are capped at this date
    catalog_only    : if True, drop drugs that didn't match a catalog NDC.
                      Recommended: filters out hospital/specialty drugs (chemo,
                      anesthetics, injectables) that retail pharmacies don't stock.
    source          : source tag for the 'source' column
    weight          : base sample weight (slightly below real PMS data)
    """
    meds    = pd.read_csv(f"{synthea_csv_dir}/medications.csv")
    patients = pd.read_csv(f"{synthea_csv_dir}/patients.csv")
    catalog  = pd.read_csv(catalog_file)

    catalog["ndc"] = standardize_ndc(catalog["ndc"])
    catalog_index  = _build_catalog_index(catalog)

    # Parse timestamps; coerce bad values to NaT
    meds["START"] = pd.to_datetime(meds["START"], errors="coerce", utc=True).dt.tz_localize(None)
    meds["STOP"]  = pd.to_datetime(meds["STOP"],  errors="coerce", utc=True).dt.tz_localize(None)

    # Cap ongoing prescriptions at history_end; floor at history_start
    meds["STOP"]  = meds["STOP"].fillna(history_end).clip(upper=history_end)
    meds["START"] = meds["START"].clip(lower=history_start)

    # Drop rows outside the training window or with zero dispenses
    meds = meds.dropna(subset=["START"])
    meds = meds[meds["START"] <= history_end]
    meds = meds[meds["STOP"]  >= history_start]
    meds = meds[meds["DISPENSES"] > 0].copy()

    # Only keep patients whose records we want (all alive + deceased within window)
    valid_patients = set(patients["Id"])
    meds = meds[meds["PATIENT"].isin(valid_patients)]

    population_size = len(valid_patients)

    # Build per-RxNorm-code mapping (one lookup per unique drug, not per row)
    unique_drugs = meds[["CODE", "DESCRIPTION"]].drop_duplicates("CODE")
    code_map: dict[int, dict] = {}
    for _, row in unique_drugs.iterrows():
        code = int(row["CODE"])
        desc = str(row["DESCRIPTION"])
        match = _match_description(desc, catalog_index)
        if match:
            code_map[code] = {"ndc": match["ndc"], "drug_name": match["drug_name"],
                              "category": match["category"]}
        else:
            code_map[code] = {"ndc": _rxnorm_to_synthetic_ndc(code),
                              "drug_name": desc[:60], "category": _infer_category(desc)}

    # Attach mapped columns to meds (vectorized)
    meds["ndc"]        = meds["CODE"].map(lambda c: code_map[int(c)]["ndc"])
    meds["drug_name"]  = meds["CODE"].map(lambda c: code_map[int(c)]["drug_name"])
    meds["category"]   = meds["CODE"].map(lambda c: code_map[int(c)]["category"])
    meds["catalog_hit"] = meds["CODE"].map(
        lambda c: code_map[int(c)]["ndc"] != _rxnorm_to_synthetic_ndc(int(c))
    )

    if catalog_only:
        meds = meds[meds["catalog_hit"]].copy()

    # Snap dates to Monday of each week
    meds["week_start"] = to_monday(meds["START"])
    meds["week_end"]   = to_monday(meds["STOP"])

    # Number of weeks in each prescription window
    meds["n_weeks"] = (
        ((meds["week_end"] - meds["week_start"]).dt.days // 7 + 1).clip(lower=1)
    )
    meds["fills_per_week"] = meds["DISPENSES"] / meds["n_weeks"]

    # Expand each prescription into one row per week using vectorized repeat + date offsets
    week_counts  = meds["n_weeks"].values
    repeat_idx   = meds.index.repeat(week_counts)
    week_offsets = pd.array(
        [i for count in week_counts for i in range(count)], dtype="int64"
    )

    expanded = meds.loc[repeat_idx, ["ndc", "drug_name", "category", "fills_per_week", "week_start"]].copy()
    expanded = expanded.reset_index(drop=True)
    expanded["ds"] = expanded["week_start"] + pd.to_timedelta(week_offsets * 7, unit="D")
    expanded = expanded.rename(columns={"fills_per_week": "fills"})

    weekly = (
        expanded
        .groupby(["ds", "ndc", "drug_name", "category"])
        .agg(fills=("fills", "sum"))
        .reset_index()
    )

    weekly["ds"]         = pd.to_datetime(weekly["ds"])
    weekly["source"]     = source
    weekly["population"] = population_size
    weekly["weight"]     = weight

    return validate(weekly, source_label=source)
