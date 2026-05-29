"""
Synthetic pharmacy dispensing data generator for Velozen AI.

Generates ~2 years of daily dispensing records across ~150 SKUs for a single
rural pharmacy, with realistic demand patterns baked in:
  - Weekday vs weekend drop-off
  - Payday spikes (1st and 15th of month)
  - Seasonal illness curves (flu, allergy, etc.)
  - Chronic-med steady baselines with slow demographic drift
  - Occasional stockouts and over-order events
  - Inventory on-hand and expiry date tracking

Output: data/synthetic/dispensing_records.csv
         data/synthetic/inventory_snapshots.csv
         data/synthetic/sku_catalog.csv
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# SKU Catalog — 150 NDC-like codes across drug categories
# ---------------------------------------------------------------------------

CHRONIC_MEDS = [
    # (drug_name, ndc_stub, avg_daily_units, unit_size_options)
    ("Metformin 500mg",       "57237-0001", 18, [30, 90]),
    ("Metformin 1000mg",      "57237-0002", 12, [30, 90]),
    ("Lisinopril 10mg",       "57237-0011", 15, [30, 90]),
    ("Lisinopril 20mg",       "57237-0012", 10, [30, 90]),
    ("Atorvastatin 20mg",     "57237-0021", 14, [30, 90]),
    ("Atorvastatin 40mg",     "57237-0022", 11, [30, 90]),
    ("Amlodipine 5mg",        "57237-0031", 13, [30, 90]),
    ("Amlodipine 10mg",       "57237-0032",  8, [30, 90]),
    ("Omeprazole 20mg",       "57237-0041", 16, [30, 90]),
    ("Omeprazole 40mg",       "57237-0042",  9, [30, 90]),
    ("Losartan 50mg",         "57237-0051", 11, [30, 90]),
    ("Losartan 100mg",        "57237-0052",  7, [30, 90]),
    ("Gabapentin 300mg",      "57237-0061", 10, [90]),
    ("Gabapentin 600mg",      "57237-0062",  7, [90]),
    ("Levothyroxine 50mcg",   "57237-0071", 12, [30, 90]),
    ("Levothyroxine 100mcg",  "57237-0072",  9, [30, 90]),
    ("Metoprolol 25mg",       "57237-0081", 11, [30, 90]),
    ("Metoprolol 50mg",       "57237-0082",  9, [30, 90]),
    ("Sertraline 50mg",       "57237-0091", 10, [30, 90]),
    ("Sertraline 100mg",      "57237-0092",  7, [30, 90]),
    ("Escitalopram 10mg",     "57237-0101",  9, [30, 90]),
    ("Bupropion SR 150mg",    "57237-0111",  8, [30, 90]),
    ("Hydrochlorothiazide 25mg", "57237-0121", 10, [30, 90]),
    ("Furosemide 40mg",       "57237-0131",  8, [30, 90]),
    ("Carvedilol 6.25mg",     "57237-0141",  7, [60, 90]),
    ("Pantoprazole 40mg",     "57237-0151", 10, [30, 90]),
    ("Montelukast 10mg",      "57237-0161",  9, [30, 90]),
    ("Glipizide 5mg",         "57237-0171",  7, [30, 90]),
    ("Tamsulosin 0.4mg",      "57237-0181",  6, [30, 90]),
    ("Alendronate 70mg",      "57237-0191",  4, [4, 12]),
]

SEASONAL_MEDS = [
    ("Amoxicillin 500mg",     "57237-0201", 6,  [10, 14, 21]),
    ("Amoxicillin 875mg",     "57237-0202", 4,  [10, 14]),
    ("Azithromycin 250mg",    "57237-0211", 3,  [6]),
    ("Albuterol Inhaler",     "57237-0221", 8,  [1]),
    ("Oseltamivir 75mg",      "57237-0231", 2,  [10]),
    ("Cetirizine 10mg",       "57237-0241", 9,  [30, 90]),
    ("Loratadine 10mg",       "57237-0242", 8,  [30, 90]),
    ("Fexofenadine 180mg",    "57237-0243", 6,  [30, 90]),
    ("Fluticasone Nasal 50mcg","57237-0251", 5, [1]),
    ("Prednisone 20mg",       "57237-0261", 4,  [5, 7, 10]),
    ("Doxycycline 100mg",     "57237-0271", 3,  [14, 21]),
    ("Cephalexin 500mg",      "57237-0281", 4,  [10, 14]),
    ("Trimethoprim/Sulfa DS", "57237-0291", 3,  [10, 14]),
    ("Promethazine 25mg",     "57237-0301", 3,  [12, 20]),
    ("Ondansetron 4mg",       "57237-0311", 4,  [9, 18]),
]

OTHER_MEDS = [
    ("Ibuprofen 600mg",       "57237-0401", 8,  [30, 60]),
    ("Ibuprofen 800mg",       "57237-0402", 6,  [30, 60]),
    ("Cyclobenzaprine 10mg",  "57237-0411", 5,  [15, 30]),
    ("Tramadol 50mg",         "57237-0421", 5,  [30, 60]),
    ("Oxycodone/APAP 5/325mg","57237-0431", 3,  [20, 30]),
    ("Methylprednisolone 4mg","57237-0441", 4,  [21]),
    ("Clonazepam 0.5mg",      "57237-0451", 4,  [30, 90]),
    ("Alprazolam 0.5mg",      "57237-0461", 3,  [30, 90]),
    ("Zolpidem 10mg",         "57237-0471", 4,  [30]),
    ("Hydroxyzine 25mg",      "57237-0481", 5,  [30, 60]),
    ("Linaclotide 145mcg",    "57237-0491", 3,  [30, 90]),
    ("Insulin Glargine U-100","57237-0501", 4,  [1, 3]),
    ("Insulin Aspart U-100",  "57237-0511", 3,  [1, 3]),
    ("Methotrexate 2.5mg",    "57237-0521", 2,  [4, 8]),
    ("Allopurinol 300mg",     "57237-0531", 5,  [30, 90]),
    ("Colchicine 0.6mg",      "57237-0541", 4,  [30, 60]),
    ("Finasteride 5mg",       "57237-0551", 4,  [30, 90]),
    ("Sildenafil 50mg",       "57237-0561", 3,  [6, 12, 30]),
    ("Topiramate 25mg",       "57237-0571", 3,  [30, 90]),
    ("Quetiapine 25mg",       "57237-0581", 4,  [30, 90]),
    ("Aripiprazole 10mg",     "57237-0591", 3,  [30]),
    ("Duloxetine 30mg",       "57237-0601", 5,  [30, 90]),
    ("Venlafaxine ER 75mg",   "57237-0611", 5,  [30, 90]),
    ("Memantine 10mg",        "57237-0621", 3,  [30, 60]),
    ("Donepezil 10mg",        "57237-0631", 4,  [30]),
    ("Rosuvastatin 20mg",     "57237-0641", 8,  [30, 90]),
    ("Ezetimibe 10mg",        "57237-0651", 5,  [30, 90]),
    ("Clopidogrel 75mg",      "57237-0661", 6,  [30, 90]),
    ("Warfarin 5mg",          "57237-0671", 5,  [30, 90]),
    ("Rivaroxaban 20mg",      "57237-0681", 4,  [30, 90]),
    ("Apixaban 5mg",          "57237-0691", 4,  [60]),
    ("Spironolactone 25mg",   "57237-0701", 4,  [30, 90]),
    ("Clindamycin 300mg",     "57237-0711", 3,  [21, 28]),
    ("Nitrofurantoin 100mg",  "57237-0721", 3,  [7, 14]),
    ("Ciprofloxacin 500mg",   "57237-0731", 3,  [10, 14]),
    ("Metronidazole 500mg",   "57237-0741", 3,  [14, 21]),
    ("Valacyclovir 500mg",    "57237-0751", 3,  [7, 30]),
    ("Acyclovir 400mg",       "57237-0761", 3,  [7, 30]),
    ("Fluconazole 150mg",     "57237-0771", 2,  [1]),
    ("Nystatin Oral Susp",    "57237-0781", 2,  [1]),
    ("Latanoprost 0.005%",    "57237-0791", 3,  [1]),
    ("Timolol 0.5% Eye",      "57237-0801", 2,  [1]),
    ("Metoclopramide 10mg",   "57237-0811", 3,  [30]),
    ("Dicyclomine 20mg",      "57237-0821", 4,  [30, 60]),
    ("Famotidine 40mg",       "57237-0831", 5,  [30, 90]),
    ("Lactulose Solution",    "57237-0841", 2,  [1]),
    ("Polyethylene Glycol 3350","57237-0851", 4, [1]),
    ("Testosterone Cypionate","57237-0861", 2,  [1]),
    ("Progesterone 200mg",    "57237-0871", 3,  [30]),
    ("Estradiol 1mg",         "57237-0881", 4,  [30, 90]),
    ("Norethindrone/EE",      "57237-0891", 5,  [28]),
]

ALL_MEDS = CHRONIC_MEDS + SEASONAL_MEDS + OTHER_MEDS


def build_sku_catalog() -> pd.DataFrame:
    rows = []
    for category, med_list in [
        ("chronic", CHRONIC_MEDS),
        ("seasonal", SEASONAL_MEDS),
        ("other", OTHER_MEDS),
    ]:
        for drug_name, ndc_stub, avg_daily, pack_sizes in med_list:
            rows.append({
                "ndc": ndc_stub + "-01",
                "drug_name": drug_name,
                "category": category,
                "avg_daily_rx_fills": avg_daily,
                "pack_sizes": "|".join(str(s) for s in pack_sizes),
                "unit_cost_usd": round(random.uniform(0.08, 4.50), 2),
                "shelf_life_days": random.choice([365, 548, 730]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demand signal helpers
# ---------------------------------------------------------------------------

def weekday_multiplier(d: date) -> float:
    """Rural pharmacies are noticeably slower on weekends."""
    dow = d.weekday()  # 0=Mon, 6=Sun
    return {0: 1.10, 1: 1.05, 2: 1.05, 3: 1.00, 4: 1.15, 5: 0.65, 6: 0.50}[dow]


def payday_multiplier(d: date) -> float:
    """Rural patients often fill prescriptions on payday (1st and 15th)."""
    if d.day in (1, 2, 15, 16):
        return 1.35
    if d.day in (3, 17):
        return 1.15
    return 1.0


def flu_season_multiplier(d: date) -> float:
    """Peak flu demand Nov–Feb, taper in Oct and Mar."""
    m = d.month
    curve = {1: 1.6, 2: 1.5, 3: 1.2, 4: 1.0, 5: 0.9, 6: 0.8,
             7: 0.8, 8: 0.8, 9: 0.9, 10: 1.2, 11: 1.5, 12: 1.6}
    return curve[m]


def allergy_season_multiplier(d: date) -> float:
    """Spring (Apr–Jun) and fall (Aug–Oct) allergy peaks."""
    m = d.month
    curve = {1: 0.7, 2: 0.7, 3: 0.9, 4: 1.5, 5: 1.6, 6: 1.4,
             7: 1.0, 8: 1.2, 9: 1.4, 10: 1.3, 11: 0.8, 12: 0.7}
    return curve[m]


SEASONAL_DRUG_SIGNALS = {
    # (drug_name fragment) → multiplier function
    "Amoxicillin":      flu_season_multiplier,
    "Azithromycin":     flu_season_multiplier,
    "Oseltamivir":      flu_season_multiplier,
    "Prednisone":       flu_season_multiplier,
    "Doxycycline":      flu_season_multiplier,
    "Cephalexin":       flu_season_multiplier,
    "Albuterol":        flu_season_multiplier,
    "Promethazine":     flu_season_multiplier,
    "Cetirizine":       allergy_season_multiplier,
    "Loratadine":       allergy_season_multiplier,
    "Fexofenadine":     allergy_season_multiplier,
    "Fluticasone":      allergy_season_multiplier,
    "Montelukast":      allergy_season_multiplier,
}


def seasonal_multiplier(drug_name: str, d: date) -> float:
    for fragment, fn in SEASONAL_DRUG_SIGNALS.items():
        if fragment in drug_name:
            return fn(d)
    return 1.0


# ---------------------------------------------------------------------------
# Dispensing record generator
# ---------------------------------------------------------------------------

def generate_dispensing_records(
    sku_catalog: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    records = []
    date_range = pd.date_range(start_date, end_date, freq="D")

    for _, sku in sku_catalog.iterrows():
        ndc = sku["ndc"]
        drug_name = sku["drug_name"]
        base_daily = sku["avg_daily_rx_fills"]
        pack_sizes = [int(x) for x in sku["pack_sizes"].split("|")]
        category = sku["category"]

        for ts in date_range:
            d = ts.date()

            # Compose demand multipliers
            mult = (
                weekday_multiplier(d)
                * payday_multiplier(d)
                * seasonal_multiplier(drug_name, d)
            )

            # Slow upward drift for chronic meds over 2 years (aging rural pop)
            if category == "chronic":
                days_elapsed = (d - start_date).days
                mult *= 1.0 + 0.0002 * days_elapsed  # ~14% growth over 2 years

            # Poisson-distributed fills per day
            expected = base_daily * mult
            fills = int(np.random.poisson(max(expected, 0.1)))

            if fills == 0:
                continue

            for _ in range(fills):
                pack_size = random.choice(pack_sizes)
                records.append({
                    "date": d,
                    "ndc": ndc,
                    "drug_name": drug_name,
                    "category": category,
                    "units_dispensed": pack_size,
                    "rx_fill_count": 1,
                })

    df = pd.DataFrame(records)
    # Aggregate to daily per-SKU totals (more useful for forecasting)
    agg = (
        df.groupby(["date", "ndc", "drug_name", "category"])
        .agg(
            rx_fill_count=("rx_fill_count", "sum"),
            units_dispensed=("units_dispensed", "sum"),
        )
        .reset_index()
        .sort_values(["date", "ndc"])
    )
    return agg


# ---------------------------------------------------------------------------
# Inventory snapshot generator
# ---------------------------------------------------------------------------

def generate_inventory_snapshots(
    dispensing: pd.DataFrame,
    sku_catalog: pd.DataFrame,
    start_date: date,
) -> pd.DataFrame:
    """
    Simulates weekly inventory snapshots.
    Tracks on-hand units, reorder events, expiry batches, and occasional
    stockout / over-order events.
    """
    snapshots = []
    weekly_dates = pd.date_range(start_date, dispensing["date"].max(), freq="W-MON")

    for _, sku in sku_catalog.iterrows():
        ndc = sku["ndc"]
        drug_name = sku["drug_name"]
        shelf_life_days = int(sku["shelf_life_days"])
        unit_cost = float(sku["unit_cost_usd"])
        pack_sizes = [int(x) for x in sku["pack_sizes"].split("|")]

        # Demand for this SKU by week
        sku_disp = dispensing[dispensing["ndc"] == ndc].copy()
        sku_disp["week"] = pd.to_datetime(sku_disp["date"]).dt.to_period("W")
        weekly_disp = sku_disp.groupby("week")["units_dispensed"].sum().to_dict()

        # Starting stock: ~6 weeks of average demand
        avg_weekly = sku["avg_daily_rx_fills"] * 7 * random.choice(pack_sizes)
        on_hand = int(avg_weekly * 6)
        expiry_date = start_date + timedelta(days=shelf_life_days)

        for snap_ts in weekly_dates:
            snap_date = snap_ts.date()
            period = pd.Period(snap_ts, freq="W")
            dispensed_this_week = int(weekly_disp.get(period, 0))

            on_hand -= dispensed_this_week

            # Stockout event
            stockout = False
            if on_hand < 0:
                stockout = True
                on_hand = 0

            # Reorder logic: reorder if < 3 weeks of avg demand on hand
            reorder_threshold = int(avg_weekly * 3)
            reorder_qty = 0
            if on_hand < reorder_threshold:
                # Occasionally over-order (simulation of bad manual ordering)
                over_order_chance = random.random()
                if over_order_chance < 0.12:
                    reorder_qty = int(avg_weekly * random.uniform(8, 14))  # over-order
                else:
                    reorder_qty = int(avg_weekly * random.uniform(4, 7))
                on_hand += reorder_qty
                expiry_date = snap_date + timedelta(days=shelf_life_days)

            # Expiry check — units expire
            expired_units = 0
            if snap_date >= expiry_date and on_hand > 0:
                # Partial expiry if on_hand is large (multiple batches in reality)
                expired_units = min(on_hand, int(on_hand * random.uniform(0.05, 0.30)))
                on_hand -= expired_units
                expiry_date = snap_date + timedelta(days=shelf_life_days)

            snapshots.append({
                "snapshot_date": snap_date,
                "ndc": ndc,
                "drug_name": drug_name,
                "on_hand_units": on_hand,
                "dispensed_past_week": dispensed_this_week,
                "reorder_qty": reorder_qty,
                "expired_units": expired_units,
                "stockout_flag": stockout,
                "expiry_date": expiry_date,
                "inventory_value_usd": round(on_hand * unit_cost, 2),
            })

    return pd.DataFrame(snapshots).sort_values(["snapshot_date", "ndc"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "synthetic"
    )
    os.makedirs(out_dir, exist_ok=True)

    start_date = date(2024, 1, 1)
    end_date   = date(2025, 12, 31)

    print("Building SKU catalog...")
    catalog = build_sku_catalog()
    catalog_path = os.path.join(out_dir, "sku_catalog.csv")
    catalog.to_csv(catalog_path, index=False)
    print(f"  {len(catalog)} SKUs -> {catalog_path}")

    print("Generating dispensing records...")
    dispensing = generate_dispensing_records(catalog, start_date, end_date)
    disp_path = os.path.join(out_dir, "dispensing_records.csv")
    dispensing.to_csv(disp_path, index=False)
    print(f"  {len(dispensing):,} daily SKU rows -> {disp_path}")

    print("Generating inventory snapshots...")
    inventory = generate_inventory_snapshots(dispensing, catalog, start_date)
    inv_path = os.path.join(out_dir, "inventory_snapshots.csv")
    inventory.to_csv(inv_path, index=False)
    print(f"  {len(inventory):,} weekly snapshot rows -> {inv_path}")

    print("\nDone. Quick stats:")
    print(f"  Date range:      {start_date} to {end_date}")
    print(f"  SKUs:            {len(catalog)}")
    print(f"  Total Rx fills:  {dispensing['rx_fill_count'].sum():,}")
    print(f"  Stockout events: {inventory['stockout_flag'].sum():,}")
    print(f"  Expired units:   {inventory['expired_units'].sum():,}")
    print(f"  Over-order events (reorder > 7× avg weekly): "
          f"{(inventory['reorder_qty'] > catalog.set_index('ndc')['avg_daily_rx_fills'].mean() * 7 * 7).sum()}")


if __name__ == "__main__":
    main()
