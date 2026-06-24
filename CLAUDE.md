# Velozen AI — Rural Pharmacy Predictive Ordering System
## Claude Code Context File
> This file gives Claude full project context at the start of every session.
> Keep it updated as major decisions are made.

---

## Project Overview

**Company:** Velozen AI
**Product:** AI-powered forecasting and inventory optimization system for small rural pharmacies
**Core Value Prop:** Use hospital diagnostic trends + pharmacy prescription history to predict medication demand, reduce over-ordering, minimize expiration waste, and prevent critical stockouts.
**Current Stage:** Early scaffold / pre-data. Building synthetic data pipeline and forecasting model core before real pharmacy data is available.

---

## The Problem We're Solving

- Rural pharmacies over-order due to unpredictable local demand → expiration losses
- Under-ordering causes medication shortages and delayed patient care
- No predictive inventory tooling exists at this market level
- Hospital diagnostic data is almost never integrated into pharmacy inventory planning

---

## Core System Components (per blueprint)

1. Secure backend database (prescriptions + inventory)
2. Hospital & pharmacy data ingestion pipelines
3. Data normalization and cleaning engine
4. Medication demand forecasting models (ML core)
5. Recommendation engine for optimized ordering quantities
6. Overstock / understock risk scoring system
7. Web dashboard for pharmacy staff
8. Automated alerts and notifications
9. Analytics and reporting tools

---

## Agreed Build Sequence

> Do NOT deviate from this order without updating this file.

1. ✅ Project scaffold + CLAUDE.md
2. ✅ Synthetic data generator (realistic dispensing records, no real data needed)
3. ✅ Core forecasting model pipeline (LightGBM, offline validation) — median MAPE 19.4% (chronic 13.1%, seasonal 27.9%)
4. ✅ Basic dashboard skeleton (mock data, for demo purposes)
4b. ✅ Data normalization layer (multi-source ingestion pipeline)
5. 🔲 Real data ingestion pipeline (PMS exports from pilot pharmacies)
6. 🔲 Hospital data integration (HL7/FHIR — hardest step, comes last in v1)
7. 🔲 Recommendation + alert engine (layered on validated forecasts)
8. 🔲 HIPAA compliance hardening (before any real patient-adjacent data)

---

## Key Technical Decisions (and why)

| Decision | Choice | Reason |
|---|---|---|
| Forecasting library | LightGBM (global model across all SKUs) | Prophet abandoned — Python 3.14 incompatible with bundled Stan binary; LightGBM faster and portable |
| Drug identifier standard | NDC codes | Universal standard across all pharmacy systems |
| Data ingestion target | PMS export files first, API second | Fastest path to real data; PioneerRx/QS1/Liberty are common rural PMS platforms |
| Hospital data approach | Start with CDC/state public health data as proxy | Direct hospital HL7/FHIR feeds require lengthy data sharing agreements |
| Model feedback loop | Must be designed in from day one | Prediction → recommendation → actual order → actual dispensing → model update |
| UI approach | Human-in-the-loop (pharmacist reviews, doesn't auto-order) | Builds trust in v1; avoids liability of fully automated ordering |

---

## Data Strategy

### Synthetic Data (current phase)
Simulate realistic pharmacy dispensing records including:
- 100–200 common rural pharmacy SKUs
- Focus: chronic condition meds (metformin, lisinopril, atorvastatin, amlodipine, omeprazole)
- Seasonal meds (amoxicillin, albuterol, oseltamivir, cetirizine)
- Realistic patterns: weekday/weekend drop-off, payday spikes (1st & 15th of month), seasonal illness curves
- Inventory on-hand levels, expiry dates, occasional stockout and over-order events

### Real Data (pending pilot pharmacy confirmation)
- **Blocker:** Need to confirm which pharmacy management system (PMS) pilot pharmacies use
- **Action item:** Friend (Velozen founder) is currently out of the country — confirm PMS platform on return
- Common rural PMS platforms: PioneerRx, QS/1, Liberty Software
- NDC code standardization will be required across all sources

### External Data Sources (future)
- CDC ILINet (flu surveillance) — free, public
- State health department disease surveillance feeds
- Regional demographic data
- Hospital diagnostic feeds (HL7/FHIR) — long-term, requires partnership agreements

---

## Compliance Requirements

- **HIPAA compliant infrastructure** is non-negotiable before any real patient data touches the system
- Encrypted storage and transmission (at rest + in transit)
- Role-based access control (RBAC)
- Audit logging on all data access
- Minimize use of patient-identifiable information — aggregate/anonymized where possible
- `.env` files for all secrets — never commit to version control

---

## Monetization Model

- Monthly SaaS subscription
- Tiered pricing based on pharmacy size
- Regional healthcare contracts (longer term)
- Future: hospital system partnerships

---

## Long-Term Vision

- Expand into regional healthcare networks
- Real-time outbreak and epidemiological forecasting integration
- Full pharmacy supply chain optimization
- Become predictive infrastructure for medication demand planning across rural healthcare

---

## Project Structure

```
velozen-pharmacy/
├── CLAUDE.md              ← You are here. Update this as decisions are made.
├── README.md              ← Human-facing project overview
├── .env.example           ← Environment variable template (never commit .env)
├── .gitignore
├── requirements.txt       ← Python dependencies
├── data/
│   ├── raw/               ← Original source data, NEVER modify files here
│   ├── processed/         ← Cleaned and transformed data
│   └── synthetic/         ← Generated mock data for development
├── models/                ← Saved/serialized model files
├── notebooks/             ← Jupyter notebooks for exploration and analysis
├── src/
│   ├── ingestion/         ← Data pipeline and ingestion code
│   ├── features/          ← Feature engineering
│   ├── models/            ← Model training and inference code
│   └── dashboard/         ← Web dashboard UI code
├── tests/                 ← Unit and integration tests
└── docs/
    └── decisions.md       ← Log of architectural decisions and reasoning
```

---

## Open Questions / Blockers

- [ ] Which PMS platform(s) do pilot pharmacies use? (ask founder on return)
- [ ] Which specific pharmacies are being targeted for pilot?
- [ ] Does founder have any existing hospital relationships for data sharing?
- [ ] Has any legal/compliance review been done on the data sharing model?

---

## Session Notes

*Update this section at the end of each working session with what was completed and what's next.*

**Session 1: 5/28/26**
- Reviewed full project blueprint (Velozen AI — Rural Pharmacy Predictive Ordering System PDF)
- Established build sequence and technical approach
- Scaffolded project structure
- Created this CLAUDE.md

**Session 2: 5/28/26**
- Built synthetic data generator (`src/generate_synthetic_data.py`) — COMPLETE
  - 96 SKUs, 67,635 daily dispensing rows, 10,080 weekly inventory snapshots
  - Demand patterns: weekday/weekend, payday spikes, flu/allergy seasonal curves, chronic drift
- Built LightGBM forecasting pipeline (`src/models/forecaster.py`, `src/models/train_and_evaluate.py`)
  - Abandoned Prophet — Python 3.14 incompatible with bundled Stan binary
  - Global model architecture: one LightGBM across all SKUs (lag features, rolling means, calendar, regressors)
  - Train on 2024, evaluate on 2025 H1

**Session 3: 5/29/26**
- Ran forecasting pipeline — date-alignment fix confirmed working (2,496/2,496 rows matched)
- Eval results: median MAPE 19.4%, chronic meds 13.1%, seasonal 27.9%, worst SKU Oseltamivir 37.9%
- Will become more robust with years of real pharmacy data
- Outputs written: `data/synthetic/forecasts.csv`, `eval_metrics.csv`, `feature_importance.csv`
- Step 3 complete.
- Built Streamlit dashboard skeleton (`src/dashboard/app.py`) — COMPLETE
  - Four pages: Overview, Forecast Explorer, SKU Detail, Model Insights
  - Reads directly from `data/synthetic/` CSVs (forecasts, eval_metrics, feature_importance)
  - Plotly charts: MAPE histogram, actual vs forecast line chart, feature importance bar, bias scatter
  - Run: `streamlit run src/dashboard/app.py`
- Step 4 complete.

**Session 5: 5/31/26**
- Built data normalization layer (`src/ingestion/`) — COMPLETE
  - `normalize.py` — standard schema, NDC standardization, validation, `build_training_set()`
  - `source_synthetic.py` — live adapter for existing synthetic data
  - `source_synthea.py` — documented stub, ready for Synthea SD population data
  - `source_cms.py` — documented stub, ready for CMS Medicaid Drug Utilization data
  - `train_and_evaluate.py` updated to load data through normalization layer
- Step 4b complete.

**Session 6: 6/7/26**
- Downloaded Synthea JAR, generated 2,000-patient Aberdeen SD population (`data/synthea/csv/`)
- Implemented `source_synthea.py` — keyword-based drug matching to catalog NDCs, vectorized week expansion, date/time normalization fix
- Retrained model on combined synthetic + Synthea data — median MAPE 21.7%
- Lag features dominate over rolling means — Synthea adds week-to-week variability

**Session 7: 6/8/26**
- Downloaded CMS SD Medicaid Drug Utilization data 2022-2024 (`data/cms/sd_drug_utilization_2022_2024.csv`)
- Implemented `source_cms.py` — vectorized quarterly→weekly expansion, proportional scaling to single-pharmacy population
- Key lesson: CMS data is statewide (~140k patients) so fills must be scaled to single-pharmacy level (~5k patients); also limit CMS to pre-2024 to avoid contradicting synthetic signal on same dates
- Retrained on synthetic + Synthea + CMS (2022-2023 only): median MAPE 22.2% — `week_of_year` entered top 5 features for first time (CMS teaching real seasonal patterns)
- Updated .gitignore to exclude large data files and tools/
- **Next:** Real pilot pharmacy data pipeline (Step 5) — blocked pending PMS platform confirmation from founder

**Session 8: 6/15/26**
- Reviewed original blueprint Section 5 (10 developer steps) against current build status
- Identified gap: Step 1 (secure database schema) was skipped in favor of getting the ML core working first
- Built PostgreSQL database layer (`src/db/`) — COMPLETE
  - `models.py` — SQLAlchemy ORM: pharmacies, drugs, dispensing_records, inventory_snapshots, forecasts, model_eval_metrics
  - `connection.py` — engine + session factory, context manager, reads DATABASE_URL from .env
  - `seed.py` — loads all existing CSVs into the database; safe to re-run
  - Switched from psycopg2 to psycopg3 (`psycopg[binary]`) for Python 3.14 compatibility
- Updated requirements.txt (added sqlalchemy, psycopg[binary], streamlit, tqdm; removed prophet/dash)
- Updated .env.example with DATABASE_URL format

**Session 9: 6/17/26**
- Installed PostgreSQL 18 on dev machine; created `velozen` database
- Added PostgreSQL bin to user PATH
- Ran `python src/db/seed.py` — database fully seeded (96 drugs, 1 pharmacy, 2,496 forecasts, 96 eval metrics, 10,080 inventory rows, 67,635 dispensing records)
- Fixed Streamlit Cloud deployment issues:
  - Switched `DATA_DIR` from `os.path.dirname(__file__)` to `Path(__file__).resolve()` for reliable path resolution on cloud
  - Split `requirements.txt` into lean dashboard version (4 packages) and `requirements-dev.txt` (full local dev stack)
  - Removed version pins from dashboard requirements to avoid Streamlit Cloud conflicts
  - Added `python-dotenv` to dashboard requirements
- Dashboard live at: https://velozen-pharmacy-f557spnmjtarrt9aotd54z.streamlit.app/

**Session 10: 6/18/26**
- Wired dashboard to read from PostgreSQL when `DATABASE_URL` is set, CSV fallback otherwise
  - Uses `@st.cache_resource` for engine, `@st.cache_data(ttl=300)` for DB queries, `@st.cache_data` for CSV
  - `feature_importance.csv` still loaded from file (not in DB schema — model artifact)
  - Sidebar shows "Data source: PostgreSQL" or "Data source: CSV (local)"
  - Streamlit Cloud deployment continues using CSV fallback until cloud DB is provisioned
- Verified: DB path returns correct data (2,496 rows, 96 SKUs, 26 weeks, median MAPE 22.19%)
- **Next:** Recommendation engine (Step 7 in blueprint) — or provision cloud Postgres (Neon/Supabase free tier) so Streamlit Cloud also reads from DB

**Session 10 continued: 6/18/26**
- Built recommendation engine (`src/models/recommender.py`) — COMPLETE
  - Inputs: latest inventory snapshot, forecast demand, model accuracy metrics, drug catalog
  - Outputs: per-SKU order qty, days of supply, stockout risk (HIGH/MEDIUM/LOW), expiration risk, reasoning
  - Safety stock = lead-time demand × (1 + MAPE/100)
  - Bias correction: partial (-bias × 0.5) to avoid amplifying systematic forecast error
  - Rounds order qty to smallest pack size
  - Uses `as_of_date` from latest inventory snapshot (not today's date) so date math is consistent with data
- Added "Recommendations" page to dashboard
  - KPI row: SKUs to order, high-risk count, expiration alerts, estimated order value
  - Filterable/sortable table with color-coded risk badges
  - Drug detail panel with reasoning text
  - Parameters (target supply days, lead time) in sidebar expander
- Synthetic data reveals overstock story: 77/96 SKUs at expiration risk, only 2 need reordering ($469 total)
  - This is exactly the Velozen value prop: pharmacy has massively over-ordered slow-moving drugs
- **Next:** Risk scoring module (Step 6) or alert engine, OR begin real data ingestion prep

**Session 11: 6/23/26**
- Built risk scoring module (`src/models/risk_scorer.py`) — COMPLETE
  - `understock_score` (0–100): scales across three zones (comfortable → warning → critical) based on days_of_supply vs lead time + safety buffer; adjusted for MAPE uncertainty (+up to 10 pts) and bias direction
  - `overstock_score` (0–100): log₂ magnitude (2×→25, 4×→50, 8×→75) + expiry waste pressure (up to +25); log₂ avoids hard saturation so extreme overstock stays differentiated
  - `risk_label`: STOCKOUT_CRITICAL/WARNING, OVERSTOCK_CRITICAL/WARNING, OK
  - Fixed bug: linear magnitude formula was clipping all drugs with >4× overstock at exactly 50; replaced with log₂ curve — zero SKUs at exactly 50 after fix
- Updated `src/models/recommender.py` to expose `days_until_expiry` in output (needed by risk scorer)
- Added "Risk Overview" page to dashboard
  - KPI row: stockout critical/warning counts, overstock critical/warning counts
  - Risk quadrant scatter plot: x=understock score, y=overstock score, dashed threshold lines at 40 and 75
  - Side-by-side top-10 tables for each risk type with progress bar columns
  - Full expandable SKU table with both scores
- Synthetic data result: 26 OVERSTOCK_CRITICAL, 2 OVERSTOCK_WARNING, 0 stockout risk — consistent with overstock story
- **Next:** Alert engine (email/in-dashboard notifications), real data ingestion prep, or cloud Postgres provisioning
