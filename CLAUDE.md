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
- Step 4 complete. **Next:** Step 5 — real data ingestion pipeline (PMS exports from pilot pharmacies)
