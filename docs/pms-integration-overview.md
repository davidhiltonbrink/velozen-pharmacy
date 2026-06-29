# Velozen AI — PMS Integration Overview

**The goal:** Velozen's forecasting and risk intelligence should feel like a native part of the pharmacy's existing workflow, not a separate tool they have to remember to open.

---

## How integration works

Velozen exposes a standard REST API. A pharmacy management system (PMS) calls that API to retrieve ordering recommendations, risk scores, and demand forecasts for any drug in the formulary. The PMS displays those insights inside its own interface — in the ordering screen, on the drug detail page, or as alerts in the daily workflow — without the pharmacist ever leaving the software they already know.

---

## What the API provides

| Capability | What it returns |
|---|---|
| **Ordering recommendations** | For each SKU: suggested order quantity, estimated order value, days of supply remaining, plain-English reasoning |
| **Risk scores** | Understock score (0–100) and overstock score (0–100) per drug, with a clear label (e.g. *OVERSTOCK_CRITICAL*, *STOCKOUT_WARNING*) |
| **Demand forecasts** | Predicted weekly fill counts per drug, informed by historical dispensing patterns and seasonal trends |
| **Inventory push** | Accepts live inventory snapshots from the PMS so Velozen's predictions always reflect current stock levels |

---

## Integration levels — pick your depth

### Level 1 — Embedded panel *(fastest, weeks not months)*
The PMS adds a Velozen panel to its drug detail or ordering screen using a single embed tag. Pharmacist sees recommendations without leaving the PMS. No PMS development required beyond placing the panel.

### Level 2 — API-powered *(fully native, recommended for serious partners)*
The PMS calls the Velozen API and renders recommendations using its own UI components. Completely seamless — pharmacists see Velozen data with no visual indication it comes from a third party. Full control over placement, styling, and workflow integration.

### Level 3 — Native plugin *(longest timeline, highest value)*
Velozen is listed in the PMS partner marketplace and installs like any other module. Requires a formal partnership agreement with the PMS vendor but delivers the deepest, most maintainable integration.

---

## What Velozen needs from the PMS

To keep predictions accurate, Velozen needs a current inventory snapshot — ideally updated automatically whenever the PMS updates stock levels. This can happen in whichever way is easiest for the PMS:

- **Webhook** — PMS sends a notification to Velozen when inventory changes *(real-time)*
- **Scheduled export** — PMS exports a data file nightly; Velozen ingests it automatically *(next-day)*
- **API pull** — Velozen polls the PMS on a schedule *(requires PMS API access)*

Velozen is designed to work with all three approaches and can adapt to whatever the PMS already supports.

---

## Data and compliance

Velozen operates on **dispensing volumes and inventory counts — not patient records.** No protected health information (PHI) is required for the forecasting or ordering workflow. Predictions are made at the drug-SKU level, aggregated across all patients.

Full HIPAA infrastructure hardening is on the roadmap before any patient-adjacent data is processed.

---

## Suggested pilot integration path

| Milestone | Estimated timeline |
|---|---|
| Velozen REST API live (all endpoints) | 1–2 weeks |
| Embedded panel proof-of-concept inside PMS | Same week as API |
| PMS-specific data adapter (maps PMS export to Velozen format) | 1–2 weeks after PMS platform confirmed |
| API-powered native integration (PMS builds their UI against our API) | Depends on PMS dev bandwidth |

The embedded panel approach can be running inside the pilot pharmacy's PMS within days of the API going live, giving real users real data while the deeper native integration is built out in parallel.

---

## Questions for the pilot pharmacy

To scope the integration work accurately, it helps to know:

1. Which PMS platform are you on? *(PioneerRx, QS/1, Liberty, Computer-Rx, Rx30, other)*
2. Does your PMS have an API or support third-party integrations/plugins?
3. Would you prefer Velozen recommendations to appear in the ordering workflow, the drug detail screen, or as daily alerts — or all three?
4. How frequently does your inventory update in the PMS? *(real-time, end-of-day, weekly)*

---

*Velozen AI · velozen.ai · Contact: [founder contact]*
