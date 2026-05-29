# Velozen AI — Architectural Decisions Log

> Record every significant technical or product decision here with reasoning.
> This is the project memory. When something seems weird later, this explains why.

---

## 2025 — Initial Build

### Forecasting Library: Prophet (primary)
**Decision:** Use Meta's Prophet as the primary forecasting model for v1.
**Reason:** Works well with limited historical data, handles seasonality natively (weekly, yearly), interpretable output for non-technical pharmacy owners. LightGBM layered on top once more pharmacy-specific data is available.

### Drug Identifier: NDC Codes
**Decision:** Standardize all drug references on National Drug Codes (NDC).
**Reason:** Universal standard across all pharmacy management systems. Handles generic vs. brand name disambiguation.

### Hospital Data: Defer Direct Integration
**Decision:** Start with CDC/state public health feeds as hospital data proxy in v1. Direct HL7/FHIR hospital feeds deferred to v2.
**Reason:** Hospital data sharing agreements take months of legal review. Public health data is free, available now, and sufficient for initial seasonal demand signals.

### UI Philosophy: Human-in-the-Loop
**Decision:** Pharmacists review and approve all ordering recommendations. No auto-ordering.
**Reason:** Builds trust with pharmacy owners in v1. Reduces liability. Easier regulatory path.

### Data Ingestion: PMS Export First, API Second
**Decision:** Build ingestion around PMS flat file exports before pursuing live API connections.
**Reason:** Fastest path to real data. Every PMS supports CSV/Excel export; API availability varies by platform and requires vendor agreements.

### Feedback Loop: Design In From Day One
**Decision:** Closed feedback loop (prediction → recommendation → actual order → actual dispensing → model update) must be part of the initial architecture.
**Reason:** Without it the model cannot improve over time. Retrofitting this later is architecturally painful.

---
