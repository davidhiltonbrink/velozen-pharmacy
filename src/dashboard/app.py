"""
Velozen AI — Pharmacy Forecasting Dashboard

Run:
  streamlit run src/dashboard/app.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR.parent / "data" / "synthetic"

# Add src/ to path so db package is importable
sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Velozen AI — Pharmacy Forecasting",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading — PostgreSQL when DATABASE_URL is set, CSV fallback otherwise
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_engine():
    """Return SQLAlchemy engine if DATABASE_URL is configured, else None."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("DATABASE_URL"):
            return None
        from db.connection import engine
        return engine
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_from_db(_engine):
    forecasts = pd.read_sql(
        "SELECT drug_name, category, forecast_week AS ds, "
        "y::float AS y, yhat::float AS yhat FROM forecasts",
        _engine,
        parse_dates=["ds"],
    )
    metrics = pd.read_sql(
        "SELECT ndc, drug_name, category, "
        "mape_pct::float AS mape_pct, mae::float AS mae, "
        "rmse::float AS rmse, bias::float AS bias, high_error_flag "
        "FROM model_eval_metrics",
        _engine,
    )
    features = pd.read_csv(DATA_DIR / "feature_importance.csv")
    return forecasts, metrics, features


@st.cache_data
def _load_from_csv():
    forecasts = pd.read_csv(DATA_DIR / "forecasts.csv", parse_dates=["ds"])
    metrics   = pd.read_csv(DATA_DIR / "eval_metrics.csv")
    features  = pd.read_csv(DATA_DIR / "feature_importance.csv")
    return forecasts, metrics, features


def load_data():
    engine = _get_engine()
    if engine is not None:
        return _load_from_db(engine)
    return _load_from_csv()


@st.cache_data(ttl=300)
def _load_rec_data_db(_engine):
    inventory = pd.read_sql(
        "SELECT DISTINCT ON (ndc) ndc, snapshot_date, "
        "on_hand_units::float, dispensed_past_week::float, "
        "expiry_date, stockout_flag "
        "FROM inventory_snapshots ORDER BY ndc, snapshot_date DESC",
        _engine,
        parse_dates=["snapshot_date", "expiry_date"],
    )
    drugs = pd.read_sql(
        "SELECT ndc, drug_name, category, pack_sizes, "
        "unit_cost_usd::float, shelf_life_days FROM drugs",
        _engine,
    )
    forecasts_ndc = pd.read_sql(
        "SELECT ndc, yhat::float AS yhat FROM forecasts",
        _engine,
    )
    return inventory, drugs, forecasts_ndc


@st.cache_data
def _load_rec_data_csv():
    inv = pd.read_csv(DATA_DIR / "inventory_snapshots.csv", parse_dates=["snapshot_date", "expiry_date"])
    inv["ndc"] = inv["ndc"].astype(str).str.replace("-", "", regex=False).str.zfill(11)
    inventory = (
        inv.sort_values("snapshot_date")
        .groupby("ndc", as_index=False)
        .last()[["ndc", "snapshot_date", "on_hand_units", "dispensed_past_week", "expiry_date", "stockout_flag"]]
    )
    drugs = pd.read_csv(DATA_DIR / "sku_catalog.csv")
    drugs["ndc"] = drugs["ndc"].astype(str).str.replace("-", "", regex=False).str.zfill(11)
    fc = pd.read_csv(DATA_DIR / "forecasts.csv")
    fc["ndc"] = fc["ndc"].astype(str).str.zfill(11)
    return inventory, drugs, fc[["ndc", "yhat"]]


def load_recommendation_data():
    engine = _get_engine()
    if engine is not None:
        return _load_rec_data_db(engine)
    return _load_rec_data_csv()


forecasts, metrics, features = load_data()

CATEGORY_COLORS = {"chronic": "#2196F3", "seasonal": "#FF9800", "other": "#9E9E9E"}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Velozen AI")
st.sidebar.caption("Rural Pharmacy Forecasting")

_source_label = "PostgreSQL" if _get_engine() is not None else "CSV (local)"
st.sidebar.caption(f"Data source: {_source_label}")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Forecast Explorer", "SKU Detail", "Model Insights", "Recommendations"],
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Inventory Forecast Overview")
    st.caption("Evaluation period: 2025-01-06 → 2025-06-30 · Synthetic data · Model: LightGBM global")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs Tracked", len(metrics))
    col2.metric("Median MAPE", f"{metrics['mape_pct'].median():.1f}%")
    col3.metric("High-Error SKUs", int(metrics["high_error_flag"].sum()),
                help="SKUs with MAPE > 25%")
    over  = int((metrics["bias"] > 1).sum())
    under = int((metrics["bias"] < -1).sum())
    col4.metric("Systematic Bias", f"{over} over / {under} under",
                help="SKUs with |bias| > 1 unit/week")

    st.divider()

    # Category breakdown
    st.subheader("Performance by Drug Category")
    cat_summary = (
        metrics.groupby("category")
        .agg(
            SKUs=("ndc", "count"),
            median_mape=("mape_pct", "median"),
            mean_mape=("mape_pct", "mean"),
            high_error=("high_error_flag", "sum"),
        )
        .reset_index()
        .rename(columns={
            "category":    "Category",
            "SKUs":        "# SKUs",
            "median_mape": "Median MAPE %",
            "mean_mape":   "Mean MAPE %",
            "high_error":  "High-Error SKUs",
        })
    )
    cat_summary["Median MAPE %"] = cat_summary["Median MAPE %"].round(1)
    cat_summary["Mean MAPE %"]   = cat_summary["Mean MAPE %"].round(1)
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)

    st.divider()

    # MAPE distribution
    st.subheader("MAPE Distribution Across All SKUs")
    fig = px.histogram(
        metrics, x="mape_pct", color="category",
        color_discrete_map=CATEGORY_COLORS,
        nbins=20, barmode="overlay", opacity=0.75,
        labels={"mape_pct": "MAPE (%)", "category": "Category"},
    )
    fig.add_vline(x=metrics["mape_pct"].median(), line_dash="dash",
                  annotation_text=f"Median {metrics['mape_pct'].median():.1f}%")
    fig.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Worst SKUs table
    st.subheader("Highest-Error SKUs")
    worst = (
        metrics[metrics["high_error_flag"]]
        [["drug_name", "category", "mape_pct", "mae", "bias"]]
        .sort_values("mape_pct", ascending=False)
        .rename(columns={
            "drug_name": "Drug",
            "category":  "Category",
            "mape_pct":  "MAPE %",
            "mae":       "MAE (units/wk)",
            "bias":      "Bias",
        })
    )
    st.dataframe(worst, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Forecast Explorer
# ---------------------------------------------------------------------------

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Browse predicted vs actual demand for any week in the evaluation period.")

    col_f, col_c = st.columns([2, 1])

    available_weeks = sorted(forecasts["ds"].dt.date.unique())
    selected_week = col_f.selectbox(
        "Select week (Monday)",
        available_weeks,
        format_func=lambda d: d.strftime("%B %d, %Y"),
    )
    cat_filter = col_c.multiselect(
        "Category filter",
        options=["chronic", "seasonal", "other"],
        default=["chronic", "seasonal", "other"],
    )

    week_df = forecasts[
        (forecasts["ds"].dt.date == selected_week) &
        (forecasts["category"].isin(cat_filter))
    ].copy()

    week_df["error_pct"] = (
        ((week_df["yhat"] - week_df["y"]).abs() / week_df["y"].clip(lower=1)) * 100
    ).round(1)
    week_df["over_under"] = week_df.apply(
        lambda r: "Over" if r["yhat"] > r["y"] else "Under", axis=1
    )

    # Summary for this week
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs this week", len(week_df))
    c2.metric("Median error", f"{week_df['error_pct'].median():.1f}%")
    c3.metric("Over-forecast SKUs", int((week_df["over_under"] == "Over").sum()))

    st.divider()

    display = week_df[[
        "drug_name", "category", "y", "yhat", "error_pct", "over_under"
    ]].sort_values("error_pct", ascending=False).rename(columns={
        "drug_name":  "Drug",
        "category":   "Category",
        "y":          "Actual",
        "yhat":       "Forecast",
        "error_pct":  "Error %",
        "over_under": "Direction",
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Error %": st.column_config.ProgressColumn(
                "Error %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Page: SKU Detail
# ---------------------------------------------------------------------------

elif page == "SKU Detail":
    st.title("SKU Detail")
    st.caption("Actual vs forecast over the full 26-week evaluation window.")

    drug_options = sorted(forecasts["drug_name"].unique())
    selected_drug = st.selectbox("Select drug", drug_options)

    sku_fc  = forecasts[forecasts["drug_name"] == selected_drug].sort_values("ds")
    sku_met = metrics[metrics["drug_name"] == selected_drug].iloc[0]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE", f"{sku_met['mape_pct']:.1f}%")
    c2.metric("MAE", f"{sku_met['mae']:.1f} units/wk")
    c3.metric("RMSE", f"{sku_met['rmse']:.1f}")
    bias_val = sku_met["bias"]
    bias_label = f"+{bias_val:.1f}" if bias_val > 0 else f"{bias_val:.1f}"
    bias_delta = "over-forecast" if bias_val > 0.5 else ("under-forecast" if bias_val < -0.5 else "balanced")
    c4.metric("Bias", bias_label, delta=bias_delta,
              delta_color="inverse" if bias_val < -0.5 else "normal")

    if sku_met["high_error_flag"]:
        st.warning("High-error SKU (MAPE > 25%) — forecast confidence is lower for this drug.")

    st.divider()

    # Actual vs forecast chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sku_fc["ds"], y=sku_fc["y"],
        name="Actual", line=dict(color="#333333", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sku_fc["ds"], y=sku_fc["yhat"],
        name="Forecast", line=dict(color=CATEGORY_COLORS.get(sku_met["category"], "#999"), width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"{selected_drug} — Actual vs Forecast (2025 H1)",
        xaxis_title="Week",
        yaxis_title="Units dispensed",
        height=400,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data table
    with st.expander("Show raw data"):
        raw = sku_fc[["ds", "y", "yhat"]].copy()
        raw["error"] = (raw["yhat"] - raw["y"]).round(1)
        raw["error_pct"] = ((raw["error"].abs() / raw["y"].clip(lower=1)) * 100).round(1)
        raw.columns = ["Week", "Actual", "Forecast", "Error", "Error %"]
        st.dataframe(raw, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Model Insights
# ---------------------------------------------------------------------------

elif page == "Model Insights":
    st.title("Model Insights")
    st.caption("Feature importance and error distribution from the LightGBM global model.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Feature Importance (Gain)")
        top_n = st.slider("Show top N features", 5, len(features), 10)
        fi_plot = features.head(top_n).sort_values("importance")
        fig = px.bar(
            fi_plot, x="importance", y="feature", orientation="h",
            labels={"importance": "Gain", "feature": "Feature"},
            color="importance", color_continuous_scale="Blues",
        )
        fig.update_layout(height=400, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("MAPE by Category (Box Plot)")
        fig2 = px.box(
            metrics, x="category", y="mape_pct",
            color="category", color_discrete_map=CATEGORY_COLORS,
            points="all",
            labels={"mape_pct": "MAPE (%)", "category": "Category"},
        )
        fig2.add_hline(y=25, line_dash="dash", line_color="red",
                       annotation_text="25% threshold")
        fig2.update_layout(height=400, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Bias Analysis")
    st.caption("Positive = model over-forecasts (over-order risk). Negative = under-forecasts (stockout risk).")

    fig3 = px.scatter(
        metrics, x="mape_pct", y="bias",
        color="category", color_discrete_map=CATEGORY_COLORS,
        hover_data=["drug_name"],
        labels={"mape_pct": "MAPE (%)", "bias": "Bias (units/wk)", "category": "Category"},
    )
    fig3.add_hline(y=0, line_dash="solid", line_color="#aaa")
    fig3.add_vline(x=25, line_dash="dash", line_color="red", annotation_text="25% threshold")
    fig3.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Recommendations
# ---------------------------------------------------------------------------

elif page == "Recommendations":
    from models.recommender import compute_recommendations, DEFAULT_TARGET_SUPPLY_DAYS, DEFAULT_LEAD_TIME_DAYS

    st.title("Order Recommendations")
    st.caption("AI-generated reorder suggestions. Review and adjust before placing any order.")

    # Parameters in sidebar expander
    with st.sidebar.expander("Parameters", expanded=False):
        target_days = st.slider("Target supply (days)", 14, 60, DEFAULT_TARGET_SUPPLY_DAYS, step=7)
        lead_time   = st.slider("Lead time (days)", 1, 14, DEFAULT_LEAD_TIME_DAYS)

    # Load data and compute
    inventory_df, drugs_df, forecasts_ndc_df = load_recommendation_data()

    # Use the most recent snapshot date as reference so all date math
    # is relative to the actual inventory data, not today's wall-clock date
    as_of = None
    if "snapshot_date" in inventory_df.columns:
        as_of = pd.to_datetime(inventory_df["snapshot_date"].max()).date()

    # metrics already loaded at top — pull ndc + needed cols
    metrics_for_rec = metrics[["ndc", "mape_pct", "bias", "high_error_flag"]].copy() if "ndc" in metrics.columns else pd.DataFrame(columns=["ndc", "mape_pct", "bias", "high_error_flag"])

    recs = compute_recommendations(
        inventory_df=inventory_df,
        forecasts_df=forecasts_ndc_df,
        metrics_df=metrics_for_rec,
        drugs_df=drugs_df,
        target_supply_days=target_days,
        lead_time_days=lead_time,
        as_of_date=as_of,
    )

    if as_of:
        st.caption(f"Based on inventory snapshot: {as_of.strftime('%B %d, %Y')} · Synthetic data · Model: LightGBM global")

    # KPI row
    needs_order   = recs[recs["recommended_order_qty"] > 0]
    high_risk     = recs[recs["stockout_risk"] == "HIGH"]
    expiry_risk   = recs[recs["expiration_risk"]]
    total_value   = needs_order["order_value_usd"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs to Order",    len(needs_order))
    c2.metric("High-Risk SKUs",   len(high_risk),  help="Days of supply < lead time + minimum buffer")
    c3.metric("Expiration Alerts", len(expiry_risk), help="Current stock may expire before dispensed")
    c4.metric("Est. Order Value", f"${total_value:,.0f}")

    st.divider()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    risk_filter = col_f1.multiselect(
        "Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
    )
    cat_filter = col_f2.multiselect(
        "Category", ["chronic", "seasonal", "other"],
        default=["chronic", "seasonal", "other"]
    )
    order_only = col_f3.checkbox("Show only SKUs needing an order", value=False)

    display = recs[
        recs["stockout_risk"].isin(risk_filter) &
        recs["category"].isin(cat_filter)
    ].copy()
    if order_only:
        display = display[display["recommended_order_qty"] > 0]

    # Risk color badge
    RISK_COLORS = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    display["Risk"] = display["stockout_risk"].map(RISK_COLORS) + " " + display["stockout_risk"]

    table = display[[
        "Risk", "drug_name", "category",
        "on_hand_units", "forecast_weekly_demand",
        "days_of_supply", "expiration_risk",
        "recommended_order_qty", "order_value_usd",
    ]].rename(columns={
        "drug_name":              "Drug",
        "category":               "Category",
        "on_hand_units":          "On Hand",
        "forecast_weekly_demand": "Avg Demand/Wk",
        "days_of_supply":         "Days Supply",
        "expiration_risk":        "Exp Risk",
        "recommended_order_qty":  "Order Qty",
        "order_value_usd":        "Order Value ($)",
    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Days Supply":    st.column_config.NumberColumn(format="%.0f"),
            "Avg Demand/Wk": st.column_config.NumberColumn(format="%.1f"),
            "Order Value ($)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    st.divider()

    # Detail expander for any selected drug
    st.subheader("Drug Detail")
    drug_options = display["drug_name"].tolist()
    if drug_options:
        selected = st.selectbox("Select a drug to see reasoning", drug_options)
        row = display[display["drug_name"] == selected].iloc[0]
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("On Hand",        f"{row['on_hand_units']:.0f} units")
        dc2.metric("Avg Demand/Wk",  f"{row['forecast_weekly_demand']:.1f} units")
        dc3.metric("Days of Supply", f"{row['days_of_supply']:.0f}")
        dc4.metric("Recommended Qty", f"{int(row['recommended_order_qty'])} units")
        st.info(row["reasoning"])
    else:
        st.info("No SKUs match the current filters.")
