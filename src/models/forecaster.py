"""
Per-SKU demand forecasting for Velozen AI using LightGBM.

Architecture: one global LightGBM model trained across all SKUs simultaneously.
This gives far more training signal than per-SKU models when history is short
(~52 weekly observations per SKU at launch).

Features per row (one row = one SKU x one week):
  Lag features       : fill count 1, 2, 4, 8, 13 weeks ago
  Rolling means      : 4-week, 8-week, 13-week trailing avg
  Calendar           : week_of_year, month, quarter, year_offset (slow trend)
  Demand regressors  : payday_week, flu_index, allergy_index
  SKU identity       : category (encoded), avg_daily_rx, log_avg_daily_rx

All public methods return plain DataFrames — callers have no LightGBM dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Seasonality index helpers  (mirror the synthetic data generator curves)
# ---------------------------------------------------------------------------

_FLU_CURVE = {
    1: 1.0, 2: 0.94, 3: 0.75, 4: 0.50, 5: 0.44, 6: 0.38,
    7: 0.38, 8: 0.38, 9: 0.44, 10: 0.75, 11: 0.94, 12: 1.0,
}
_ALLERGY_CURVE = {
    1: 0.21, 2: 0.21, 3: 0.43, 4: 1.0, 5: 1.0, 6: 0.86,
    7: 0.50, 8: 0.71, 9: 0.93, 10: 0.86, 11: 0.36, 12: 0.21,
}

_CATEGORY_MAP = {"chronic": 0, "seasonal": 1, "other": 2}

LAG_WEEKS    = [1, 2, 4, 8, 13]
ROLLING_WINS = [4, 8, 13]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(weekly: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """
    weekly  : columns [ds, ndc, drug_name, category, y]  (all SKUs, all dates)
    catalog : columns [ndc, avg_daily_rx_fills, ...]

    Returns a feature matrix ready for LightGBM (one row per SKU × week).
    Rows with NaN lag values (first LAG_WEEKS weeks per SKU) are dropped.
    """
    df = weekly.copy().sort_values(["ndc", "ds"]).reset_index(drop=True)

    # Calendar features
    ds = pd.to_datetime(df["ds"])
    df["week_of_year"]  = ds.dt.isocalendar().week.astype(int)
    df["month"]         = ds.dt.month
    df["quarter"]       = ds.dt.quarter
    df["year_offset"]   = (ds.dt.year - ds.dt.year.min()).astype(float)

    # Demand regressors
    df["payday_week"]   = df.groupby("ndc", group_keys=False).apply(
        lambda g: _payday_week_series(g["ds"]), include_groups=False
    ).values
    df["flu_index"]     = ds.dt.month.map(_FLU_CURVE).values
    df["allergy_index"] = ds.dt.month.map(_ALLERGY_CURVE).values

    # SKU identity
    cat_map = catalog.set_index("ndc")["avg_daily_rx_fills"].to_dict()
    df["avg_daily_rx"]     = df["ndc"].map(cat_map).fillna(5.0)
    df["log_avg_daily_rx"] = np.log1p(df["avg_daily_rx"])
    df["category_enc"]     = df["category"].map(_CATEGORY_MAP).fillna(2)

    # Lag and rolling features — computed within each SKU's time series
    for lag in LAG_WEEKS:
        df[f"lag_{lag}"] = df.groupby("ndc")["y"].shift(lag)
    for win in ROLLING_WINS:
        df[f"roll_mean_{win}"] = (
            df.groupby("ndc")["y"]
            .transform(lambda s: s.shift(1).rolling(win, min_periods=win // 2).mean())
        )

    df = df.dropna(subset=[f"lag_{LAG_WEEKS[-1]}"]).copy()
    return df


def _payday_week_series(ds_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ds_series)
    result = []
    for d in dates:
        week_days = pd.date_range(d, periods=7, freq="D")
        result.append(1 if any(day.day in (1, 15) for day in week_days) else 0)
    return pd.Series(result, index=ds_series.index)


FEATURE_COLS = (
    [f"lag_{l}" for l in LAG_WEEKS]
    + [f"roll_mean_{w}" for w in ROLLING_WINS]
    + ["week_of_year", "month", "quarter", "year_offset",
       "payday_week", "flu_index", "allergy_index",
       "avg_daily_rx", "log_avg_daily_rx", "category_enc"]
)


# ---------------------------------------------------------------------------
# Global LightGBM forecaster
# ---------------------------------------------------------------------------

class GlobalForecaster:
    """
    Trains one LightGBM model across all SKUs, predicts one week ahead
    via a recursive multi-step loop for longer horizons.
    """

    def __init__(self):
        self._model: lgb.Booster | None = None
        self._history: pd.DataFrame | None = None   # needed for recursive prediction
        self._catalog: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, weekly: pd.DataFrame, catalog: pd.DataFrame) -> None:
        """
        weekly  : [ds, ndc, drug_name, category, y]
        catalog : SKU catalog with avg_daily_rx_fills
        """
        self._history = weekly.copy()
        self._catalog = catalog.copy()

        feat_df = build_features(weekly, catalog)
        X = feat_df[FEATURE_COLS].values
        y = feat_df["y"].values

        params = {
            "objective":        "regression",
            "metric":           "rmse",
            "num_leaves":       63,
            "learning_rate":    0.05,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq":     5,
            "min_child_samples": 10,
            "verbose":          -1,
            "n_jobs":           -1,
        }
        dtrain = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)
        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
            valid_sets=[dtrain],
        )

    # ------------------------------------------------------------------
    # Prediction — recursive multi-step
    # ------------------------------------------------------------------

    def predict(self, horizon_weeks: int) -> pd.DataFrame:
        """
        Predict `horizon_weeks` weeks ahead for every SKU in history.
        Returns DataFrame with columns: ds, ndc, drug_name, yhat.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        running = self._history.copy()
        all_preds = []

        last_date = pd.to_datetime(running["ds"]).max()
        # Use 7D so anchor stays on whichever weekday last_date is (Monday after load_weekly)
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=7), periods=horizon_weeks, freq="7D"
        )

        for step_date in future_dates:
            feat_df = build_features(running, self._catalog)
            # Keep only the most recent row per SKU (latest state)
            latest = feat_df.sort_values("ds").groupby("ndc").tail(1).copy()
            latest["ds"] = step_date

            X = latest[FEATURE_COLS].values
            latest["yhat"] = np.maximum(self._model.predict(X), 0.0)

            # Append predictions back as actuals for next step's lags
            new_rows = latest[["ds", "ndc", "drug_name", "category"]].copy()
            new_rows["y"] = latest["yhat"].values
            running = pd.concat([running, new_rows], ignore_index=True)

            all_preds.append(latest[["ds", "ndc", "drug_name", "yhat"]])

        return pd.concat(all_preds, ignore_index=True)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return pd.DataFrame({
            "feature":    self._model.feature_name(),
            "importance": self._model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
