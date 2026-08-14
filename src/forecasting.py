"""Feature engineering, STL component models, and forecast helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL


def build_feature_frame(
    complaints_series: pd.Series,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build engineered features from known/predicted history + deterministic calendar signals."""
    if dates is None:
        dates = complaints_series.index

    frame = pd.DataFrame(index=dates)
    frame["y"] = complaints_series.reindex(dates)

    # Lag and rolling features
    frame["lag_1"] = frame["y"].shift(1)
    frame["lag_7"] = frame["y"].shift(7)
    frame["lag_14"] = frame["y"].shift(14)
    frame["lag_28"] = frame["y"].shift(28)
    frame["roll7_mean"] = frame["y"].shift(1).rolling(7).mean()
    frame["roll28_mean"] = frame["y"].shift(1).rolling(28).mean()
    frame["roll7_std"] = frame["y"].shift(1).rolling(7).std()
    # Causal 28-day volatility: uses prior observations only, so it is safe for forecasting.
    frame["roll28_std"] = frame["y"].shift(1).rolling(28).std()

    # Momentum/change
    frame["mom_1_7"] = frame["lag_1"] - frame["lag_7"]
    frame["pct_change_7"] = (frame["lag_1"] - frame["lag_7"]) / (frame["lag_7"].abs() + 1e-6)

    # Deterministic calendar + trend
    dow = frame.index.dayofweek
    doy = frame.index.dayofyear
    frame["is_weekend"] = (dow >= 5).astype(int)
    frame["is_month_start"] = frame.index.is_month_start.astype(int)
    frame["is_month_end"] = frame.index.is_month_end.astype(int)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["time_idx"] = np.arange(len(frame), dtype=float)

    return frame


def score_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_pred = y_pred.reindex(y_true.index)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape}


def naive_forecast(history: pd.Series, horizon: int) -> pd.Series:
    """Repeat the last observed value for horizon days (random-walk / last-value naive)."""
    future_index = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(float(history.iloc[-1]), index=future_index, name="prediction")


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 7) -> pd.Series:
    """Repeat the last full seasonal cycle for horizon days."""
    last_cycle = history.iloc[-season:].to_numpy()
    future_index = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    values = np.tile(last_cycle, int(np.ceil(horizon / season)))[:horizon]
    return pd.Series(values, index=future_index, name="prediction")


def holt_winters_forecast(
    history: pd.Series,
    horizon: int,
    seasonal_periods: int = 7,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 0.2,
) -> pd.Series:
    """Additive Holt-Winters with weekly seasonality (pure NumPy; avoids broken statsmodels import)."""
    y = history.astype(float).to_numpy()
    n = len(y)
    m = seasonal_periods

    level = np.empty(n)
    trend = np.empty(n)
    season = np.empty(n + horizon)

    season[:m] = y[:m] - y[:m].mean()
    level[0] = y[0] - season[0]
    trend[0] = (y[m : 2 * m].mean() - y[:m].mean()) / m if n >= 2 * m else 0.0

    for t in range(1, n):
        s = season[t - m]
        level[t] = alpha * (y[t] - s) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        season[t] = gamma * (y[t] - level[t]) + (1 - gamma) * s

    future_index = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = []
    for h in range(1, horizon + 1):
        season_idx = n - m + ((h - 1) % m)
        yhat = level[-1] + h * trend[-1] + season[season_idx]
        preds.append(max(0.0, float(yhat)))

    return pd.Series(preds, index=future_index, name="prediction")


def stl_structure_components(history: pd.Series, horizon: int, period: int = 7) -> tuple[pd.Series, pd.Series]:
    """Forecast trend and seasonal components from complaints itself."""
    history = history.astype(float)
    stl_fit = STL(history, period=period, robust=True).fit()
    trend = pd.Series(stl_fit.trend, index=history.index)
    seasonal = pd.Series(stl_fit.seasonal, index=history.index)

    future_index = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Stable tendency forecast for noisy data: carry forward recent average trend level.
    trend_level = float(trend.iloc[-14:].mean())
    trend_future = pd.Series(trend_level, index=future_index, name="trend_pred")

    season_cycle = seasonal.iloc[-period:].to_numpy()
    season_future = np.tile(season_cycle, int(np.ceil(horizon / period)))[:horizon]
    season_future = pd.Series(season_future, index=future_index, name="seasonal_pred")

    return trend_future, season_future


def stl_structure_forecast(history: pd.Series, horizon: int, period: int = 7) -> pd.Series:
    """Ablation: forecast from trend+seasonal only (noise=0)."""
    trend_f, seasonal_f = stl_structure_components(history, horizon=horizon, period=period)
    yhat = (trend_f + seasonal_f).clip(lower=0.0)
    yhat.name = "prediction"
    return yhat


def _build_noise_training_frame(index: pd.DatetimeIndex, exog: pd.DataFrame) -> pd.DataFrame:
    """Build lagged exogenous + calendar features for residual modeling."""
    exog = exog.reindex(index).copy()
    for col in exog.columns:
        exog[col] = exog[col].interpolate(method="time").ffill().bfill()

    feat = pd.DataFrame(index=index)
    for col in exog.columns:
        feat[f"{col}_lag1"] = exog[col].shift(1)

    dow = feat.index.dayofweek
    doy = feat.index.dayofyear
    feat["is_weekend"] = (dow >= 5).astype(int)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return feat


def stl_hybrid_forecast(
    history: pd.Series,
    exog_hist: pd.DataFrame,
    horizon: int,
    period: int = 7,
) -> pd.Series:
    """Hybrid STL forecast: trend+seasonal from complaints, residual from exogenous features."""
    history = history.astype(float)
    stl_fit = STL(history, period=period, robust=True).fit()
    residual = pd.Series(stl_fit.resid, index=history.index, name="resid")

    train_feat = _build_noise_training_frame(history.index, exog_hist)
    train_df = train_feat.join(residual).dropna()
    X_resid = train_df.drop(columns=["resid"])
    y_resid = train_df["resid"]

    resid_model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    resid_model.fit(X_resid, y_resid)

    trend_f, seasonal_f = stl_structure_components(history, horizon=horizon, period=period)

    # For true future, unknown exogenous drivers are approximated by persisting the last known values.
    last_exog = exog_hist.reindex(history.index).copy()
    for col in last_exog.columns:
        last_exog[col] = last_exog[col].interpolate(method="time").ffill().bfill()
    last_vals = last_exog.iloc[-1]

    future_index = trend_f.index
    future_exog = pd.DataFrame(index=future_index)
    for col in exog_hist.columns:
        future_exog[col] = last_vals[col]

    future_feat = _build_noise_training_frame(
        index=history.index.append(future_index),
        exog=pd.concat([last_exog, future_exog]),
    ).loc[future_index]

    resid_raw = resid_model.predict(future_feat)
    resid_low, resid_high = np.quantile(y_resid, [0.01, 0.99])
    resid_f = pd.Series(np.clip(resid_raw, resid_low, resid_high), index=future_index, name="resid_pred")

    yhat = (trend_f + seasonal_f + resid_f).clip(lower=0.0)
    yhat.name = "prediction"
    return yhat


def fit_hgb(history: pd.Series):
    """Fit HistGradientBoosting on engineered target-history features."""
    frame = build_feature_frame(history)
    feature_cols = [c for c in frame.columns if c != "y"]
    data = frame.dropna()
    X = data[feature_cols]
    y = data["y"]

    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model, feature_cols


def build_raw_table_feature_frame(
    history: pd.Series,
    exog: pd.DataFrame,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a direct (non-decomposed) feature set from raw table fields only.

    Uses lag-1 operational covariates so same-day future leakage is avoided.
    Calendar fields that are known in advance are allowed at the forecast date.
    """
    if dates is None:
        dates = history.index

    exog = exog.reindex(dates).copy()
    for col in exog.columns:
        exog[col] = exog[col].interpolate(method="time").ffill().bfill()

    frame = pd.DataFrame(index=dates)
    for col in exog.columns:
        if col in {"is_weekend", "bank_holiday_flag"}:
            # Known calendar flags can be used on the forecast day.
            frame[col] = exog[col]
        else:
            frame[f"{col}_lag1"] = exog[col].shift(1)

    # Lightweight deterministic calendar encodings (still not STL components).
    dow = frame.index.dayofweek
    doy = frame.index.dayofyear
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return frame


def hgb_raw_table_forecast(
    history: pd.Series,
    exog_hist: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Single HGB trained on raw-table features with no STL decomposition."""
    history = history.astype(float).copy()
    train_features = build_raw_table_feature_frame(history, exog_hist)
    train_df = train_features.join(history.rename("y")).dropna()
    feature_cols = [c for c in train_df.columns if c != "y"]

    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["y"])

    exog_history = exog_hist.reindex(history.index).copy()
    for col in exog_history.columns:
        exog_history[col] = exog_history[col].interpolate(method="time").ffill().bfill()
    last_exog = exog_history.iloc[-1]

    preds = []
    for _ in range(horizon):
        next_date = history.index.max() + pd.Timedelta(days=1)
        future_exog_row = pd.DataFrame([last_exog], index=pd.DatetimeIndex([next_date]))
        # Keep known calendar flags accurate for the future date when available in the original table.
        if "is_weekend" in future_exog_row.columns:
            future_exog_row.loc[next_date, "is_weekend"] = int(next_date.dayofweek >= 5)
        if "bank_holiday_flag" in future_exog_row.columns and next_date in exog_hist.index:
            future_exog_row.loc[next_date, "bank_holiday_flag"] = exog_hist.loc[next_date, "bank_holiday_flag"]

        exog_extended = pd.concat([exog_history, future_exog_row])
        dates = history.index.append(pd.DatetimeIndex([next_date]))
        X_next = build_raw_table_feature_frame(history, exog_extended, dates=dates).loc[[next_date], feature_cols]
        yhat = max(0.0, float(model.predict(X_next)[0]))
        history.loc[next_date] = yhat
        exog_history = exog_extended
        preds.append((next_date, yhat))

    return pd.Series(dict(preds), name="prediction").sort_index()


def build_component_feature_frame(
    complaints_series: pd.Series,
    exog: pd.DataFrame,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Create causal complaint-history, lagged-exogenous, and calendar features."""
    if dates is None:
        dates = complaints_series.index

    features = build_feature_frame(complaints_series, dates=dates).drop(columns="y")
    exog = exog.reindex(dates).copy()
    for col in exog.columns:
        exog[col] = exog[col].interpolate(method="time").ffill().bfill()
        features[f"{col}_lag1"] = exog[col].shift(1)
    return features


def stl_component_targets(history: pd.Series, period: int = 7) -> pd.DataFrame:
    """Decompose historical complaints into the three additive STL component targets."""
    fitted = STL(history.astype(float), period=period, robust=True).fit()
    return pd.DataFrame(
        {
            "tendency": fitted.trend,
            "period": fitted.seasonal,
            "noise": fitted.resid,
        },
        index=history.index,
    )


def select_component_features(features: pd.DataFrame, components: pd.DataFrame, top_n: int = 8):
    """Select each component's top absolute Pearson-correlated causal features."""
    joined = features.join(components).dropna()
    corr = joined.corr(numeric_only=True)
    selected = {}
    records = []

    for component in components.columns:
        component_corr = corr[component].drop(labels=list(components.columns), errors="ignore")
        component_corr = component_corr.dropna().sort_values(key=np.abs, ascending=False)
        selected[component] = component_corr.head(top_n).index.tolist()
        records.extend(
            {
                "component": component,
                "feature": feature,
                "pearson_correlation": value,
                "abs_correlation": abs(value),
                "selected_for_model": feature in selected[component],
            }
            for feature, value in component_corr.items()
        )

    return selected, pd.DataFrame(records)


def component_hgb_forecast(
    history: pd.Series,
    exog_hist: pd.DataFrame,
    horizon: int,
    period: int = 7,
    top_n: int = 8,
):
    """Forecast STL tendency/period/noise using separately selected causal features."""
    history = history.astype(float).copy()
    components = stl_component_targets(history, period=period)
    train_features = build_component_feature_frame(history, exog_hist)
    selected, component_corr = select_component_features(train_features, components, top_n=top_n)

    models = {}
    for component, cols in selected.items():
        train_data = train_features[cols].join(components[[component]]).dropna()
        model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train_data[cols], train_data[component])
        models[component] = model

    # Future exogenous values are unknown; hold their last observed values constant.
    exog_history = exog_hist.reindex(history.index).copy()
    for col in exog_history.columns:
        exog_history[col] = exog_history[col].interpolate(method="time").ffill().bfill()
    last_exog = exog_history.iloc[-1]

    forecast_rows = []
    for _ in range(horizon):
        next_date = history.index.max() + pd.Timedelta(days=1)
        future_exog_row = pd.DataFrame([last_exog], index=pd.DatetimeIndex([next_date]))
        exog_extended = pd.concat([exog_history, future_exog_row])
        dates = history.index.append(pd.DatetimeIndex([next_date]))
        next_features = build_component_feature_frame(history, exog_extended, dates=dates).loc[[next_date]]

        row = {"date": next_date}
        for component, cols in selected.items():
            row[component] = float(models[component].predict(next_features[cols])[0])
        row["prediction"] = max(0.0, row["tendency"] + row["period"] + row["noise"])

        history.loc[next_date] = row["prediction"]
        exog_history = exog_extended
        forecast_rows.append(row)

    component_forecast = pd.DataFrame(forecast_rows).set_index("date")
    return component_forecast["prediction"], component_forecast, component_corr, selected


def generate_residual_bootstrap_paths(
    history: pd.Series,
    component_forecast: pd.DataFrame,
    n_paths: int = 30,
    random_state: int = 42,
):
    """Keep tendency+period(+model noise), restore fluctuation via residual bootstrap.

    deterministic_structure_t = tendency_t + period_t + model_noise_t
    fluctuated_path_t = max(0, deterministic_structure_t + bootstrap_residual_t * vol_scale)

    vol_scale uses recent complaint volatility relative to historical residual std.
    """
    history = history.astype(float)
    stl_fit = STL(history, period=7, robust=True).fit()
    hist_resid = pd.Series(stl_fit.resid, index=history.index).dropna()
    hist_resid = hist_resid - hist_resid.mean()

    recent_vol = float(history.diff().dropna().iloc[-28:].std())
    base_vol = float(hist_resid.std())
    vol_scale = recent_vol / base_vol if base_vol > 1e-8 else 1.0
    vol_scale = float(np.clip(vol_scale, 0.5, 2.0))

    structure = (
        component_forecast["tendency"]
        + component_forecast["period"]
        + component_forecast["noise"]
    )
    deterministic = structure.clip(lower=0.0).rename("deterministic_forecast")

    rng = np.random.default_rng(random_state)
    path_cols = {}
    for i in range(n_paths):
        boot = rng.choice(hist_resid.to_numpy(), size=len(structure), replace=True)
        path = (structure + boot * vol_scale).clip(lower=0.0)
        path_cols[f"path_{i+1}"] = path

    paths_df = pd.DataFrame(path_cols, index=structure.index)
    summary = pd.DataFrame(
        {
            "deterministic_forecast": deterministic,
            "fluctuated_mean": paths_df.mean(axis=1),
            "fluctuated_p10": paths_df.quantile(0.10, axis=1),
            "fluctuated_p50": paths_df.quantile(0.50, axis=1),
            "fluctuated_p90": paths_df.quantile(0.90, axis=1),
        },
        index=structure.index,
    )
    return paths_df, summary, vol_scale
