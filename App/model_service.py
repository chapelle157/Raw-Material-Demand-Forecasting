from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from data_loader import PROJECT_ROOT, add_sku_column


MODEL_PATH = PROJECT_ROOT / "hybrid_model.pkl"
TFT_BEST_PATH = PROJECT_ROOT / "tft_best (3).ckpt"
TFT_FINAL_PATH = PROJECT_ROOT / "tft_final (1).ckpt"
TARGET_COL = "Order_Demand"
ID_COLS = ["Product_Code", "Warehouse", "Date", TARGET_COL]


@dataclass
class ModelRuntime:
    architecture: str
    bundle: dict[str, Any] | None = None
    tft_model: Any | None = None
    feature_names: list[str] = field(default_factory=list)
    model_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    status: str = "Not loaded"


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(y_log), 0)


def load_model_runtime(architecture: str, metadata: dict | None = None) -> ModelRuntime:
    runtime = ModelRuntime(architecture=architecture)
    metadata = metadata or {}

    if architecture.startswith("TFT"):
        runtime.model_path = TFT_BEST_PATH if TFT_BEST_PATH.exists() else TFT_FINAL_PATH
        if not runtime.model_path.exists():
            runtime.warnings.append(f"TFT checkpoint not found: {runtime.model_path}")
            return runtime
        try:
            import torch
            from pytorch_forecasting import TemporalFusionTransformer

            try:
                runtime.tft_model = TemporalFusionTransformer.load_from_checkpoint(
                    str(runtime.model_path),
                    map_location=torch.device("cpu"),
                )
            except AssertionError as exc:
                if "Torch not compiled with CUDA enabled" not in str(exc):
                    raise
                runtime.tft_model = _load_tft_cpu_sanitized(TemporalFusionTransformer, runtime.model_path)
            runtime.feature_names = _tft_feature_names(runtime.tft_model)
            runtime.status = f"Loaded TFT checkpoint from {runtime.model_path.name}"
        except Exception as exc:  # noqa: BLE001
            runtime.warnings.append(
                "Could not load TFT checkpoint. Install compatible torch, lightning, and "
                f"pytorch-forecasting packages in the dashboard environment. Details: {exc}"
            )
            runtime.feature_names = _metadata_features(metadata)
        return runtime

    runtime.model_path = MODEL_PATH
    if not MODEL_PATH.exists():
        runtime.warnings.append(f"Model file not found: {MODEL_PATH}")
        runtime.feature_names = _metadata_features(metadata)
        return runtime

    try:
        with MODEL_PATH.open("rb") as f:
            bundle = pickle.load(f)
        runtime.bundle = bundle
        runtime.feature_names = list(bundle.get("model_features") or [])
        if not runtime.feature_names:
            runtime.feature_names = _metadata_features(metadata)
            runtime.warnings.append("Model bundle has no model_features key; using metadata feature list.")
        runtime.status = f"Loaded hybrid LightGBM bundle from {MODEL_PATH.name}"
    except Exception as exc:  # noqa: BLE001
        runtime.warnings.append(f"Could not load {MODEL_PATH.name}: {exc}")
        runtime.feature_names = _metadata_features(metadata)
    return runtime


def _load_tft_cpu_sanitized(model_cls: Any, checkpoint_path: Path) -> Any:
    import torch

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    for container_name in ["hyper_parameters", "__special_save__"]:
        container = checkpoint.get(container_name)
        if isinstance(container, dict):
            container.pop("loss", None)
            container.pop("logging_metrics", None)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as temp_file:
            temp_path = temp_file.name
        torch.save(checkpoint, temp_path)
        return model_cls.load_from_checkpoint(temp_path, map_location=torch.device("cpu"))
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _metadata_features(metadata: dict) -> list[str]:
    selected = metadata.get("selected_features", {}).get("selected_features", [])
    pipeline = metadata.get("pipeline", {}).get("feature_cols", [])
    return list(selected or pipeline)


def _tft_dataset_parameters(model: Any) -> dict[str, Any]:
    hparams = getattr(model, "hparams", {})
    if hasattr(hparams, "get"):
        params = hparams.get("dataset_parameters")
    else:
        params = getattr(hparams, "dataset_parameters", None)
    if params is None:
        params = getattr(model, "dataset_parameters", None)
    if not isinstance(params, dict):
        raise RuntimeError("TFT checkpoint does not include TimeSeriesDataSet parameters.")
    return params


def _tft_feature_names(model: Any) -> list[str]:
    params = _tft_dataset_parameters(model)
    names: list[str] = []
    for key in [
        "static_categoricals",
        "static_reals",
        "time_varying_known_categoricals",
        "time_varying_known_reals",
        "time_varying_unknown_categoricals",
        "time_varying_unknown_reals",
    ]:
        value = params.get(key) or []
        names.extend([item for item in value if item not in {TARGET_COL, "relative_time_idx"}])
    return list(dict.fromkeys(names))


def predict_hybrid(runtime: ModelRuntime, X: pd.DataFrame) -> np.ndarray:
    if runtime.bundle is None:
        raise RuntimeError("Hybrid model bundle is not loaded.")
    missing = [c for c in runtime.feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing[:8]}")
    X_model = X[runtime.feature_names].replace([np.inf, -np.inf], 0).fillna(0)
    cal = runtime.bundle["cal_tuned"]
    reg = runtime.bundle["reg_tuned"]
    threshold = float(runtime.bundle["threshold"])
    nonzero_prob = cal.predict_proba(X_model)[:, 1]
    demand_if_nonzero = inverse_log_transform(reg.predict(X_model))
    return np.where(nonzero_prob < threshold, 0.0, demand_if_nonzero)


def build_forecast_dataset(
    runtime: ModelRuntime,
    historical: pd.DataFrame,
    horizon_days: int,
    residual_std: float,
) -> tuple[pd.DataFrame, list[str]]:
    if runtime.architecture.startswith("TFT"):
        return build_tft_forecast_dataset(runtime, historical, horizon_days, residual_std)

    warnings: list[str] = []
    hist = add_sku_column(historical)
    if hist.empty:
        return pd.DataFrame(), ["No historical data is available."]
    if not runtime.feature_names:
        return pd.DataFrame(), ["No model feature list is available; run inspect_model.py in the model environment."]
    if runtime.bundle is None:
        return pd.DataFrame(), ["Forecast inference skipped because the trained hybrid model could not be loaded."]

    series_state: list[dict[str, Any]] = []
    for sku, group in hist.groupby("SKU", sort=False):
        group = group.sort_values("Date").copy()
        history_values = group[TARGET_COL].fillna(0).clip(lower=0).astype(float).tolist()
        latest = group.iloc[-1].to_dict()
        last_date = pd.to_datetime(latest.get("Date"))
        if pd.isna(last_date):
            warnings.append(f"{sku}: missing latest date; skipped.")
            continue
        series_state.append(
            {
                "sku": sku,
                "latest": latest,
                "latest_features": {
                    feature: _as_float(latest.get(feature, 0.0))
                    for feature in runtime.feature_names
                },
                "last_date": last_date,
                "history_values": history_values,
            }
        )

    rows: list[dict[str, Any]] = []
    for step in range(1, horizon_days + 1):
        feature_rows = []
        active_state = []
        for state in series_state:
            future_date = state["last_date"] + pd.Timedelta(days=step)
            feature_row = _future_feature_row(
                state["latest_features"],
                state["history_values"],
                future_date,
                runtime.feature_names,
            )
            feature_rows.append(feature_row)
            active_state.append((state, future_date, feature_row))

        if not feature_rows:
            continue

        try:
            predictions = predict_hybrid(runtime, pd.DataFrame(feature_rows))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Batch prediction failed at step {step}: {exc}")
            predictions = np.full(len(feature_rows), np.nan)

        for (state, future_date, feature_row), pred_value in zip(active_state, predictions):
            pred = float(pred_value)
            state["history_values"].append(0 if np.isnan(pred) else pred)
            latest = state["latest"]
            lower = max(0.0, pred - 1.28 * residual_std) if np.isfinite(pred) else np.nan
            upper = pred + 1.28 * residual_std if np.isfinite(pred) else np.nan
            rows.append(
                {
                    "SKU": state["sku"],
                    "Product_Code": latest.get("Product_Code"),
                    "Warehouse": latest.get("Warehouse"),
                    "Date": future_date,
                    "HistoricalDemand": np.nan,
                    "ForecastDemand": pred,
                    "LowerConfidence": lower,
                    "UpperConfidence": upper,
                    "IntervalLabel": "Approx. 80% residual interval",
                }
            )

    return pd.DataFrame(rows), warnings


def build_tft_forecast_dataset(
    runtime: ModelRuntime,
    historical: pd.DataFrame,
    horizon_days: int,
    residual_std: float,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if runtime.tft_model is None:
        return pd.DataFrame(), ["Forecast inference skipped because the TFT checkpoint could not be loaded."]

    try:
        from pytorch_forecasting import TimeSeriesDataSet
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), [f"TFT inference requires pytorch-forecasting in this environment: {exc}"]

    hist = add_sku_column(historical)
    if hist.empty:
        return pd.DataFrame(), ["No historical data is available."]

    params = _tft_dataset_parameters(runtime.tft_model)
    max_encoder_length = int(params.get("max_encoder_length") or 52)
    max_prediction_length = int(params.get("max_prediction_length") or 4)
    time_idx_col = str(params.get("time_idx") or "time_idx")
    group_ids = list(params.get("group_ids") or ["Product_Code", "Warehouse"])
    categorical_cols = _tft_categorical_columns(params)
    required_cols = _tft_required_columns(params)
    horizon_periods = max(1, int(np.ceil(horizon_days / 7)))

    prepared_hist = _prepare_tft_history(hist, group_ids, categorical_cols, time_idx_col)
    if prepared_hist.empty:
        return pd.DataFrame(), ["No TFT-compatible historical rows are available."]
    prepared_hist, unknown_warnings = _drop_tft_unknown_categories(prepared_hist, params, categorical_cols, group_ids)
    warnings.extend(unknown_warnings)
    if prepared_hist.empty:
        return pd.DataFrame(), warnings + ["No rows match the categories learned by the TFT checkpoint."]

    series_state = _tft_series_state(prepared_hist, group_ids, time_idx_col)
    rows: list[dict[str, Any]] = []

    for chunk_start in range(1, horizon_periods + 1, max_prediction_length):
        chunk_len = min(max_prediction_length, horizon_periods - chunk_start + 1)
        future_rows: list[dict[str, Any]] = []
        active_state: list[dict[str, Any]] = []

        for state in series_state:
            for offset in range(chunk_start, chunk_start + chunk_len):
                future_date = state["last_date"] + pd.Timedelta(days=7 * offset)
                time_idx = int(state["last_time_idx"] + offset)
                row = _tft_future_row(
                    state,
                    future_date,
                    time_idx,
                    required_cols,
                    categorical_cols,
                    group_ids,
                    time_idx_col,
                )
                future_rows.append(row)
            active_state.append(state)

        predict_frame = pd.concat(
            [
                prepared_hist.groupby(group_ids, group_keys=False).tail(max_encoder_length),
                pd.DataFrame(future_rows),
            ],
            ignore_index=True,
        )
        predict_frame = _finalize_tft_frame(predict_frame, required_cols, categorical_cols, time_idx_col)

        try:
            dataset = TimeSeriesDataSet.from_parameters(
                params,
                predict_frame,
                predict=True,
                stop_randomization=True,
            )
            dataloader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
            predictions = _predict_tft(runtime.tft_model, dataloader)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"TFT batch prediction failed for forecast weeks {chunk_start}-{chunk_start + chunk_len - 1}: {exc}")
            predictions = np.full((len(active_state), chunk_len), np.nan)

        for state_index, state in enumerate(active_state):
            latest = state["latest"]
            state_preds = np.asarray(predictions[state_index]).ravel()[:chunk_len] if state_index < len(predictions) else np.full(chunk_len, np.nan)
            for local_index, pred_value in enumerate(state_preds, start=0):
                offset = chunk_start + local_index
                future_date = state["last_date"] + pd.Timedelta(days=7 * offset)
                time_idx = int(state["last_time_idx"] + offset)
                pred = max(0.0, float(pred_value)) if np.isfinite(pred_value) else np.nan
                lower = max(0.0, pred - 1.28 * residual_std) if np.isfinite(pred) else np.nan
                upper = pred + 1.28 * residual_std if np.isfinite(pred) else np.nan
                state["history_values"].append(0 if np.isnan(pred) else pred)
                row = {
                    "SKU": state["sku"],
                    "Product_Code": latest.get("Product_Code"),
                    "Warehouse": latest.get("Warehouse"),
                    "Date": future_date,
                    "HistoricalDemand": np.nan,
                    "ForecastDemand": pred,
                    "LowerConfidence": lower,
                    "UpperConfidence": upper,
                    "IntervalLabel": "Approx. 80% residual interval",
                    time_idx_col: time_idx,
                }
                rows.append(row)

    forecast = pd.DataFrame(rows)
    if horizon_days < horizon_periods * 7 and not forecast.empty:
        warnings.append("TFT was trained on weekly periods; the selected day horizon was rounded up to full weeks.")
    return forecast, warnings


def _tft_categorical_columns(params: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ["static_categoricals", "time_varying_known_categoricals", "time_varying_unknown_categoricals"]:
        names.extend(params.get(key) or [])
    return list(dict.fromkeys(names))


def _tft_required_columns(params: dict[str, Any]) -> list[str]:
    names = [TARGET_COL, params.get("time_idx") or "time_idx", *(params.get("group_ids") or [])]
    for key in [
        "static_categoricals",
        "static_reals",
        "time_varying_known_categoricals",
        "time_varying_known_reals",
        "time_varying_unknown_categoricals",
        "time_varying_unknown_reals",
    ]:
        names.extend(params.get(key) or [])
    return list(dict.fromkeys(str(name) for name in names if name))


def _drop_tft_unknown_categories(
    hist: pd.DataFrame,
    params: dict[str, Any],
    categorical_cols: list[str],
    group_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    encoders = params.get("categorical_encoders") or {}
    out = hist.copy()
    unknown_mask = pd.Series(False, index=out.index)

    for col in categorical_cols:
        encoder = encoders.get(col)
        classes = getattr(encoder, "classes_", None)
        if classes is None or col not in out.columns:
            continue
        allowed = {str(value) for value in classes.keys()}
        values = out[col].fillna("nan").astype(str)
        col_unknown = ~values.isin(allowed)
        if col_unknown.any():
            unknown_mask |= col_unknown

    if not unknown_mask.any():
        return out, []

    if set(group_ids).issubset(out.columns):
        unknown_groups = out.loc[unknown_mask, group_ids].drop_duplicates()
        out = out.merge(unknown_groups.assign(_drop_tft_unknown=True), on=group_ids, how="left")
        out = out[out["_drop_tft_unknown"].isna()].drop(columns=["_drop_tft_unknown"])
    else:
        out = out.loc[~unknown_mask].copy()

    return out, []


def _prepare_tft_history(hist: pd.DataFrame, group_ids: list[str], categorical_cols: list[str], time_idx_col: str) -> pd.DataFrame:
    out = hist.copy()
    out = _attach_tft_cluster_labels(out)
    if "Date" not in out.columns:
        return pd.DataFrame()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", *[col for col in group_ids if col in out.columns]])
    if out.empty:
        return out
    out = out.sort_values([*group_ids, "Date"]).reset_index(drop=True)
    if time_idx_col not in out.columns:
        min_date = out["Date"].min()
        out[time_idx_col] = ((out["Date"] - min_date).dt.days // 7).astype(int)
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].fillna("nan").astype(str)
    if TARGET_COL in out.columns:
        out[TARGET_COL] = out[TARGET_COL].fillna(0).clip(lower=0).astype(float)
    return out


def _attach_tft_cluster_labels(hist: pd.DataFrame) -> pd.DataFrame:
    if "Demand_Cluster" in hist.columns:
        return hist
    cluster_path = PROJECT_ROOT / "cluster_labels.parquet"
    if not cluster_path.exists() or not {"Product_Code", "Warehouse"}.issubset(hist.columns):
        out = hist.copy()
        out["Demand_Cluster"] = "0"
        return out
    try:
        labels = pd.read_parquet(cluster_path)
        keep = [col for col in ["Product_Code", "Warehouse", "Demand_Cluster"] if col in labels.columns]
        labels = labels[keep].drop_duplicates(["Product_Code", "Warehouse"])
        out = hist.merge(labels, on=["Product_Code", "Warehouse"], how="left")
        out["Demand_Cluster"] = out["Demand_Cluster"].fillna(0).astype(int).astype(str)
        return out
    except Exception:
        out = hist.copy()
        out["Demand_Cluster"] = "0"
        return out


def _tft_series_state(hist: pd.DataFrame, group_ids: list[str], time_idx_col: str) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for _, group in hist.groupby(group_ids, sort=False):
        group = group.sort_values(time_idx_col)
        latest = group.iloc[-1].to_dict()
        values = group[TARGET_COL].fillna(0).clip(lower=0).astype(float).tolist()
        sku = " / ".join(str(latest.get(col, "")) for col in group_ids).strip(" /") or latest.get("SKU", "UNKNOWN")
        states.append(
            {
                "sku": sku,
                "latest": latest,
                "last_date": pd.to_datetime(latest["Date"]),
                "last_time_idx": int(latest[time_idx_col]),
                "history_values": values,
            }
        )
    return states


def _tft_future_row(
    state: dict[str, Any],
    future_date: pd.Timestamp,
    time_idx: int,
    required_cols: list[str],
    categorical_cols: list[str],
    group_ids: list[str],
    time_idx_col: str,
) -> dict[str, Any]:
    latest = state["latest"]
    numeric_feature_names = [
        name
        for name in required_cols
        if name not in categorical_cols and name not in group_ids and name != "Date"
    ]
    latest_numeric_features = {
        name: _as_float(latest.get(name, 0.0))
        for name in numeric_feature_names
    }
    feature_row = _future_feature_row(
        latest_numeric_features,
        state["history_values"],
        future_date,
        numeric_feature_names,
    )
    row: dict[str, Any] = {name: latest.get(name, 0) for name in required_cols}
    row.update(feature_row)
    for col in group_ids:
        row[col] = latest.get(col)
    for col in categorical_cols:
        row[col] = str(latest.get(col, "nan"))
    row["Date"] = future_date
    row[time_idx_col] = time_idx
    row[TARGET_COL] = 0.0
    return row


def _finalize_tft_frame(frame: pd.DataFrame, required_cols: list[str], categorical_cols: list[str], time_idx_col: str) -> pd.DataFrame:
    out = frame.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = "nan" if col in categorical_cols else 0.0
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].fillna("nan").astype(str)
    numeric_cols = [col for col in required_cols if col not in categorical_cols and col != "Date"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    out[time_idx_col] = out[time_idx_col].astype(int)
    out[TARGET_COL] = out[TARGET_COL].clip(lower=0)
    return out


def _predict_tft(model: Any, dataloader: Any) -> np.ndarray:
    prediction = model.predict(
        dataloader,
        mode="prediction",
        return_index=False,
        trainer_kwargs={"accelerator": "cpu", "devices": 1},
    )
    output = getattr(prediction, "output", prediction)
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    return np.asarray(output, dtype=float)


def _future_feature_row(latest_features: dict[str, float], history_values: list[float], future_date: pd.Timestamp, feature_names: list[str]) -> dict[str, float]:
    row = latest_features.copy()
    values = np.asarray(history_values, dtype=np.float64)
    prev = float(values[-1]) if values.size else 0.0

    week = int(future_date.isocalendar().week)
    row.update(
        {
            "Week_Of_Year": week,
            "WeekOfYear": week,
            "Month": int(future_date.month),
            "Quarter": int(future_date.quarter),
            "Year": int(future_date.year),
            "Day": int(future_date.day),
            "DayOfYear": int(future_date.dayofyear),
            "DayOfWeek": int(future_date.dayofweek),
            "Is_Weekend": int(future_date.dayofweek >= 5),
            "Month_Sin": float(np.sin(2 * np.pi * future_date.month / 12)),
            "Month_Cos": float(np.cos(2 * np.pi * future_date.month / 12)),
            "Week_Sin": float(np.sin(2 * np.pi * week / 52)),
            "Week_Cos": float(np.cos(2 * np.pi * week / 52)),
            "Is_Quarter_End": int(future_date.is_quarter_end),
            "Is_Zero_Demand": int(prev == 0),
        }
    )

    for lag in [1, 2, 4, 8, 12, 52]:
        row[f"Demand_Lag{lag}W"] = _lag(values, lag)
    row["Demand_Lag_7"] = _lag(values, 7)
    row["Demand_Lag_14"] = _lag(values, 14)
    row["Demand_Lag_30"] = _lag(values, 30)

    for window in [4, 8, 12]:
        row[f"Demand_RollMean{window}W"] = _rolling_mean(values, window)
        row[f"Demand_RollStd{window}W"] = _rolling_std(values, window)
    row["Demand_Rolling_Mean_7"] = _rolling_mean(values, 7)
    row["Demand_Rolling_Mean_30"] = _rolling_mean(values, 30)
    row["Demand_Trend"] = row["Demand_RollMean4W"] - row["Demand_RollMean12W"]
    row["Demand_Expanding_Mean"] = float(values.mean()) if values.size else 0.0
    yoy_lag = _lag(values, 53)
    row["Demand_YoY_Growth"] = (prev - yoy_lag) / (abs(yoy_lag) + 1)
    row["Zero_Rate_7d"] = _zero_rate(values, 7)
    row["Zero_Rate_30d"] = _zero_rate(values, 30)
    row["Demand_Velocity_7d"] = prev - _lag(values, 7)
    periods_since_positive = _periods_since_positive(values)
    row["Days_Since_Last_Sale"] = periods_since_positive
    row["Weeks_Since_Last_Order"] = periods_since_positive
    row["Consecutive_Zeros"] = _consecutive_zeros(values)

    return {name: float(row.get(name, 0.0)) for name in feature_names}


def _as_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _lag(values: np.ndarray, lag: int) -> float:
    return float(values[-lag]) if values.size >= lag else 0.0


def _rolling_mean(values: np.ndarray, window: int) -> float:
    return float(values[-window:].mean()) if values.size else 0.0


def _rolling_std(values: np.ndarray, window: int) -> float:
    value = float(values[-window:].std(ddof=0)) if values.size else 0.0
    return value if np.isfinite(value) else 0.0


def _zero_rate(values: np.ndarray, window: int) -> float:
    if not values.size:
        return 0.0
    tail = values[-window:]
    return float((tail == 0).mean())


def _periods_since_positive(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    positives = np.flatnonzero(values > 0)
    return float(values.size - positives[-1]) if positives.size else float(values.size)


def _consecutive_zeros(values: np.ndarray) -> float:
    count = 0
    for value in values[::-1]:
        if value == 0:
            count += 1
        else:
            break
    return float(count)


def compute_holdout_metrics(runtime: ModelRuntime, test: pd.DataFrame) -> dict[str, float | str]:
    if runtime.architecture.startswith("TFT"):
        return {
            "label": "TFT Forecast",
            "value": np.nan,
            "residual_std": 0.0,
            "detail": "TFT holdout scoring is not run inside the dashboard.",
        }
    if test.empty or TARGET_COL not in test.columns:
        return {"label": "Holdout WAPE", "value": np.nan, "detail": "No test dataset"}
    try:
        y_true = test[TARGET_COL].fillna(0).clip(lower=0).astype(float).to_numpy()
        y_pred = predict_hybrid(runtime, test)
        abs_error = np.abs(y_true - y_pred)
        wape = 100 * abs_error.sum() / max(np.abs(y_true).sum(), 1.0)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        residual_std = float(np.nanstd(y_true - y_pred))
        return {
            "label": "Holdout Accuracy (100-WAPE)",
            "value": max(0.0, 100 - wape),
            "wape": wape,
            "rmse": rmse,
            "residual_std": residual_std if np.isfinite(residual_std) and residual_std > 0 else 0.0,
            "detail": "Computed from test_preprocessed.parquet",
        }
    except Exception as exc:  # noqa: BLE001
        return {"label": "Holdout Accuracy", "value": np.nan, "residual_std": 0.0, "detail": str(exc)}


def historical_dashboard_frame(historical: pd.DataFrame) -> pd.DataFrame:
    hist = add_sku_column(historical)
    if hist.empty:
        return hist
    keep_cols = [c for c in ["SKU", "Product_Code", "Warehouse", "Date", TARGET_COL] if c in hist.columns]
    out = hist[keep_cols].rename(columns={TARGET_COL: "HistoricalDemand"}).copy()
    out["ForecastDemand"] = np.nan
    out["LowerConfidence"] = np.nan
    out["UpperConfidence"] = np.nan
    return out


def compute_clusters(historical: pd.DataFrame) -> pd.DataFrame:
    hist = add_sku_column(historical)
    if hist.empty:
        return pd.DataFrame()
    stats = (
        hist.groupby("SKU")[TARGET_COL]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Volume (Weekly Average)", "std": "Demand Std"})
        .reset_index()
    )
    stats["Volatility (COV %)"] = np.where(
        stats["Volume (Weekly Average)"].abs() > 0,
        stats["Demand Std"].fillna(0) / stats["Volume (Weekly Average)"].abs() * 100,
        0,
    )
    try:
        from sklearn.cluster import KMeans

        n_clusters = min(4, max(1, len(stats)))
        X = stats[["Volume (Weekly Average)", "Volatility (COV %)"]].fillna(0)
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)
    except Exception:
        labels = _quantile_cluster(stats)
    stats["Cluster"] = labels.astype(int)
    stats["Material SKU"] = stats["SKU"]
    stats["Group"] = stats["Cluster"].map(cluster_name)
    return stats


def _quantile_cluster(stats: pd.DataFrame) -> np.ndarray:
    volume_rank = stats["Volume (Weekly Average)"].rank(pct=True)
    vol_rank = stats["Volatility (COV %)"].rank(pct=True)
    return np.select(
        [
            (volume_rank >= 0.6) & (vol_rank < 0.5),
            vol_rank >= 0.75,
            (volume_rank >= 0.4) & (vol_rank >= 0.5),
        ],
        [0, 1, 2],
        default=3,
    )


def cluster_name(cluster_id: int) -> str:
    names = {
        0: "Cluster 0: Active & Stable",
        1: "Cluster 1: Intermittent / Sparse",
        2: "Cluster 2: High Seasonal Peak",
        3: "Cluster 3: Emerging Trend",
    }
    return names.get(int(cluster_id), f"Cluster {cluster_id}: Demand Segment")


def compute_global_importance(runtime: ModelRuntime, reference: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    features = runtime.feature_names
    if runtime.architecture.startswith("TFT"):
        return pd.DataFrame(columns=["feature", "importance"]), "TFT attention interpretation unavailable"
    if not features:
        return pd.DataFrame(columns=["feature", "importance"]), "Unavailable"
    try:
        import shap

        model = runtime.bundle["reg_tuned"] if runtime.bundle else None
        sample = reference[features].replace([np.inf, -np.inf], 0).fillna(0).tail(2000)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        importance = np.abs(values).mean(axis=0)
        label = "SHAP TreeExplainer"
    except Exception:
        model = runtime.bundle.get("reg_tuned") if runtime.bundle else None
        if model is not None and hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
            label = "Model Gain Importance"
        else:
            importance = np.ones(len(features), dtype=float)
            label = "Uniform Importance (model unavailable)"
    total = float(np.sum(importance))
    weights = importance / total if total > 0 else np.zeros(len(features))
    return (
        pd.DataFrame({"feature": features, "importance": weights})
        .sort_values("importance", ascending=False)
        .head(12)
        .sort_values("importance"),
        label,
    )


def compute_local_contributions(runtime: ModelRuntime, row: pd.Series, reference: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    features = runtime.feature_names
    if runtime.architecture.startswith("TFT"):
        return pd.DataFrame(), "TFT local explanation unavailable"
    if not features or row.empty:
        return pd.DataFrame(), "Unavailable"
    x = pd.DataFrame([{f: row.get(f, 0.0) for f in features}]).replace([np.inf, -np.inf], 0).fillna(0)
    try:
        import shap

        model = runtime.bundle["reg_tuned"]
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(x))[0]
        base = float(np.asarray(explainer.expected_value).ravel()[0])
        label = "SHAP TreeExplainer"
    except Exception:
        values = np.zeros(len(features), dtype=float)
        base = 0.0
        label = "Local explanation unavailable"
    data = pd.DataFrame({"feature": features, "contribution": values})
    top = data.reindex(data["contribution"].abs().sort_values(ascending=False).index).head(4)
    return pd.concat([pd.DataFrame({"feature": ["Base Model Value"], "contribution": [base]}), top]), label


def compute_tft_explainability(
    runtime: ModelRuntime,
    historical: pd.DataFrame,
    selected_sku: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    warnings: list[str] = []
    empty = {
        "variables": pd.DataFrame(columns=["feature", "importance", "source"]),
        "attention": pd.DataFrame(columns=["horizon", "encoder_step", "attention"]),
    }
    if not runtime.architecture.startswith("TFT") or runtime.tft_model is None:
        return empty, ["TFT checkpoint is not loaded."]

    try:
        from pytorch_forecasting import TimeSeriesDataSet
    except Exception as exc:  # noqa: BLE001
        return empty, [f"TFT explainability requires pytorch-forecasting: {exc}"]

    hist = add_sku_column(historical)
    hist = hist[hist["SKU"] == selected_sku].copy() if "SKU" in hist.columns else pd.DataFrame()
    if hist.empty:
        return empty, [f"No historical rows found for {selected_sku}."]

    try:
        params = _tft_dataset_parameters(runtime.tft_model)
        max_encoder_length = int(params.get("max_encoder_length") or 52)
        max_prediction_length = int(params.get("max_prediction_length") or 4)
        time_idx_col = str(params.get("time_idx") or "time_idx")
        group_ids = list(params.get("group_ids") or ["Product_Code", "Warehouse"])
        categorical_cols = _tft_categorical_columns(params)
        required_cols = _tft_required_columns(params)

        prepared_hist = _prepare_tft_history(hist, group_ids, categorical_cols, time_idx_col)
        prepared_hist, unknown_warnings = _drop_tft_unknown_categories(prepared_hist, params, categorical_cols, group_ids)
        warnings.extend(unknown_warnings)
        if prepared_hist.empty:
            return empty, warnings + ["Selected SKU is not compatible with the TFT checkpoint categories."]

        state = _tft_series_state(prepared_hist, group_ids, time_idx_col)[0]
        future_rows = [
            _tft_future_row(
                state,
                state["last_date"] + pd.Timedelta(days=7 * offset),
                int(state["last_time_idx"] + offset),
                required_cols,
                categorical_cols,
                group_ids,
                time_idx_col,
            )
            for offset in range(1, max_prediction_length + 1)
        ]
        predict_frame = pd.concat(
            [
                prepared_hist.groupby(group_ids, group_keys=False).tail(max_encoder_length),
                pd.DataFrame(future_rows),
            ],
            ignore_index=True,
        )
        predict_frame = _finalize_tft_frame(predict_frame, required_cols, categorical_cols, time_idx_col)
        dataset = TimeSeriesDataSet.from_parameters(
            params,
            predict_frame,
            predict=True,
            stop_randomization=True,
        )
        dataloader = dataset.to_dataloader(train=False, batch_size=8, num_workers=0)
        raw = runtime.tft_model.predict(
            dataloader,
            mode="raw",
            return_x=True,
            return_index=True,
            trainer_kwargs={"accelerator": "cpu", "devices": 1},
        )

        base_interpretation = runtime.tft_model.interpret_output(raw.output, reduction="sum")
        variable_rows = _tft_variable_importance_rows(runtime.tft_model, base_interpretation)

        attention_rows: list[dict[str, Any]] = []
        for horizon_idx in range(max_prediction_length):
            interpretation = runtime.tft_model.interpret_output(
                raw.output,
                reduction="sum",
                attention_prediction_horizon=horizon_idx,
            )
            attention = interpretation["attention"].detach().cpu().numpy()[:max_encoder_length].astype(float)
            total = float(attention.sum())
            if total > 0:
                attention = attention / total
            for step, weight in enumerate(attention):
                attention_rows.append(
                    {
                        "horizon": f"t+{horizon_idx + 1}",
                        "encoder_step": int(step - max_encoder_length + 1),
                        "attention": float(weight),
                    }
                )

        return (
            {
                "variables": variable_rows,
                "attention": pd.DataFrame(attention_rows),
            },
            warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return empty, warnings + [f"TFT explainability failed: {exc}"]


def _tft_variable_importance_rows(model: Any, interpretation: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        ("static_variables", getattr(model, "static_variables", []), "static"),
        ("encoder_variables", getattr(model, "encoder_variables", []), "encoder"),
        ("decoder_variables", getattr(model, "decoder_variables", []), "decoder"),
    ]
    for key, names, source in sources:
        values = interpretation.get(key)
        if values is None:
            continue
        values_np = values.detach().cpu().numpy().astype(float).ravel()
        for name, importance in zip(names, values_np):
            rows.append({"feature": str(name), "importance": float(importance), "source": source})

    if not rows:
        return pd.DataFrame(columns=["feature", "importance", "source"])

    data = pd.DataFrame(rows)
    data = data.groupby("feature", as_index=False).agg(
        importance=("importance", "max"),
        source=("source", lambda values: " / ".join(sorted(set(values)))),
    )
    total = float(data["importance"].sum())
    if total > 0:
        data["importance"] = data["importance"] / total
    return data.sort_values("importance", ascending=False).head(12).sort_values("importance")


def compute_inventory_policy(
    forecast: pd.DataFrame,
    selected_sku: str,
    lead_time: int,
    service_level: int,
    multiplier: float,
    holding_cost: float,
    stockout_cost: float,
    historical: pd.DataFrame | None = None,
) -> dict[str, Any]:
    z = NormalDist().inv_cdf(service_level / 100)
    required_columns = {"SKU", "Date", "ForecastDemand", "LowerConfidence", "UpperConfidence"}
    if forecast.empty or not required_columns.issubset(forecast.columns):
        return _unavailable_inventory_policy()
    sku_fc = forecast[forecast["SKU"] == selected_sku].sort_values("Date")
    if sku_fc.empty:
        return _unavailable_inventory_policy()

    forecast_period_days = _median_period_days(sku_fc)
    demand = sku_fc["ForecastDemand"].fillna(0).clip(lower=0).to_numpy()
    demand_daily = demand / forecast_period_days

    upper_gap = sku_fc["UpperConfidence"] - sku_fc["ForecastDemand"]
    lower_gap = sku_fc["ForecastDemand"] - sku_fc["LowerConfidence"]
    band_std = pd.concat([upper_gap, lower_gap], axis=1).max(axis=1) / 1.28
    band_std = band_std.replace([np.inf, -np.inf], np.nan)
    valid_band_std = band_std[band_std > 0]

    uncertainty_source = "forecast_interval"
    if not valid_band_std.empty:
        daily_uncertainty = float(valid_band_std.mean()) / np.sqrt(forecast_period_days)
    else:
        daily_uncertainty = _historical_daily_uncertainty(historical, selected_sku)
        uncertainty_source = "historical_variability" if np.isfinite(daily_uncertainty) else "unavailable"
        if uncertainty_source == "unavailable":
            return _unavailable_inventory_policy()

    daily_mean = float(np.mean(demand_daily)) if len(demand_daily) else 0.0
    lead_time_demand = daily_mean * lead_time
    lead_time_std = daily_uncertainty * np.sqrt(max(lead_time, 1))
    safety_stock = z * lead_time_std * multiplier
    reorder_point = lead_time_demand + safety_stock
    expected_holding_cost = (safety_stock + lead_time_demand / 2) * holding_cost
    expected_stockout_penalty = lead_time_std * stockout_cost * (1 - service_level / 100)

    inventory = reorder_point + safety_stock
    rows = []
    for day in range(30):
        consumption = float(demand_daily[day % len(demand_daily)]) if len(demand_daily) else 0.0
        inventory = max(0.0, inventory - consumption)
        if inventory <= reorder_point:
            inventory += lead_time_demand
        rows.append({"Day": day + 1, "Inventory Level": inventory, "Safety Stock Threshold": safety_stock, "Reorder Point": reorder_point})
    return {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "holding_cost": expected_holding_cost,
        "stockout_penalty": expected_stockout_penalty,
        "total_cost": expected_holding_cost + expected_stockout_penalty,
        "uncertainty_source": uncertainty_source,
        "series": pd.DataFrame(rows),
    }


def _unavailable_inventory_policy() -> dict[str, Any]:
    return {
        "safety_stock": np.nan,
        "reorder_point": np.nan,
        "holding_cost": np.nan,
        "stockout_penalty": np.nan,
        "total_cost": np.nan,
        "uncertainty_source": "unavailable",
        "series": pd.DataFrame(),
    }


def _median_period_days(frame: pd.DataFrame) -> float:
    dates = pd.to_datetime(frame["Date"], errors="coerce").dropna().drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna()
    positive_gaps = gaps[gaps > 0]
    return float(positive_gaps.median()) if not positive_gaps.empty else 1.0


def _historical_daily_uncertainty(historical: pd.DataFrame | None, selected_sku: str) -> float:
    if historical is None or historical.empty:
        return np.nan
    hist = add_sku_column(historical)
    sku_hist = hist[hist["SKU"] == selected_sku].sort_values("Date").tail(52)
    if TARGET_COL not in sku_hist:
        return np.nan
    values = pd.to_numeric(sku_hist[TARGET_COL], errors="coerce").dropna().clip(lower=0)
    if len(values) < 2:
        return np.nan
    period_std = float(values.std(ddof=1))
    return period_std / np.sqrt(_median_period_days(sku_hist)) if np.isfinite(period_std) else np.nan


def compute_executive_summary(forecast: pd.DataFrame, historical: pd.DataFrame) -> dict[str, Any]:
    if forecast.empty:
        return {"forecast_total": 0.0, "growth_pct": 0.0, "risk_count": 0, "overstock_count": 0, "action": "No forecast available", "narrative": "Forecast output is unavailable."}
    hist = add_sku_column(historical)
    horizon = forecast["Date"].nunique()
    forecast_total = float(forecast["ForecastDemand"].fillna(0).sum())
    hist_recent = hist.sort_values("Date").groupby("SKU").tail(horizon)
    hist_total = float(hist_recent[TARGET_COL].fillna(0).sum()) if not hist_recent.empty else 0.0
    growth_pct = (forecast_total - hist_total) / hist_total * 100 if hist_total else 0.0
    sku_fc = forecast.groupby("SKU")["ForecastDemand"].sum()
    sku_hist = hist_recent.groupby("SKU")[TARGET_COL].sum() if not hist_recent.empty else pd.Series(dtype=float)
    risk = (sku_fc > sku_hist.reindex(sku_fc.index).fillna(0) * 1.15).sort_values(ascending=False)
    overstock = (sku_fc < sku_hist.reindex(sku_fc.index).fillna(0) * 0.85).sort_values(ascending=False)
    accelerate = sku_fc.sort_values(ascending=False).head(1).index.tolist()
    delay = sku_fc.sort_values(ascending=True).head(1).index.tolist()
    action = f"Accelerate {accelerate[0]}" if accelerate else "Hold sourcing plan"
    if delay and delay[0] != accelerate[0]:
        action += f", Delay {delay[0]}"
    return {
        "forecast_total": forecast_total,
        "growth_pct": growth_pct,
        "risk_count": int(risk.sum()),
        "overstock_count": int(overstock.sum()),
        "action": action,
        "narrative": (
            f"Aggregate demand is expected to change by {growth_pct:+.1f}% over the selected horizon. "
            f"{int(risk.sum())} SKU(s) exceed their recent demand baseline and should be reviewed for replenishment."
        ),
    }
