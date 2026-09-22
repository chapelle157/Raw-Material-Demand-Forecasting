from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import DATA_PATHS, DemandData, load_demand_data, validate_uploaded_csv
from model_service import (
    TARGET_COL,
    build_forecast_dataset,
    compute_executive_summary,
    compute_global_importance,
    compute_holdout_metrics,
    compute_inventory_policy,
    compute_tft_explainability,
    historical_dashboard_frame,
    load_model_runtime,
)
from ui_components import render_kpi, render_metric, render_section_end, render_section_start


st.set_page_config(
    page_title="Raw Material Demand Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #111111 !important;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
        .stApp, .main, .block-container, [data-testid="stMarkdownContainer"],
        [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] { color: #0f172a !important; }
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *, [data-testid="stRadio"] label, [data-testid="stRadio"] label *,
        [data-testid="stSlider"] label, [data-testid="stSlider"] *, [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] label *, [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section *,
        [data-testid="stDownloadButton"] button { color: #0f172a !important; }
        [data-baseweb="select"], [data-baseweb="select"] *, [data-baseweb="input"],
        [data-baseweb="input"] *, [data-baseweb="textarea"], [data-baseweb="textarea"] *,
        [data-baseweb="popover"], [data-baseweb="popover"] *, [role="listbox"], [role="listbox"] * {
            color: #0f172a !important;
        }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] textarea,
        [data-testid="stTextArea"] textarea, [data-testid="stFileUploader"] section {
            background-color: #ffffff !important; border-color: #cbd5e1 !important;
        }
        [data-baseweb="select"] svg, [data-testid="stRadio"] svg, [data-testid="stSlider"] svg {
            color: #334155 !important; fill: currentColor !important;
        }
        [data-baseweb="input"] input::placeholder, [data-baseweb="textarea"] textarea::placeholder,
        [data-testid="stTextArea"] textarea::placeholder { color: #64748b !important; opacity: 1 !important; }
        [data-testid="stSlider"] [role="slider"] { background-color: #0d6efd !important; border-color: #0d6efd !important; }
        [data-testid="stMetric"], [data-testid="stMetric"] * { color: #0f172a !important; }
        [data-testid="stAlert"] {
            background-color: #eff6ff !important; color: #0f172a !important;
            border: 1px solid #bfdbfe !important;
        }
        [data-testid="stAlert"] * { color: #0f172a !important; }
        div[data-testid="stSidebar"] {
            color: #111111; background-color: rgba(255, 255, 255, 0.96) !important;
            border-right: 1px solid rgba(200, 200, 200, 0.6);
        }
        .section-panel {
            background: rgba(248, 250, 252, 0.95); color: #111111;
            border: 1px solid rgba(200, 200, 200, 0.6); border-radius: 8px;
            padding: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 24px;
        }
        .kpi-container {
            display: flex; flex-direction: column; justify-content: center;
            background: rgba(241, 245, 249, 0.96);
            border: 1px solid rgba(203, 213, 225, 0.8); border-radius: 8px;
            padding: 16px; min-height: 118px; color: #111111;
        }
        .kpi-title, .metric-title {
            font-size: 0.75rem; color: #334155; text-transform: uppercase;
            letter-spacing: 0.04em; margin-bottom: 4px;
        }
        .kpi-value { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
        .kpi-delta { font-size: 0.8rem; font-weight: 500; color: #334155; margin-top: 4px; }
        .metric-value { margin-top: 0; color: #0f172a; }
        .stButton > button {
            background: #0d6efd !important; color: #ffffff !important; font-weight: 700 !important;
            border: none !important; border-radius: 8px !important; padding: 8px 24px !important;
        }
        .stDownloadButton > button {
            background: #ffffff !important; color: #0f172a !important; font-weight: 700 !important;
            border: 1px solid #cbd5e1 !important; border-radius: 8px !important; padding: 8px 20px !important;
        }
        .summary-panel {
            background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px; padding: 24px; margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_data():
    return load_demand_data()


@st.cache_resource(show_spinner=False)
def cached_runtime(architecture: str, metadata: dict, model_version: int):
    return load_model_runtime(architecture, metadata)


@st.cache_data(show_spinner=False)
def cached_uploaded_validation(file_bytes: bytes):
    return validate_uploaded_csv(file_bytes)


def plot_theme(fig: go.Figure, height: int, top_margin: int = 30) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155"),
        margin=dict(l=20, r=20, t=top_margin, b=20),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)")
    return fig


def build_inference_state(architecture: str, selected_horizon: int, data_version: int, model_version: int):
    runtime = cached_runtime(architecture, data.metadata, model_version)
    metrics = compute_holdout_metrics(runtime, data.test)
    residual_std = float(metrics.get("residual_std") or 0.0)
    hist_frame = historical_dashboard_frame(data.full)
    forecast_frame, forecast_warnings = build_forecast_dataset(runtime, data.full, selected_horizon, residual_std)
    df_all = pd.concat([hist_frame, forecast_frame], ignore_index=True) if not hist_frame.empty else forecast_frame
    summary = compute_executive_summary(forecast_frame, data.full)
    global_importance, importance_label = compute_global_importance(
        runtime,
        data.test if not data.test.empty else data.full,
    )
    return {
        "key": (architecture, selected_horizon, data_version, model_version),
        "runtime": runtime,
        "metrics": metrics,
        "hist_frame": hist_frame,
        "forecast_frame": forecast_frame,
        "forecast_warnings": forecast_warnings,
        "df_all": df_all,
        "export_csv": forecast_frame.to_csv(index=False) if not forecast_frame.empty else "",
        "summary": summary,
        "global_importance": global_importance,
        "importance_label": importance_label,
    }


def clear_inference_state():
    st.session_state.inference_state = None
    st.session_state.tft_explanation_cache = {}


def select_data_source():
    st.session_state.active_data_source = st.session_state.data_source_selection
    st.session_state.data_version += 1
    clear_inference_state()


def reset_data_source():
    st.session_state.data_source_selection = "Built-in demo dataset"
    st.session_state.active_data_source = "Built-in demo dataset"
    st.session_state.uploaded_dataset = None
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_digest = None
    st.session_state.pending_upload_data = None
    st.session_state.pending_upload_digest = None
    st.session_state.pending_upload_filename = None
    st.session_state.upload_validation_result = None
    st.session_state.data_version += 1
    clear_inference_state()


demo_data = cached_data()
state_defaults = {
    "data_version": 0,
    "model_version": 0,
    "inference_state": None,
    "data_source_selection": "Built-in demo dataset",
    "active_data_source": "Built-in demo dataset",
    "uploaded_dataset": None,
    "uploaded_filename": None,
    "uploaded_digest": None,
    "pending_upload_data": None,
    "pending_upload_digest": None,
    "pending_upload_filename": None,
    "upload_validation_result": None,
    "tft_explanation_cache": {},
}
for state_key, default_value in state_defaults.items():
    st.session_state.setdefault(state_key, default_value)

source_ready = st.session_state.data_source_selection == "Built-in demo dataset"

with st.sidebar:
    st.title("Model Engine Control")
    st.markdown("---")
    st.subheader("Data source")
    st.radio(
        "Choose the dataset used by the dashboard",
        ["Built-in demo dataset", "Upload CSV"],
        key="data_source_selection",
        on_change=select_data_source,
    )

    if st.session_state.data_source_selection == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload demand history",
            type=["csv"],
            key="production_csv_uploader",
            help="Required: date, demand, and SKU/product. Warehouse is optional.",
        )
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            current_digest = hashlib.sha256(uploaded_bytes).hexdigest()
            if current_digest != st.session_state.pending_upload_digest:
                st.session_state.pending_upload_digest = current_digest
                st.session_state.pending_upload_filename = uploaded_file.name
                st.session_state.pending_upload_data = None
                st.session_state.upload_validation_result = None
                st.session_state.active_data_source = "Upload CSV"
                st.session_state.data_version += 1
                clear_inference_state()
                with st.spinner("Validating uploaded CSV..."):
                    validation = cached_uploaded_validation(uploaded_bytes)
                st.session_state.upload_validation_result = validation
                if validation.is_valid:
                    st.session_state.pending_upload_data = validation.data

            validation = st.session_state.upload_validation_result
            if validation is not None:
                if validation.is_valid:
                    st.success("CSV validation passed.")
                    for warning in validation.warnings:
                        st.warning(warning)
                    st.dataframe(validation.data.head(8), hide_index=True)
                    is_confirmed_file = st.session_state.uploaded_digest == current_digest
                    if st.button(
                        "Use uploaded data" if not is_confirmed_file else "Uploaded data is active",
                        disabled=is_confirmed_file,
                        key="confirm_uploaded_data",
                    ):
                        st.session_state.uploaded_dataset = validation.data.copy()
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.session_state.uploaded_digest = current_digest
                        st.session_state.active_data_source = "Upload CSV"
                        st.session_state.data_version += 1
                        clear_inference_state()
                        st.toast("Uploaded data is now active across the dashboard.")
                else:
                    for error in validation.errors:
                        st.error(error)
        else:
            if (
                st.session_state.uploaded_dataset is not None
                and st.session_state.uploaded_digest == st.session_state.pending_upload_digest
            ):
                st.success(f"Using confirmed upload: {st.session_state.uploaded_filename}.")
                st.caption("Upload a different file to validate and replace the active dataset.")
            else:
                st.info("Choose a CSV file, review validation, then confirm it for the dashboard.")

        source_ready = (
            st.session_state.uploaded_dataset is not None
            and st.session_state.uploaded_digest is not None
            and st.session_state.uploaded_digest == st.session_state.pending_upload_digest
        )
        if st.button("Return to demo dataset", key="return_to_demo", on_click=reset_data_source):
            st.toast("Built-in demo dataset restored.")

    if st.session_state.data_source_selection == "Built-in demo dataset":
        status_data = demo_data.full
        status_name = DATA_PATHS["full"].name
        validation_status = "Valid" if not status_data.empty else "Unavailable"
        active_source_label = "Built-in demo dataset"
    elif source_ready:
        status_data = st.session_state.uploaded_dataset
        status_name = st.session_state.uploaded_filename
        validation_status = "Valid and confirmed"
        active_source_label = "Upload CSV"
    else:
        validation = st.session_state.upload_validation_result
        status_data = validation.data if validation is not None and validation.is_valid else pd.DataFrame()
        status_name = st.session_state.pending_upload_filename or "No file selected"
        if validation is None:
            validation_status = "Not ready"
        elif validation.is_valid:
            validation_status = "Valid, awaiting confirmation"
        else:
            validation_status = "Invalid"
        active_source_label = "None — upload not confirmed"

    if "SKU" in status_data:
        status_skus = int(status_data["SKU"].nunique())
    elif {"Product_Code", "Warehouse"}.issubset(status_data.columns):
        status_skus = int(status_data.groupby(["Product_Code", "Warehouse"]).ngroups)
    elif "Product_Code" in status_data:
        status_skus = int(status_data["Product_Code"].nunique())
    else:
        status_skus = 0
    with st.container(border=True):
        st.markdown(f"**Active data source:** {active_source_label}")
        st.caption(f"Dataset/file: {status_name}")
        st.caption(f"Rows: {len(status_data):,}")
        st.caption(f"SKUs: {status_skus:,}")
        st.caption(f"Validation status: {validation_status}")

    st.markdown("---")
    st.subheader("Model Configuration")
    architecture = st.selectbox(
        "Forecast Architecture",
        ["TFT (Temporal Fusion Transformer)", "Hybrid LightGBM Regressor"],
        index=0,
        key="forecast_architecture",
    )
    st.caption("LightGBM is recommended for fast live demos. TFT runs full neural inference and can take longer.")
    horizon_days = st.selectbox("Forecast Horizon", ["7 Days", "14 Days", "30 Days", "60 Days"], index=2)
    selected_horizon = {"7 Days": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60}[horizon_days]

    st.markdown("---")
    st.subheader("Quick Actions")
    left, right = st.columns(2)
    if left.button("Generate", disabled=not source_ready):
        st.session_state.data_version += 1
        clear_inference_state()
        with st.spinner("Running the forecast model..."):
            time.sleep(0.2)
        st.toast("Forecasts refreshed from the loaded model.")
    if right.button("Reload Model"):
        cached_runtime.clear()
        st.session_state.model_version += 1
        clear_inference_state()
        with st.spinner("Reloading the trained model artifact..."):
            time.sleep(0.2)
        st.toast("Model artifact reloaded.")

if st.session_state.data_source_selection == "Built-in demo dataset":
    data = demo_data
else:
    data = DemandData(
        full=st.session_state.uploaded_dataset.copy() if source_ready else pd.DataFrame(),
        warnings=[],
        metadata=demo_data.metadata,
    )

st.markdown(
    """
    <div style="background:#0f172a; border-radius:8px; padding:30px; margin-bottom:30px;">
        <h1 style="margin:0; font-size:2.4rem; font-weight:800; color:#ffffff;">
            RAW MATERIAL DEMAND FORECASTING
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not source_ready:
    st.info("The dashboard is paused until a valid uploaded CSV is confirmed. Built-in results are not being shown for this file.")
    st.stop()

inference_key = (architecture, selected_horizon, st.session_state.data_version, st.session_state.model_version)
if st.session_state.inference_state is None or st.session_state.inference_state.get("key") != inference_key:
    with st.spinner("Loading real demand data and running model inference..."):
        st.session_state.inference_state = build_inference_state(
            architecture,
            selected_horizon,
            st.session_state.data_version,
            st.session_state.model_version,
        )

inference_state = st.session_state.inference_state
runtime = inference_state["runtime"]
metrics = inference_state["metrics"]
hist_frame = inference_state["hist_frame"]
forecast_frame = inference_state["forecast_frame"]
forecast_warnings = inference_state["forecast_warnings"]
df_all = inference_state["df_all"]
export_csv = inference_state["export_csv"]
summary = inference_state["summary"]
global_importance = inference_state["global_importance"]
importance_label = inference_state["importance_label"]

with st.sidebar:
    st.download_button(
        label="Export Demand Data (CSV)",
        data=export_csv,
        file_name="demand_forecasting_data.csv",
        mime="text/csv",
    )
    st.caption(f"Forecasting Engine status: {runtime.status}")

for warning in data.warnings + runtime.warnings + forecast_warnings[:3]:
    st.warning(warning)

active_skus = int(df_all["SKU"].nunique()) if "SKU" in df_all else 0
inventory_health = 0.0 if forecast_frame.empty else max(0.0, min(100.0, 100.0 - abs(summary["growth_pct"])))
kpi_data = [
    ("Total Active SKUs", f"{active_skus:,} Materials", "Loaded from historical dataset", "#334155"),
    ("Active Model", runtime.architecture, runtime.status, "#334155"),
    ("Inventory Health Score", f"{inventory_health:.1f}%", "Forecast-based policy signal", "#15803d"),
]
for column, (title, value, delta, color) in zip(st.columns(3), kpi_data):
    with column:
        render_kpi(title, value, delta, color)

st.markdown("<br>", unsafe_allow_html=True)

if df_all.empty:
    st.error("No real demand data could be loaded. Install parquet/model dependencies and rerun the dashboard.")
    st.stop()

render_section_start(
    "Demand Forecast View",
    "#0f766e",
    "Historical demand patterns and model-generated forecasts with residual-based prediction intervals.",
)
left, right = st.columns(2)
forecast_skus = forecast_frame["SKU"].dropna().unique().tolist() if not forecast_frame.empty and "SKU" in forecast_frame else []
sku_options = forecast_skus or df_all["SKU"].dropna().unique().tolist()
selected_sku = left.selectbox("Select Material / SKU", sku_options, index=0)
date_range_option = right.selectbox("Historical Window View", ["7 Days", "30 Days", "90 Days", "6 Months"], index=2)

sku_df = df_all[df_all["SKU"] == selected_sku].sort_values("Date").copy()
history_window = {"7 Days": 7, "30 Days": 30, "90 Days": 90, "6 Months": 180}[date_range_option]
sku_df_hist = sku_df[sku_df["HistoricalDemand"].notna()].tail(history_window)
sku_df_forecast = sku_df[sku_df["ForecastDemand"].notna() & np.isfinite(sku_df["ForecastDemand"])].head(selected_horizon)
sku_df_combined = pd.concat([sku_df_hist, sku_df_forecast])

predicted_average = sku_df_forecast["ForecastDemand"].mean() if not sku_df_forecast.empty else np.nan
predicted_average_text = f"{predicted_average:,.1f} units" if pd.notna(predicted_average) else "N/A"
historical_std = sku_df_hist["HistoricalDemand"].std()
historical_std = historical_std if pd.notna(historical_std) else 0.0
volatility_status = "High Volatility" if historical_std > 100 else "Stable"
if not sku_df_forecast.empty and not sku_df_hist["HistoricalDemand"].dropna().empty:
    trend_direction = "Upward Trend" if sku_df_forecast["ForecastDemand"].iloc[-1] > sku_df_hist["HistoricalDemand"].dropna().iloc[-1] else "Downward Trend"
else:
    trend_direction = "N/A"
interval_width = sku_df_forecast["UpperConfidence"].sub(sku_df_forecast["LowerConfidence"]).mean() if not sku_df_forecast.empty else np.nan
interval_text = f"Avg band width {interval_width:,.1f} units" if pd.notna(interval_width) else "N/A"

metric_data = [
    ("Predicted Avg Demand", predicted_average_text, "#0f172a"),
    ("Forecast Interval", interval_text, "#15803d"),
    ("Trend Direction", trend_direction, "#0f172a"),
    ("Demand Volatility", f"{volatility_status} ({historical_std:.1f} SD)", "#0f172a"),
]
for column, item in zip(st.columns(4), metric_data):
    with column:
        render_metric(*item)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sku_df_combined["Date"], y=sku_df_combined["HistoricalDemand"], mode="lines+markers", name="Historical Demand", line=dict(color="#0ea5e9", width=3), marker=dict(size=4)))
fig.add_trace(go.Scatter(x=sku_df_combined["Date"], y=sku_df_combined["ForecastDemand"], mode="lines+markers", name="Forecast", line=dict(color="#7c3aed", width=3, dash="dash"), marker=dict(size=4)))
ci_df = sku_df_forecast.dropna(subset=["UpperConfidence", "LowerConfidence"])
if not ci_df.empty:
    fig.add_trace(go.Scatter(x=ci_df["Date"].tolist() + ci_df["Date"].tolist()[::-1], y=ci_df["UpperConfidence"].tolist() + ci_df["LowerConfidence"].tolist()[::-1], fill="toself", fillcolor="rgba(14, 165, 233, 0.10)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=True, name="Approx. 80% Residual Interval"))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(plot_theme(fig, height=450, top_margin=40), width="stretch")
render_section_end()

is_tft_runtime = runtime.architecture.startswith("TFT")
explainability_title = "Forecast Explainability"
explainability_subtitle = (
    "TFT forecasts are interpreted through variable selection and temporal attention."
    if is_tft_runtime
    else "Review the feature importance signals used by the LightGBM forecasting model."
)
render_section_start(explainability_title, "#64748b", explainability_subtitle)
if is_tft_runtime:
    tft_explain_key = (inference_key, selected_sku)
    if "tft_explanation_cache" not in st.session_state:
        st.session_state.tft_explanation_cache = {}
    if tft_explain_key not in st.session_state.tft_explanation_cache:
        with st.spinner("Computing TFT attention interpretation..."):
            st.session_state.tft_explanation_cache[tft_explain_key] = compute_tft_explainability(
                runtime,
                data.full,
                selected_sku,
            )
    tft_explanation, tft_explanation_warnings = st.session_state.tft_explanation_cache[tft_explain_key]
    for warning in tft_explanation_warnings[:2]:
        st.warning(warning)

    left, right = st.columns(2)
    with left:
        st.markdown("<h5 style='color:#0f172a; text-align:center;'>TFT Variable Selection Importance</h5>", unsafe_allow_html=True)
        tft_variables = tft_explanation["variables"]
        if tft_variables.empty:
            st.info("TFT variable selection weights are unavailable for this SKU.")
        else:
            fig_tft_vars = go.Figure(
                go.Bar(
                    x=tft_variables["importance"],
                    y=tft_variables["feature"],
                    orientation="h",
                    marker=dict(color=tft_variables["importance"], colorscale="Teal", line=dict(color="rgba(15, 118, 110, 0.5)", width=1)),
                    customdata=tft_variables["source"],
                    hovertemplate="%{y}<br>Importance: %{x:.1%}<br>Source: %{customdata}<extra></extra>",
                )
            )
            fig_tft_vars.update_xaxes(title="Relative Selection Weight", tickformat=".0%")
            st.plotly_chart(plot_theme(fig_tft_vars, height=300, top_margin=10), width="stretch")
    with right:
        st.markdown(f"<h5 style='color:#0f172a; text-align:center;'>Encoder Attention: {selected_sku}</h5>", unsafe_allow_html=True)
        tft_attention = tft_explanation["attention"]
        if tft_attention.empty:
            st.info("TFT attention weights are unavailable for this SKU.")
        else:
            attention_matrix = tft_attention.pivot(index="horizon", columns="encoder_step", values="attention").sort_index()
            fig_attention = go.Figure(
                go.Heatmap(
                    z=attention_matrix.to_numpy(),
                    x=attention_matrix.columns,
                    y=attention_matrix.index,
                    colorscale="Viridis",
                    colorbar=dict(title="Attention"),
                    hovertemplate="Horizon: %{y}<br>Lookback step: %{x}<br>Attention: %{z:.2%}<extra></extra>",
                )
            )
            fig_attention.update_xaxes(title="Encoder lookback step (0 = most recent)")
            fig_attention.update_yaxes(title="Forecast horizon")
            st.plotly_chart(plot_theme(fig_attention, height=300, top_margin=10), width="stretch")
else:
    st.markdown(f"<h5 style='color:#0f172a; text-align:center;'>Global Feature Importance ({importance_label})</h5>", unsafe_allow_html=True)
    fig_global = go.Figure(go.Bar(x=global_importance["importance"], y=global_importance["feature"], orientation="h", marker=dict(color=global_importance["importance"], colorscale="Blues", line=dict(color="rgba(14, 165, 233, 0.5)", width=1))))
    fig_global.update_xaxes(title="Relative Importance Weight", tickformat=".0%")
    st.plotly_chart(plot_theme(fig_global, height=360, top_margin=10), width="stretch")
render_section_end()

render_section_start("Dynamic Inventory Policy Simulator", "#9333ea", "Evaluate physical manufacturing runway cost structures against forecast demand and model uncertainty.")
col1, col2, col3 = st.columns(3)
lead_time = col1.slider("Procurement Lead Time (Days)", min_value=1, max_value=30, value=7)
holding_cost = col1.slider("Holding Cost ($/Unit/Month)", min_value=1.0, max_value=50.0, value=12.5)
service_level = col2.slider("Service Level Target (%)", min_value=80, max_value=99, value=95)
stockout_cost = col2.slider("Stockout Cost ($/Unit/Event)", min_value=20, max_value=500, value=150)
safety_stock_mult = col3.slider("Safety Stock Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
policy = compute_inventory_policy(
    forecast_frame,
    selected_sku,
    lead_time,
    service_level,
    safety_stock_mult,
    holding_cost,
    stockout_cost,
    historical=data.full,
)
policy_available = policy["uncertainty_source"] != "unavailable" and np.isfinite(policy["safety_stock"])
simulator_metrics = [
    ("Rec. Safety Stock", f"{policy['safety_stock']:.1f} u" if policy_available else "N/A", "#0f172a"),
    ("Reorder Point (ROP)", f"{policy['reorder_point']:.1f} u" if policy_available else "N/A", "#0f172a"),
    ("Holding Cost (Est.)", f"${policy['holding_cost']:.2f}" if policy_available else "N/A", "#0f172a"),
    ("Stockout Risk", f"{100 - service_level}%", "#dc2626"),
    ("Estimated Total Cost", f"${policy['total_cost']:.2f}" if policy_available else "N/A", "#15803d"),
]
for column, item in zip(st.columns(5), simulator_metrics):
    with column:
        render_metric(*item)
if policy["uncertainty_source"] == "historical_variability":
    st.caption("Safety stock uses the selected SKU's recent historical variability because a usable model forecast interval is unavailable.")
elif policy["uncertainty_source"] == "unavailable":
    st.warning("Safety stock is unavailable because this SKU has neither a usable forecast interval nor enough demand history.")
df_sim = policy["series"]
fig_sim = go.Figure()
if not df_sim.empty:
    fig_sim.add_trace(go.Scatter(x=df_sim["Day"], y=df_sim["Inventory Level"], mode="lines+markers", name="Physical Inventory Level", line=dict(color="#15803d", width=3)))
    fig_sim.add_trace(go.Scatter(x=df_sim["Day"], y=df_sim["Reorder Point"], mode="lines", name="Reorder Point Target", line=dict(color="#9333ea", width=1.5, dash="dash")))
    fig_sim.add_trace(go.Scatter(x=df_sim["Day"], y=df_sim["Safety Stock Threshold"], mode="lines", name="Safety Stock Floor", line=dict(color="#dc2626", width=1.5, dash="dot")))
fig_sim.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_sim.update_xaxes(title="Runway Optimization Horizon (Days)")
fig_sim.update_yaxes(title="Physical Units (in stock)")
st.plotly_chart(plot_theme(fig_sim, height=320), width="stretch")
left, right, _ = st.columns([1, 1, 4])
if left.button("Run Simulation", key="run_sim"):
    st.toast("Inventory simulation rerun from forecast demand.")
if right.button("Reset Parameters", key="reset_sim"):
    st.toast("Parameters reset to defaults. Adjust the sliders above to continue.")
render_section_end()

st.markdown(
    f"""
    <div class="summary-panel">
        <h4 style="margin:0 0 10px 0; color:#67e8f9;">Executive Supply Chain Summary</h4>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:20px;">
            <div>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-bottom:5px;">Forecasted Horizon Demand Volume</p>
                <p style="font-size:1.4rem; font-weight:700; margin:0; color:#ffffff;">{summary['forecast_total']:,.0f} Units <span style="font-size:0.9rem; color:#86efac;">({summary['growth_pct']:+.1f}%)</span></p>
            </div>
            <div>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-bottom:5px;">Materials At Risk</p>
                <p style="font-size:1.4rem; font-weight:700; margin:0; color:#fca5a5;">{summary['risk_count']} SKUs <span style="font-size:0.9rem; color:#fdba74;">(Forecast strain)</span></p>
            </div>
            <div>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-bottom:5px;">Overstocked Value Allocation</p>
                <p style="font-size:1.4rem; font-weight:700; margin:0; color:#fdba74;">{summary['overstock_count']} SKUs <span style="font-size:0.9rem; color:#cbd5e1;">(Below baseline)</span></p>
            </div>
            <div>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-bottom:5px;">Recommended Sourcing Action</p>
                <p style="font-size:1.0rem; font-weight:600; margin:0; color:#86efac;">{summary['action']}</p>
            </div>
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.10); padding-top:15px; margin-top:15px;">
            <p style="margin:0; font-size:0.9rem; color:#cbd5e1; font-style:italic;">{summary['narrative']}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
