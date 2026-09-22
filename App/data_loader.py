from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = {
    "full": PROJECT_ROOT / "df_final_enriched.parquet",
    "train": PROJECT_ROOT / "train_preprocessed.parquet",
    "val": PROJECT_ROOT / "val_preprocessed.parquet",
    "test": PROJECT_ROOT / "test_preprocessed.parquet",
    "test_predictions": PROJECT_ROOT / "test_predictions_lgbm.parquet",
    "pipeline_meta": PROJECT_ROOT / "pipeline_meta.json",
    "selected_features": PROJECT_ROOT / "selected_features.json",
    "cluster_ablation": PROJECT_ROOT / "cluster_ablation.json",
}


@dataclass
class DemandData:
    full: pd.DataFrame = field(default_factory=pd.DataFrame)
    train: pd.DataFrame = field(default_factory=pd.DataFrame)
    val: pd.DataFrame = field(default_factory=pd.DataFrame)
    test: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class UploadedDataValidation:
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and not self.data.empty

    @property
    def row_count(self) -> int:
        return len(self.data)

    @property
    def sku_count(self) -> int:
        return int(self.data["SKU"].nunique()) if "SKU" in self.data else 0


COLUMN_ALIASES = {
    "Date": ["Date", "ds"],
    "Order_Demand": ["Order_Demand", "Demand", "Quantity", "Sales", "y"],
    "Product_Code": ["Product_Code", "SKU", "product", "material"],
    "Warehouse": ["Warehouse", "location"],
}
MINIMUM_HISTORY_ROWS = 4


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_parquet(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Missing data file: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.sort_values([c for c in ["Product_Code", "Warehouse", "Date"] if c in df.columns]).reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not load {path.name}: {exc}")
        return pd.DataFrame()


def load_demand_data() -> DemandData:
    warnings: list[str] = []
    train = _read_parquet(DATA_PATHS["train"], warnings)
    val = _read_parquet(DATA_PATHS["val"], warnings)
    test = _read_parquet(DATA_PATHS["test"], warnings)
    full = _read_parquet(DATA_PATHS["full"], warnings)

    if full.empty:
        parts = [df for df in [train, val, test] if not df.empty]
        full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    metadata = {
        "pipeline": _read_json(DATA_PATHS["pipeline_meta"]),
        "selected_features": _read_json(DATA_PATHS["selected_features"]),
        "cluster_ablation": _read_json(DATA_PATHS["cluster_ablation"]),
    }
    return DemandData(full=full, train=train, val=val, test=test, warnings=warnings, metadata=metadata)


def _find_alias_column(columns: pd.Index, aliases: list[str]) -> str | None:
    available = {str(column).strip().casefold(): str(column) for column in columns}
    for alias in aliases:
        match = available.get(alias.casefold())
        if match is not None:
            return match
    return None


def validate_uploaded_csv(file_bytes: bytes) -> UploadedDataValidation:
    if not file_bytes or not file_bytes.strip():
        return UploadedDataValidation(errors=["The uploaded CSV is empty."])
    try:
        raw = pd.read_csv(BytesIO(file_bytes), low_memory=False)
    except pd.errors.EmptyDataError:
        return UploadedDataValidation(errors=["The uploaded CSV is empty."])
    except Exception as exc:
        return UploadedDataValidation(errors=[f"The CSV could not be read: {exc}"])
    if raw.empty:
        return UploadedDataValidation(errors=["The uploaded CSV contains no data rows."])

    matched = {
        canonical: _find_alias_column(raw.columns, aliases)
        for canonical, aliases in COLUMN_ALIASES.items()
    }
    missing = [
        label
        for canonical, label in [
            ("Date", "date"),
            ("Order_Demand", "demand"),
            ("Product_Code", "SKU/product"),
        ]
        if matched[canonical] is None
    ]
    if missing:
        return UploadedDataValidation(
            errors=[f"Missing required column(s): {', '.join(missing)}."]
        )

    normalized = raw.copy()
    selected_columns = {source: canonical for canonical, source in matched.items() if source is not None}
    normalized = normalized.rename(columns=selected_columns)
    if "Warehouse" not in normalized:
        normalized["Warehouse"] = "Uploaded"

    errors: list[str] = []
    raw_dates = normalized["Date"]
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    invalid_dates = raw_dates.isna() | raw_dates.astype(str).str.strip().eq("") | parsed_dates.isna()
    if invalid_dates.any():
        errors.append(f"Date contains {int(invalid_dates.sum())} missing or invalid value(s).")

    raw_demand = normalized["Order_Demand"]
    parsed_demand = pd.to_numeric(raw_demand, errors="coerce")
    invalid_demand = raw_demand.isna() | ~np.isfinite(parsed_demand)
    if invalid_demand.any():
        errors.append(f"Demand contains {int(invalid_demand.sum())} missing or non-numeric value(s).")

    product = normalized["Product_Code"].astype("string")
    invalid_product = product.isna() | product.str.strip().eq("")
    if invalid_product.any():
        errors.append(f"SKU/product contains {int(invalid_product.sum())} empty value(s).")

    warehouse = normalized["Warehouse"].astype("string")
    normalized["Warehouse"] = warehouse.mask(warehouse.isna() | warehouse.str.strip().eq(""), "Uploaded")
    if len(normalized) < MINIMUM_HISTORY_ROWS:
        errors.append(
            f"At least {MINIMUM_HISTORY_ROWS} historical rows are required; the file contains {len(normalized)}."
        )
    if errors:
        return UploadedDataValidation(errors=errors)

    normalized["Date"] = parsed_dates
    normalized["Order_Demand"] = parsed_demand.astype(float)
    normalized["Product_Code"] = product.astype(str)
    normalized["Warehouse"] = normalized["Warehouse"].astype(str)
    normalized = normalized.drop(columns=["SKU"], errors="ignore")
    normalized = add_sku_column(normalized)
    normalized = normalized.sort_values(["Product_Code", "Warehouse", "Date"]).reset_index(drop=True)
    short_skus = normalized.groupby("SKU").size().lt(MINIMUM_HISTORY_ROWS)
    warnings = []
    if short_skus.any():
        warnings.append(
            f"{int(short_skus.sum())} SKU(s) have fewer than {MINIMUM_HISTORY_ROWS} historical rows; their forecasts may be less reliable."
        )
    return UploadedDataValidation(data=normalized, warnings=warnings)


def sku_label(row: pd.Series) -> str:
    product = str(row.get("Product_Code", "UNKNOWN"))
    warehouse = str(row.get("Warehouse", ""))
    return f"{product} / {warehouse}" if warehouse else product


def add_sku_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "SKU" not in out.columns:
        if {"Product_Code", "Warehouse"}.issubset(out.columns):
            out["SKU"] = out["Product_Code"].astype(str) + " / " + out["Warehouse"].astype(str)
        elif "Product_Code" in out.columns:
            out["SKU"] = out["Product_Code"].astype(str)
        else:
            out["SKU"] = "UNKNOWN"
    return out
