import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from datetime import timedelta

df_demand = pd.read_csv("Historical Product Demand.csv")
df_demand["Date"] = pd.to_datetime(df_demand["Date"])

print(f"\n Original demand data shape: {df_demand.shape}")
print(f"Original columns: {df_demand.columns.tolist()}")
print(f"Unique Product_Code values: {df_demand['Product_Code'].nunique()}")
print(f"Unique Warehouse values: {df_demand['Warehouse'].nunique()}")

df_weekly = df_demand.groupby(
    ["Product_Code", "Warehouse", pd.Grouper(key="Date", freq="W-MON")]
)["Order_Demand"].sum().reset_index()

print(f"\n After grouping - df_weekly shape: {df_weekly.shape}")
print(f"df_weekly columns: {df_weekly.columns.tolist()}")
print(f"df_weekly head:\n{df_weekly.head()}")

df_weekly["Order_Demand"] = (
    pd.to_numeric(df_weekly["Order_Demand"], errors="coerce")
    .fillna(0)
    .clip(lower=0)
)

start_date_demand = df_weekly["Date"].min()
end_date_demand   = df_weekly["Date"].max()
start_date_external = start_date_demand - timedelta(days=90)   # wider window for rolling

print(f"Demand range: {start_date_demand.date()} to {end_date_demand.date()}")

fred = Fred(api_key="")

def get_fred_safe(series_id, name):
    try:
        print(f"  Fetching {name} ({series_id})")
        s = fred.get_series(series_id, observation_start=start_date_external)
        df = s.reset_index()
        df.columns = ["Date", name]
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        print(f" {series_id}: {e}")
        return pd.DataFrame(columns=["Date", name])

# Supply-side  
df_ipman   = get_fred_safe("IPMAN",   "Industrial_Production")
df_tcu     = get_fred_safe("TCU",     "Capacity_Utilization")
df_dgorder = get_fred_safe("DGORDER", "Durable_Goods_Orders")

# Cost  
df_ppi     = get_fred_safe("PPIACO",     "PPI_All_Commodities")
df_oil_fred= get_fred_safe("DCOILWTICO", "Crude_Oil_WTI")

# Demand-side  
df_sentiment = get_fred_safe("UMCSENT",  "Consumer_Sentiment")
df_fedfunds  = get_fred_safe("FEDFUNDS", "Fed_Funds_Rate")
df_dgs10     = get_fred_safe("DGS10",    "Treasury_10Y")

fred_frames = [
    df_ipman, df_tcu, df_dgorder,
    df_ppi, df_oil_fred,
    df_sentiment, df_fedfunds, df_dgs10,
]

# FETCH MARKET/COMMODITY DATA (yfinance)

def get_daily_data(symbol, name, start, end):
    print(f"  Downloading {name} ({symbol})...")
    data = yf.download(symbol, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame(columns=["Date", name])
    close = data["Close"] if not isinstance(data.columns, pd.MultiIndex) \
            else data[("Close", symbol)]
    df = close.reset_index()
    df.columns = ["Date", name]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

print("\n[yfinance] Fetching commodity & market prices...")

df_copper   = get_daily_data("HG=F",     "Copper_Price",   start_date_external, end_date_demand)
df_aluminum = get_daily_data("ALI=F",    "Aluminum_Price", start_date_external, end_date_demand)
df_oil_yf   = get_daily_data("CL=F",     "Oil_Price",      start_date_external, end_date_demand)
df_usd      = get_daily_data("DX-Y.NYB", "USD_Index",      start_date_external, end_date_demand)
df_sp500    = get_daily_data("^GSPC",    "SP500",          start_date_external, end_date_demand)
df_xli      = get_daily_data("XLI",      "Industrial_ETF", start_date_external, end_date_demand)

yf_frames = [df_copper, df_aluminum, df_oil_yf, df_usd, df_sp500, df_xli]

# ALIGNMENT, WEEKLY AGGREGATION & LAGGING
all_dates   = pd.date_range(start=start_date_external, end=end_date_demand, freq="D")
df_external = pd.DataFrame({"Date": all_dates})

for df in fred_frames + yf_frames:
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df_external = df_external.merge(df, on="Date", how="left")

df_external = df_external.sort_values("Date").ffill().bfill()

# Resample to weekly (Monday anchor)
df_ext_w = df_external.resample("W-MON", on="Date").mean().reset_index()

# Derived/interaction features 
df_ext_w["Copper_Aluminum_Ratio"] = (
    df_ext_w["Copper_Price"] / df_ext_w["Aluminum_Price"].replace(0, np.nan)
)

for col in ["Copper_Price", "Aluminum_Price", "Oil_Price"]:
    if col in df_ext_w.columns:
        df_ext_w[f"{col}_Momentum4W"] = (
            df_ext_w[col] - df_ext_w[col].rolling(4, min_periods=1).mean()
        )

df_ext_w["Yield_Spread"] = df_ext_w["Treasury_10Y"] - df_ext_w["Fed_Funds_Rate"]

raw_feature_cols = [
    "Copper_Price", "Aluminum_Price", "Oil_Price",
    "USD_Index", "SP500", "Industrial_ETF",
    "Industrial_Production", "Capacity_Utilization", "Durable_Goods_Orders",
    "PPI_All_Commodities", "Crude_Oil_WTI",
    "Consumer_Sentiment", "Fed_Funds_Rate", "Treasury_10Y",
    "Copper_Aluminum_Ratio",
    "Copper_Price_Momentum4W", "Aluminum_Price_Momentum4W", "Oil_Price_Momentum4W",
    "Yield_Spread",
]

lag_cols = []
for col in raw_feature_cols:
    if col in df_ext_w.columns:
        lag_name = f"{col}_Lag1"
        df_ext_w[lag_name] = df_ext_w[col].shift(1)
        lag_cols.append(lag_name)

df_features = df_ext_w[["Date"] + lag_cols]

print(f"\ndf_features shape: {df_features.shape}")
print(f"df_features date range: {df_features['Date'].min()} to {df_features['Date'].max()}")


print(f"Before merge - df_weekly shape: {df_weekly.shape}")
print(f"df_weekly date range: {df_weekly['Date'].min()} to {df_weekly['Date'].max()}")

df_final = df_weekly.merge(df_features, on="Date", how="left")

print(f"After merge - df_final shape: {df_final.shape}")
print(f"Columns after merge: {df_final.columns.tolist()[:10]} ")

if df_final['Product_Code'].isnull().any():
    print(f"Null Product_Code found.")
    
if df_final.empty:
    raise ValueError("df_final is empty. Check if df_weekly or df_features have data.")

df_final = df_final.sort_values(["Product_Code", "Warehouse", "Date"])

df_final[lag_cols] = df_final.groupby("Product_Code")[lag_cols].ffill().bfill()

# CALENDAR FEATURES
df_final["Week_Of_Year"]    = df_final["Date"].dt.isocalendar().week.astype(int)
df_final["Month"]           = df_final["Date"].dt.month
df_final["Quarter"]         = df_final["Date"].dt.quarter
df_final["Year"]            = df_final["Date"].dt.year
df_final["Month_Sin"]       = np.sin(2 * np.pi * df_final["Month"] / 12)
df_final["Month_Cos"]       = np.cos(2 * np.pi * df_final["Month"] / 12)
df_final["Week_Sin"]        = np.sin(2 * np.pi * df_final["Week_Of_Year"] / 52)
df_final["Week_Cos"]        = np.cos(2 * np.pi * df_final["Week_Of_Year"] / 52)
df_final["Is_Quarter_End"]  = (
    df_final["Date"].dt.is_quarter_end |
    (df_final["Date"] + pd.Timedelta(weeks=1)).dt.is_quarter_end
).astype(int)

# DEMAND LAG & ROLLING FEATURES  
def add_demand_features(g):
    # Sort by date
    g = g.sort_values("Date").copy()
    d = g["Order_Demand"]

    result = pd.DataFrame()
    result["Product_Code"] = g["Product_Code"]
    result["Warehouse"] = g["Warehouse"]
    result["Date"] = g["Date"]
    result["Order_Demand"] = g["Order_Demand"]
    
    external_cols = [col for col in g.columns if col not in ['Product_Code', 'Warehouse', 'Date', 'Order_Demand']]
    for col in external_cols:
        result[col] = g[col].values

    # Point lags
    for w in [1, 2, 4, 8, 12, 52]:
        result[f"Demand_Lag{w}W"] = d.shift(w).values

    # Rolling statistics  
    d_lag1 = d.shift(1)
    for w in [4, 8, 12]:
        result[f"Demand_RollMean{w}W"] = d_lag1.rolling(w, min_periods=1).mean().values
        result[f"Demand_RollStd{w}W"]  = d_lag1.rolling(w, min_periods=2).std().values

    # Trend: 4-week mean vs 12-week mean (positive = accelerating demand)
    result["Demand_Trend"] = result["Demand_RollMean4W"] - result["Demand_RollMean12W"]

    # Year-over-year: compare to same week 52 weeks ago
    result["Demand_YoY_Growth"] = (d.shift(1) - d.shift(53)) / (d.shift(53).abs() + 1)
    result["Demand_YoY_Growth"] = result["Demand_YoY_Growth"].values

    # Long-run baseline (expanding mean up to last week)
    result["Demand_Expanding_Mean"] = d_lag1.expanding(min_periods=4).mean().values

    return result

# Apply the function to each group and concatenate
df_final = pd.concat([
    add_demand_features(group) 
    for name, group in df_final.groupby(["Product_Code", "Warehouse"])
], ignore_index=True)

print(f"After demand features - df_final shape: {df_final.shape}")
print(f"First 5 columns: {df_final.columns[:10].tolist()}")
print(f"Sample Product_Code values: {df_final['Product_Code'].head(3).tolist()}")
print(f"Sample Warehouse values: {df_final['Warehouse'].head(3).tolist()}")

# INTERMITTENT DEMAND FEATURES

def add_intermittent_features(g):
    g = g.sort_values("Date").copy()
    d = g["Order_Demand"]

    g["Is_Zero_Demand"] = (d == 0).astype(int)

    cnt, weeks = 0, []
    for val in (d.shift(1) > 0):
        cnt = 1 if val else cnt + 1
        weeks.append(cnt)
    g["Weeks_Since_Last_Order"] = weeks

    return g

df_final = pd.concat([
    add_intermittent_features(grp)
    for _, grp in df_final.groupby(["Product_Code", "Warehouse"])
], ignore_index=True)

print(f"  Zero demand rate       : {df_final['Is_Zero_Demand'].mean():.1%}")
print(f"  Weeks_Since_Last_Order : median = {df_final['Weeks_Since_Last_Order'].median():.0f}")

feature_groups = {
    "External macro (lagged)": lag_cols,
    "Calendar": [
        "Week_Of_Year", "Month", "Quarter", "Year",
        "Month_Sin", "Month_Cos", "Week_Sin", "Week_Cos", "Is_Quarter_End",
    ],
    "Demand lags": [f"Demand_Lag{w}W" for w in [1, 2, 4, 8, 12, 52]],
    "Demand rolling stats": [
        f"Demand_{stat}{w}W"
        for stat in ["RollMean", "RollStd"] for w in [4, 8, 12]
    ],
    "Demand trend & growth": ["Demand_Trend", "Demand_YoY_Growth", "Demand_Expanding_Mean"], 
    "Intermittent demand": ["Is_Zero_Demand", "Weeks_Since_Last_Order"],
}

all_feature_cols = [c for cols in feature_groups.values() for c in cols]

for group, cols in feature_groups.items():
    available = [c for c in cols if c in df_final.columns]
    print(f"  {group:<35} {len(available):>3} features")
print(f"  {'TOTAL':<35} {len(all_feature_cols):>3} features")

print("\nMissing values (should be near 0 after ffill/bfill):")
missing = df_final[all_feature_cols].isna().sum()
print(missing[missing > 0].to_string() if missing.any() else "None ")

print(f"Shape: {df_final.shape}")
print(f"Columns: {df_final.columns.tolist()[:15]}(showing first 15)")
print(f"\nFirst 5 rows of key columns:")
key_cols = ["Product_Code", "Warehouse", "Date", "Order_Demand"]
print(df_final[key_cols].head())

print(f"\nFirst 5 rows with features:")
preview_cols = key_cols + all_feature_cols[:6]
print(df_final[preview_cols].head())

print(df_final.columns.tolist())

output_path = "df_final_enriched.parquet"
print(f"\nSave to {output_path}")
df_final.to_parquet(output_path, index=False)

df = pd.read_parquet(output_path)
