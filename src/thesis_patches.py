"""
THESIS PATCHES — 4 code blocks to run in your existing Colab notebooks.
Each block is self-contained; paste it into the indicated notebook/cell location.
Run them, then paste the PRINTED OUTPUT back to me — I will write the thesis
text/tables from the real numbers you get (no fabricated numbers).

============================================================================
PATCH 1 — CLUSTER ABLATION (with vs without Demand_Cluster_TE)
Where  : train_xg.ipynb, paste as a NEW CELL right after Cell 8
         (i.e. after clf_tuned / reg_tuned / bp / best_threshold_optuna /
         metrics_hybrid_tuned are all defined — the Hybrid Tuned model).
Purpose: This is the controlled ablation the thesis (Section IV.2.4,
         Limitations) explicitly says is missing. It isolates the causal
         contribution of the cluster feature, holding everything else
         (hyperparameters, threshold, data) identical.
============================================================================
"""

# ===================== [NEW] Cluster Ablation: With vs Without Demand_Cluster_TE =====================
print("\n" + "=" * 65)
print("[ABLATION] Hybrid Tuned — With vs Without Demand_Cluster_TE")
print("=" * 65)

MODEL_FEATURES_NO_CLUSTER = [c for c in MODEL_FEATURES if c != "Demand_Cluster_TE"]
print(f"  Full features     : {len(MODEL_FEATURES)}")
print(f"  Without cluster   : {len(MODEL_FEATURES_NO_CLUSTER)}")

X_train_noclu = X_train[MODEL_FEATURES_NO_CLUSTER]
X_val_noclu   = X_val[MODEL_FEATURES_NO_CLUSTER]
X_test_noclu  = X_test[MODEL_FEATURES_NO_CLUSTER]

# Same tuned hyperparameters (bp) as the headline Hybrid model — only the
# feature set changes, so any WAPE difference is attributable to the cluster
# feature alone, not to a different model configuration.
clf_noclu = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=bp["clf_lr"], max_depth=bp["clf_max_depth"], num_leaves=bp["clf_num_leaves"],
    min_child_samples=bp["clf_min_child"], subsample=bp["clf_subsample"], colsample_bytree=bp["clf_colsample"],
    reg_alpha=bp["clf_reg_alpha"], reg_lambda=bp["clf_reg_lambda"],
    is_unbalance=True, random_state=42, n_jobs=-1, verbosity=-1,
)
clf_noclu.fit(
    X_train_noclu, (y_train > 0).astype(int),
    eval_set=[(X_val_noclu, (y_val > 0).astype(int))],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
cal_noclu = CalibratedClassifierCV(clf_noclu, cv="prefit", method="isotonic")
cal_noclu.fit(X_val_noclu, (y_val > 0).astype(int))

reg_noclu = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=bp["reg_lr"], max_depth=bp["reg_max_depth"], num_leaves=bp["reg_num_leaves"],
    min_child_samples=bp["reg_min_child"], subsample=bp["reg_subsample"], colsample_bytree=bp["reg_colsample"],
    reg_alpha=bp["reg_reg_alpha"], reg_lambda=bp["reg_reg_lambda"],
    random_state=42, n_jobs=-1, verbosity=-1,
)
reg_noclu.fit(
    X_train_noclu[nonzero_tr], log_transform(y_train[nonzero_tr].values),
    eval_set=[(X_val_noclu[nonzero_val], log_transform(y_val[nonzero_val].values))],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

probs_noclu_test = cal_noclu.predict_proba(X_test_noclu)[:, 1]
demand_noclu_test = inverse_log_transform(reg_noclu.predict(X_test_noclu))
# Use the SAME decision threshold as the headline model for a fair, isolated comparison
preds_noclu = np.where(probs_noclu_test < best_threshold_optuna, 0.0, demand_noclu_test)
metrics_noclu = compute_metrics(y_test, preds_noclu, label="Hybrid (No Cluster)")

print(f"\n  {'Model':<35s} {'#Features':>10} {'RMSE':>10} {'MAE':>10} {'WAPE':>10}")
print(f"  {'-' * 76}")
print(f"  {'Hybrid Tuned (With Cluster)':<35s} {len(MODEL_FEATURES):>10} "
      f"{metrics_hybrid_tuned['rmse']:>10.2f} {metrics_hybrid_tuned['mae']:>10.2f} {metrics_hybrid_tuned['wape']:>9.2f}%")
print(f"  {'Hybrid Tuned (Without Cluster)':<35s} {len(MODEL_FEATURES_NO_CLUSTER):>10} "
      f"{metrics_noclu['rmse']:>10.2f} {metrics_noclu['mae']:>10.2f} {metrics_noclu['wape']:>9.2f}%")

delta_wape_cluster = metrics_noclu["wape"] - metrics_hybrid_tuned["wape"]
print(f"\n  Delta WAPE (No-Cluster - With-Cluster): {delta_wape_cluster:+.2f}pp")
if delta_wape_cluster > 0:
    print("  -> Removing the cluster feature HURTS WAPE: cluster carries causal predictive signal.")
elif delta_wape_cluster < 0:
    print("  -> Removing the cluster feature IMPROVES WAPE: cluster may be redundant given other features.")
else:
    print("  -> No measurable difference.")

with open("cluster_ablation.json", "w") as f:
    json.dump({
        "wape_with_cluster": metrics_hybrid_tuned["wape"],
        "wape_without_cluster": metrics_noclu["wape"],
        "rmse_with_cluster": metrics_hybrid_tuned["rmse"],
        "rmse_without_cluster": metrics_noclu["rmse"],
        "mae_with_cluster": metrics_hybrid_tuned["mae"],
        "mae_without_cluster": metrics_noclu["mae"],
        "delta_wape_pp": delta_wape_cluster,
    }, f, indent=2)
print("\n  Saved: cluster_ablation.json")


"""
============================================================================
PATCH 2 — EXPORT TUNED HYBRID FOR THE SHAP NOTEBOOK
Where  : train_xg.ipynb, paste right after Patch 1 (or right after Model 4 /
         Hybrid Tuned is trained — needs clf_tuned, cal_tuned, reg_tuned,
         best_threshold_optuna, MODEL_FEATURES to exist).
Purpose: shap_hybrid.ipynb currently trains its OWN Hybrid model with DEFAULT
         hyperparameters (see its Cell 7 warning: "BEST_PARAMS_HYBRID=None ->
         dùng config mặc định, KHÔNG phải bản Optuna đã tune"), giving Test
         WAPE=76.74% — NOT the 72.69% headline model from Table 6. This patch
         exports the EXACT tuned model so SHAP is computed on the real
         headline model, not a different one.
============================================================================
"""

import pickle
with open("hybrid_model.pkl", "wb") as f:
    pickle.dump({
        "clf_tuned": clf_tuned,
        "cal_tuned": cal_tuned,
        "reg_tuned": reg_tuned,
        "threshold": best_threshold_optuna,
        "model_features": MODEL_FEATURES,
    }, f)
print("Saved: hybrid_model.pkl")
print("Next: upload hybrid_model.pkl into the shap_hybrid.ipynb runtime, then in")
print("its Config cell set:  PATHS['hybrid_pickle'] = 'hybrid_model.pkl'")
print("...then Run All in shap_hybrid.ipynb. Its Cell 7 already has the loading")
print("branch wired up (if PATHS['hybrid_pickle'] exists: load instead of retrain).")


"""
============================================================================
PATCH 3 — FIX fill_missing_weeks() TO PRESERVE Demand_Cluster
Where  : TFT_Benchmark_Colab_v3_feature_selection.ipynb, REPLACE the existing
         fill_missing_weeks() function inside Cell 4.
Purpose: Patch 4 (below) adds 'Demand_Cluster' as a static categorical. But
         the CURRENT fill_missing_weeks() does:
             num_cols = grp.select_dtypes(include='number').columns
             grp[num_cols] = grp[num_cols].fillna(0)
         Demand_Cluster is numeric (0/1), so newly-inserted missing-week rows
         would get their cluster id silently zeroed out — corrupting the
         label for every Cluster-1 series. This patch carries the cluster id
         forward as a per-series constant before the generic numeric fillna.
============================================================================
"""

def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    filled = []
    for (prod, wh), grp in df.groupby(['Product_Code', 'Warehouse']):
        # [FIX] remember the (constant) cluster id for this series before reindexing
        cluster_val = grp['Demand_Cluster'].iloc[0] if 'Demand_Cluster' in grp.columns else None

        full_range = pd.date_range(
            start=grp['Date'].min(),
            end=grp['Date'].max(),
            freq='7D'
        )
        grp = (grp.set_index('Date')
                  .reindex(full_range)
                  .reset_index()
                  .rename(columns={'index': 'Date'}))
        grp['Product_Code'] = prod
        grp['Warehouse']    = wh
        if cluster_val is not None:
            # [FIX] re-assert the cluster id on every row (including filled ones)
            # BEFORE the generic numeric fillna(0) below would zero it out
            grp['Demand_Cluster'] = cluster_val
        grp['Order_Demand'] = grp['Order_Demand'].fillna(0)
        num_cols = grp.select_dtypes(include='number').columns
        grp[num_cols] = grp[num_cols].fillna(0)
        filled.append(grp)
    return pd.concat(filled, ignore_index=True)


"""
============================================================================
PATCH 4 — FIX TFT CONFIG (real column names + add Demand_Cluster)
Where  : TFT_Benchmark_Colab_v3_feature_selection.ipynb, REPLACE Cell 3
         entirely with the block below.
Why    : The original CFG listed feature names ('WeekOfYear', 'Demand_Lag_7',
         'Zero_Rate_7d', 'Day', 'DayOfWeek', ...) that DO NOT EXIST in
         train_preprocessed.parquet (verified against the printed column list
         in Cell 2's output). pytorch-forecasting silently drops any
         configured column not found in the dataframe, so the TFT ended up
         training on just 2 real-valued inputs (Year, Month) plus the two
         raw categorical IDs. The names below are copied verbatim from the
         actual Cell 2 column dump, so the post-config "[WARN] dropped {...}"
         line should now print nothing (empty set) — that's your sanity check
         that the fix worked.
         Also: 'Demand_Cluster' is added as a static categorical — previously
         the TFT benchmark never saw the cluster label AT ALL, even before
         this bug. This is the same feature LightGBM ranks in its top 10
         (Table 11) — TFT should at least have the chance to use it too.
         Also: max_encoder_length changed 104 -> 52 weeks to match Table 8 in
         the thesis ("one-year encoder window"). If you'd rather keep 104,
         just change it back here AND update Table 8 in the thesis instead —
         your call, just keep the two consistent.
============================================================================
"""

import random, warnings
import torch
import numpy as np
import lightning as pl
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=True)

CFG = {
    'target'   : 'Order_Demand',
    'time_col' : 'Date',
    'group_col': 'Product_Code',
    'group_ids' : ['Product_Code', 'Warehouse'],

    'forecast_horizon'  : 1,
    'max_encoder_length': 52,    # [FIX] was 104 -> 52 to match thesis Table 8
    'min_encoder_length': 4,

    'batch_size' : 128,
    'num_workers': 4,

    # [FIX] Demand_Cluster added (was completely absent before)
    'static_categoricals': ['Product_Code', 'Warehouse', 'Demand_Cluster'],
    'static_reals'       : [],

    # [FIX] names now match the ACTUAL columns in train_preprocessed.parquet
    'time_varying_known_reals': [
        # Calendar / seasonality
        'Week_Of_Year', 'Month', 'Quarter', 'Year',
        'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos', 'Is_Quarter_End',
        # External economic & market indicators (already Lag-1 shifted ->
        # known at the time the model forecasts the next week)
        'Copper_Price_Lag1', 'Aluminum_Price_Lag1', 'Oil_Price_Lag1',
        'USD_Index_Lag1', 'SP500_Lag1', 'Industrial_ETF_Lag1',
        'Industrial_Production_Lag1', 'Capacity_Utilization_Lag1',
        'Durable_Goods_Orders_Lag1', 'PPI_All_Commodities_Lag1',
        'Crude_Oil_WTI_Lag1', 'Consumer_Sentiment_Lag1',
        'Fed_Funds_Rate_Lag1', 'Treasury_10Y_Lag1',
        'Copper_Aluminum_Ratio_Lag1', 'Copper_Price_Momentum4W_Lag1',
        'Aluminum_Price_Momentum4W_Lag1', 'Oil_Price_Momentum4W_Lag1',
        'Yield_Spread_Lag1',
    ],
    'time_varying_known_categoricals': [],

    # [FIX] demand-history-derived features that actually exist in the parquet
    'time_varying_unknown_reals': [
        'Demand_Lag1W', 'Demand_Lag2W', 'Demand_Lag4W',
        'Demand_Lag8W', 'Demand_Lag12W', 'Demand_Lag52W',
        'Demand_RollMean4W', 'Demand_RollStd4W',
        'Demand_RollMean8W', 'Demand_RollStd8W',
        'Demand_RollMean12W', 'Demand_RollStd12W',
        'Demand_Trend', 'Demand_YoY_Growth', 'Demand_Expanding_Mean',
        'Is_Zero_Demand', 'Weeks_Since_Last_Order',
    ],
}

for key in ['time_varying_known_reals', 'time_varying_unknown_reals',
            'static_categoricals', 'time_varying_known_categoricals']:
    before = CFG[key]
    CFG[key] = [c for c in before if c in train_df.columns]
    dropped = set(before) - set(CFG[key])
    if dropped:
        print(f'[WARN] {key}: dropped {dropped}   <-- should be EMPTY now; if not, columns truly are missing from your parquet')

print('Config OK')
print(f'  known_reals   ({len(CFG["time_varying_known_reals"])}): {CFG["time_varying_known_reals"]}')
print(f'  unknown_reals ({len(CFG["time_varying_unknown_reals"])}): {CFG["time_varying_unknown_reals"]}')
print(f'  static_categoricals: {CFG["static_categoricals"]}')

"""
After pasting Patch 3 + Patch 4, just re-run the rest of the TFT notebook
top to bottom as-is. The existing structure already does everything you
asked for:
  - Cell 6 (Part A correlation filter) will now run on ~36 real candidate
    features instead of 2, and will actually drop correlated pairs.
  - Cells 12-15 train + evaluate "TFT (Full features, after corr filter)".
  - Cell 18 (Part B importance threshold) will now retrain a SECOND TFT on
    a reduced feature set and print a real "Full vs Selected" WMAPE table —
    exactly the LightGBM-style comparison you asked for, just for TFT.

Send me back the printed output of Cells 13-15 and Cell 18 (and the cluster
ablation + matched-hyperparameter SHAP output from Patches 1-2) and I will
write the corresponding thesis paragraphs/tables from your real numbers.
"""
