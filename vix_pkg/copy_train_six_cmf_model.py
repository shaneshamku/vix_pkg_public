
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import pickle
import os

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

from functools import reduce


def walk_forward_train_cmf(
    df,
    target_col,
    feature_cols,
    cmf_id=1,
    date_col="Date",
    train_start_date=pd.Timestamp("2005-01-01"),
    train_length_months=54, # 4.5 years
    val_length_months=6,
    test_length_months=6
):
    """
    Performs a rolling (unanchored) walk-forward with fixed-length windows:
      - Train  = ~4.5 years
      - Val    = 6 months
      - Test   = 6 months
    Then shift all windows forward by 6 months each fold.

    E.g.:
      Fold0: Train=2005-01-01..2009-06-30,  Val=2009-07-01..2009-12-31,  Test=2010-01-01..2010-06-30
      Fold1: Train=2005-07-01..2009-12-31,  Val=2010-01-01..2010-06-30,  Test=2010-07-01..2010-12-31
      ... until the test window extends beyond data end.

    Returns:
      (overall_rmse, overall_ic, all_oos_df)
        - overall_rmse, overall_ic: aggregated metrics over all test folds
        - all_oos_df: DataFrame containing columns:
             [Date, Act_CMF{cmf_id}, Pred_CMF{cmf_id}, FoldID]
          for each day in all out-of-sample test sets.
    """

    # Ensure data is sorted by date and indexed properly.
    df_sorted = df.sort_values(date_col).copy()
    df_sorted.dropna(subset=[target_col], inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)

    # Convert date col to datetime
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

    final_date = df_sorted[date_col].max()
    
    # Initialize a single model with warm_start
    model = ElasticNet(alpha = 0.001 if cmf_id <= 4 else 0.0003,
                       l1_ratio=0.5,      # Lasso-like
                       warm_start=True,
                       max_iter=1000,
                       random_state=42)
    
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Store out-of-sample predictions for each fold
    all_oos = []

    # Initialize the rolling boundaries
    fold_id = 0
    current_train_start = train_start_date



    while True:
        # 1) Train end = train start + train_length_years
        train_end_date = current_train_start + pd.DateOffset(months=train_length_months)

        # 2) Validation start = next day after train_end, ends after val_length_months
        val_start_date = train_end_date + pd.Timedelta(days=1)
        val_end_date = val_start_date + pd.DateOffset(months=val_length_months) - pd.Timedelta(days=1)

        # 3) Test start = next day after val_end, ends after test_length_months
        test_start_date = val_end_date + pd.Timedelta(days=1)
        test_end_date = test_start_date + pd.DateOffset(months=test_length_months) - pd.Timedelta(days=1)

        if test_start_date > final_date:
            # Stop if the test would start beyond the dataset
            break

        # Slice the data
        train_mask = (df_sorted[date_col] >= current_train_start) & (df_sorted[date_col] <= train_end_date)
        val_mask   = (df_sorted[date_col] >= val_start_date) & (df_sorted[date_col] <= val_end_date)
        test_mask  = (df_sorted[date_col] >= test_start_date) & (df_sorted[date_col] <= test_end_date)

        df_train = df_sorted[train_mask].copy()
        df_val   = df_sorted[val_mask].copy()
        df_test  = df_sorted[test_mask].copy()
     
        # If there's not enough data, break
        if (len(df_train) < 10) or (len(df_val) < 5) or (len(df_test) < 5):
            # we've reached a point where we can't form a valid fold
            break
        
        # ----- 2) Prepare scaled features & targets -----
        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        X_val   = df_val[feature_cols].values
        y_val   = df_val[target_col].values

        # drop rows that still contain NaNs (lagged-feature bleed)
        train_mask = ~np.isnan(X_train).any(axis=1)
        X_train    = X_train[train_mask]
        y_train    = y_train[train_mask]

        val_mask   = ~np.isnan(X_val).any(axis=1)
        X_val      = X_val[val_mask]
        y_val      = y_val[val_mask]

       # --- BEFORE fit: sanity of targets/features ---
        print(f"[TRAIN CMF{cmf_id}] y_train mean={float(y_train.mean()):.6e} std={float(y_train.std()):.6e} n={len(y_train)}")
        print(f"[TRAIN CMF{cmf_id}] X_train shape={X_train.shape}")
        # warn if target nearly constant
        if float(y_train.std()) < 1e-6:
            print(f"[WARN CMF{cmf_id}] y_train ~ constant; model will collapse to intercept.")
            
        # Fit scaler on train, transform train & val
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # ----- 3) Fit the model in two steps: train => validate -----
        # Weighted training to emphasize more recent data
        train_dates = df_train[date_col]
        days_since_start = (train_dates - train_dates.iloc[0]).dt.days
        total_days = (train_dates.iloc[-1] - train_dates.iloc[0]).days + 1e-6
        sample_weights = 1.0 + days_since_start / total_days

        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.concatenate([y_train, y_val])
        sample_weights_combined = np.concatenate([sample_weights, np.ones(len(y_val))])
     
        # Fit on training
        # Fine-tune on validation
        # model.fit(X_combined, y_combined, sample_weight=sample_weights_combined)
        model.fit(X_combined, y_combined)

        # --- AFTER fit: model richness ---
        if hasattr(model, "coef_"):
            w = np.asarray(model.coef_).ravel()
            nz = int(np.count_nonzero(w))
            print(f"[POST-FIT CMF{cmf_id}] nz={nz}/{len(w)}  L1={float(np.abs(w).sum()):.3e}  L2={float(np.linalg.norm(w)):.3e}  intercept={float(getattr(model,'intercept_',0.0)):.6e}")
        elif hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).ravel()
            nz = int(np.count_nonzero(imp))
            print(f"[POST-FIT CMF{cmf_id}] importances nz={nz}/{len(imp)} sum={float(imp.sum()):.3f}")
        else:
            print(f"[POST-FIT CMF{cmf_id}] model type {type(model).__name__} (no linear coefs/importances)")

        # ----- 4) Evaluate on that fold's test set and accumulate predictions -----
        if not df_test.empty and not df_test.isnull().values.any():
            X_test = df_test[feature_cols].values
            y_test = df_test[target_col].values

            # Scale test features with the *existing* scaler
            X_test_scaled  = scaler.transform(X_test)
            pred_test = model.predict(X_test_scaled)

            # SAFETY CHECK: Length match
            if len(y_test) == len(pred_test):
                partial_df = pd.DataFrame({
                    "Date": df_test[date_col].values,
                    f"Act_CMF{cmf_id}": y_test,
                    f"Pred_CMF{cmf_id}": pred_test,
                    "FoldID": [f"Fold{fold_id}"] * len(y_test)
                })
                all_oos.append(partial_df)
            else:
                print(f"Skipping Fold {fold_id}: length mismatch (y_test={len(y_test)}, pred_test={len(pred_test)})")


        fold_id += 1

        #  # ----- 5) Slide all windows forward by 6 months -----
        current_train_start = current_train_start + pd.DateOffset(months=6)

        if current_train_start > final_date:
            break
    
    # Save the final scaler so we can transform features during inference
    scaler_filename = f"vix_pkg/models/scaler_cmf{cmf_id}.pkl"
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)

    # ----- 6) Compute overall RMSE & IC from all OOS predictions -----
    # Combine all folds
    if len(all_oos) == 0:
        print(f"No valid folds => no out-of-sample predictions for CMF{cmf_id}.")
        return np.nan, np.nan, pd.DataFrame()

    all_oos_df = pd.concat(all_oos, ignore_index=True)
    all_oos_df.sort_values(["FoldID", "Date"], inplace=True)
    all_oos_df.reset_index(drop=True, inplace=True)

    # Compute overall RMSE & IC
    merged_act = all_oos_df[f"Act_CMF{cmf_id}"].values
    merged_pred = all_oos_df[f"Pred_CMF{cmf_id}"].values

    rmse = math.sqrt(mean_squared_error(merged_act, merged_pred))
    if np.std(merged_act) < 1e-14 or np.std(merged_pred) < 1e-14:
        ic = 0.0
    else:
        ic = np.corrcoef(merged_act, merged_pred)[0, 1]

    print(f"CMF{cmf_id}: total folds={fold_id}, final RMSE={rmse:.5f}, IC={ic:.4f}")
    return model, rmse, ic, all_oos_df

def train_six_cmf_models(data_file):

    """
    1. Reads the data file
    2. Creates a log-return target for each CMF
    3. Performs a simple correlation-based feature selection for each CMF
    4. Trains a model for each CMF in a walk-forward manner
    5) Merge OOS predictions for all 6 CMFs
    6) Save to Excel, return DataFrame
    """
    # Load the dataset.
    # %%
    df = pd.read_excel(data_file, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # 0. create next-day targets 
    for i in range(1, 7):
        df[f"target_{i}"] = np.log(df[f"CMF{i}"].shift(-1) / df[f"CMF{i}"])

    
    # -------------------------------------------------------------------------------------------------------
    # THIS BLOCK CAN BE IGNORED IF WE ARE USING THE CURRENT DAY CMF FOR PREDICTIONS
    # THIS BLOCK WILL CREATE A 1 DAY LAG AND USE YESTERDAYS CMF FOR PREDICTIONS
    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------
    # 1. and 2.  build safe, after-close features
    # 1. One-day-lag CMF levels  (we trade next-day, so Lag1 is 100 % safe)
    # For June 6 EOD trade, use June 5 row data because otherwise June 6 row data is being fed into the target and also trained on (atp we are measuring math rather than market skill)
    for i in range(1, 7):
        df[f"CMF{i}_Lag1"] = df[f"CMF{i}"].shift(1)


    # 2. Term-structure features based on *lagged* levels
    for k in (2, 3, 4, 5, 6):
        df[f"Delta_Roll_{k}"] = df["CMF1_Lag1"] - df[f"CMF{k}_Lag1"]
        df[f"Pct_Roll_{k}"]   = df[f"CMF{k}_Lag1"] / df["CMF1_Lag1"] - 1

    # # 3. Drop same-day CMF levels – they would leak today’s hierarchy
    # # original CMF1-6 columns leak the intraday hierachy
    # # If the model is allowed to see today’s close CMF𝑡 and the target label’s denominator also contains CMF𝑡, a linear model can “solve the fraction” instead of learning market structure
    # raw_cmf_cols = [c for c in df.columns if c.startswith("CMF") and len(c) == 4]
    # df.drop(columns=raw_cmf_cols, inplace=True)
    # -------------------------------------------------------------------------------------------------------


    # --------------------------------------------------------------
    # Leak scan & one-shot drop
    # drop columns that have a suspisiously high correlation with the target columns (arithmetic echoes, forward filled data)
    target_next = df["target_1"].copy()     # keep a reference BEFORE we drop it
    target_next2 = df["target_2"].copy()
    target_next3 = df["target_3"].copy()
    target_next4 = df["target_4"].copy()
    target_next5 = df["target_5"].copy()
    target_next6 = df["target_6"].copy()

    leak_cols = []
    for col in df.columns:
        if col in ("Date",) or col.startswith("target_"):
            continue                        # skip the target_* columns themselves
        # >>> use target_next here – NOT df["target_1"] <<<
        if abs(df[col].corr(target_next)) > 0.95:
            leak_cols.append(col)
        

    if leak_cols:
        print("[WARN]  Dropping leaky columns:", leak_cols)
        df.drop(columns=leak_cols, inplace=True)
        print(f" Dropped {len(leak_cols)} highly correlated features: {leak_cols}")

    # --------------------------------------------------------------
    
    # # # using lagged CMF and delta roll
    # features_cmf1 = ["CMF1_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF1_02"]
    # features_cmf2 = ["CMF2_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF2_02"]
    # features_cmf3 = ["CMF3_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF3_02"]
    # features_cmf4 = ["CMF4_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF4_02"]
    # features_cmf5 = ["CMF5_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF5_02"]
    # features_cmf6 = ["CMF6_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF6_02"]

    # # using lagged CMF and pct roll
    # features_cmf1 = ["CMF1_Lag1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF1_02"]
    # features_cmf2 = ["CMF2_Lag1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF2_02"]
    # features_cmf3 = ["CMF3_Lag1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF3_02"]
    # features_cmf4 = ["CMF4_Lag1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF4_02"]
    # features_cmf5 = ["CMF5_Lag1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF5_02"]
    # features_cmf6 = ["CMF6_Lag1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF6_02"]
    
    # # using current CMF and delta roll
    features_cmf1 = ["CMF1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF1_02"]
    features_cmf2 = ["CMF2", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF2_02"]
    features_cmf3 = ["CMF3", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF3_02"]
    features_cmf4 = ["CMF4", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF4_02"]
    # AFTER (surgical)
    features_cmf5 = ["CMF5_Lag1","Delta_Roll_3","Delta_Roll_4","Delta_Roll_5","Delta_Roll_6","Delta_Roll_2","TLT US Equity","deltalag_CMF5_02"]
    features_cmf6 = ["CMF6_Lag1","Delta_Roll_3","Delta_Roll_4","Delta_Roll_5","Delta_Roll_6","Delta_Roll_2","TLT US Equity","deltalag_CMF6_02"]
    # # # using current CMF and pct roll
    # features_cmf1 = ["CMF1", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF1_02"]
    # features_cmf2 = ["CMF2", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF2_02"]
    # features_cmf3 = ["CMF3", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF3_02"]
    # features_cmf4 = ["CMF4", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF4_02"]
    # features_cmf5 = ["CMF5", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF5_02"]
    # features_cmf6 = ["CMF6", "Pct_Roll_3", "Pct_Roll_4", "Pct_Roll_5", "Pct_Roll_6", "Pct_Roll_2", "TLT US Equity", "deltalag_CMF6_02"]

    # Create a mapping from CMF number to its feature set.
    features_by_cmf = {
        1: features_cmf1,
        2: features_cmf2,
        3: features_cmf3,
        4: features_cmf4,
        5: features_cmf5,
        6: features_cmf6
    }


    # 4. Train a model for each CMF using its selected features
    merged_results = []
    for i in range(1, 7):
        feature_cols = features_by_cmf[i]
        target_name = f"target_{i}"


        # Subset DataFrame to [Date, features, target]
        required_cols = ["Date"] + feature_cols + [target_name]
        df_subset = df[required_cols].copy()

        print(f"\n--- Training model for CMF{i} with features: {feature_cols} ---")
        
        model, rmse, ic, oos_df = walk_forward_train_cmf(
            df_subset,
            target_col=target_name,
            feature_cols=feature_cols,
            cmf_id=i,
            date_col="Date",
            train_start_date=pd.Timestamp("2005-01-01"),
            train_length_months=54, # 4.5 years
            val_length_months=6,
            test_length_months=6
        )
        merged_results.append(oos_df)

        # Save the trained model
        model_filename = f"vix_pkg/models/model_cmf{i}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        print(f"CMF{i} -> Final RMSE: {rmse:.6f}, IC: {ic:.4f}")
        print(f"Saved model for CMF{i} to {model_filename}")     

    # 5. Merge all CMFs' OOS data
    if len(merged_results) == 0:
        print("No OOS results for any CMF.")
        return pd.DataFrame()

    final_df = reduce(
        lambda left, right: pd.merge(left, right, on=["Date", "FoldID"], how="outer"),
        merged_results
    )
    final_df.sort_values(["FoldID", "Date"], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    #  OOS coverage sanity-check
    # (e.g., 2010-01-01 → 2025-01-30)
    # Confirms that every CMF model produced predictions for the period we expect, If the start date is much later than the raw file suggests, something upstream is still filtering data away.
    print(
        "OOS date range:",
        final_df["Date"].min().date(), "->", final_df["Date"].max().date(),
        f"({len(final_df):,} rows)")

    #  DUPLICATE-COLUMN / NAN SCAN 
    dup_cols = [c for c in final_df.columns if c.endswith(("_x", "_y"))]
    if dup_cols:
        print("[WARN]  Duplicate columns created by outer-merge:", dup_cols)

    # Every date should have *exactly one* row per CMF in each fold;
    # if NaNs appear here, the outer merge was sparse.
    # Computes the percentage of missing values in every column, then prints only those ≥ 10 %
    nan_frac = final_df.isna().mean().round(3)
    print("Fraction NaNs per column >=0.10:\n",
          nan_frac[nan_frac >= 0.10])

    # Reshape to long form  ➜  Date | CMF_ID | Act | Pred
    act_cols  = [f"Act_CMF{i}"  for i in range(1, 7)]
    pred_cols = [f"Pred_CMF{i}" for i in range(1, 7)]

    # melted the six Act columns into one numeric column (Act) plus a categorical identifier CMF_ID telling which maturity bucket (1…6) the row belongs to
    long_act = (
        final_df.melt(id_vars=["Date"], value_vars=act_cols,
                      var_name="tmp", value_name="Act")
                .assign(CMF_ID=lambda x: x["tmp"].str.extract(r"(\d+)").astype(int))
                .drop(columns="tmp")
    )

    # melted the six Act columns into one numeric column (Act) plus a categorical identifier CMF_ID telling which maturity bucket (1…6) the row belongs to
    long_pred = (
        final_df.melt(id_vars=["Date"], value_vars=pred_cols,
                      var_name="tmp", value_name="Pred")
                .assign(CMF_ID=lambda x: x["tmp"].str.extract(r"(\d+)").astype(int))
                .drop(columns="tmp")
    )

    long_df = long_act.merge(long_pred, on=["Date", "CMF_ID"])

    # ----- ROW-COUNT & GLOBAL CORRELATION CHECK ----------------------
    rows_per_day = long_df.groupby("Date").size()
    print("Rows per date (value counts):\n", rows_per_day.value_counts().head())

    overall_corr = long_df["Pred"].corr(long_df["Act"])
    print("Overall time-series Pred vs Act corr:", round(overall_corr, 3))
    # ----------------------------------------------------------------

    # 2) Main metric  –  daily Spearman rank-IC across the six CMFs
    # For each trading day:
    # Rank the six CMFs by predicted return (g["Pred"].rank()).
    # Rank them by their realised return (g["Act"].rank()).
    # Compute the Spearman (rank) correlation between those two 6-element rank vectors.

    daily_cs_ic = (
        long_df.groupby("Date", group_keys=False)      # <-- avoids future-pandas warning
               .apply(lambda g: g["Pred"].rank()
                                .corr(g["Act"].rank(), method="spearman"))
               .dropna()
    )
    cs_ic_mean = daily_cs_ic.mean()
    cs_ic_t    = cs_ic_mean / daily_cs_ic.std(ddof=1) * np.sqrt(len(daily_cs_ic))

    print(f"\nCross-sectional IC (OOS): {cs_ic_mean:.3f}  "
          f"(t = {cs_ic_t:.2f}, N = {len(daily_cs_ic)})")
    #  Cross-sectional IC (OOS): 0.164  (t = 11.98, N = 3792)
    
    # ---------------------------------------------------------------------------
    # 3) QUICK SANITY TESTS  –  baseline, shuffle, 1-day shift
    # ---------------------------------------------------------------------------

    # A) Naïve baseline: rank by yesterday’s return only
    # If I simply ranked contracts by yesterday’s performance, how much of today’s hierarchy would I already capture?
    # Serves as a low benchmark your model should beat.
    long_df["Act_lag1"] = long_df.groupby("CMF_ID")["Act"].shift(1)

    baseline_ic = (
        long_df.dropna()
               .groupby("Date", group_keys=False)
               .apply(lambda g: g["Act_lag1"].rank()
                                .corr(g["Act"].rank(), method="spearman"))
               .mean()
    )
    print(f"Baseline IC (rank by Act_lag1-1 only): {baseline_ic:.3f}")

    # B) Shuffle test: randomly permute predictions (should → ~0)
    shuffled = long_df.copy()
    shuffled["Pred_shuff"] = shuffled["Pred"].sample(frac=1, random_state=0).values

    shuffle_ic = (
        shuffled.groupby("Date", group_keys=False)
                .apply(lambda g: g["Pred_shuff"].rank()
                                 .corr(g["Act"].rank(), method="spearman"))
                .mean()
    )
    print(f"IC after shuffling predictions: {shuffle_ic:.3f}")

    # C) Forward-shift test: use T-2 preds to forecast T
    long_df["Pred_shift1"] = long_df.groupby("CMF_ID")["Pred"].shift(1)

    shift_ic = (
        long_df.dropna()
               .groupby("Date", group_keys=False)
               .apply(lambda g: g["Pred_shift1"].rank()
                                .corr(g["Act"].rank(), method="spearman"))
               .mean()
    )
    print(f"IC when predictions are shifted +1 day: {shift_ic:.3f}")
    # ---------- END NEW BLOCK -------------------------------------------------- #

    # Merge SPX column into final df
    spy_column = None
    for col in df.columns:
        if 'SPX' in col:
            spy_column = col
            break
    if spy_column:
        merged_df = final_df.merge(df[['Date', spy_column]], on='Date', how='left')
        merged_df.rename(columns={spy_column: 'SPX'}, inplace=True)
        

    # Save
    merged_df.to_excel("vix_pkg/data/all_cmfs_oos_predictions.xlsx", index=False)
    print("Saved combined day-by-day OOS to all_cmfs_oos_predictions.xlsx")

    return final_df

if __name__ == "__main__":
    data = "vix_pkg/data/vix_features_calculated.xlsx"
    train_six_cmf_models(data)
# %%
