
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
    model = ElasticNet(alpha=0.001,
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

    # 1) Next-day log returns
    # log-return = ln(CMF_{t+1} / CMF_{t})
    for i in range(1, 7):
        df[f"target_{i}"] = np.log(df[f"CMF{i}"].shift(-1) / df[f"CMF{i}"])

    # # Compute feature correlations
    # correlation_matrix = df.corr()
    # print("Feature Correlation Matrix:\n", correlation_matrix)
    
    # Define feature sets for each CMF. Modify these arrays as needed.
    features_cmf1 = ["CMF1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF1_02"]
    features_cmf2 = ["CMF2", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF2_02"]
    features_cmf3 = ["CMF3", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF3_02"]
    features_cmf4 = ["CMF4", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF4_02"]
    features_cmf5 = ["CMF5", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF5_02"]
    features_cmf6 = ["CMF6", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF6_02"]
    
    # Create a mapping from CMF number to its feature set.
    features_by_cmf = {
        1: features_cmf1,
        2: features_cmf2,
        3: features_cmf3,
        4: features_cmf4,
        5: features_cmf5,
        6: features_cmf6
    }

    # 2) Correlation-based feature selection
    # Pick top 10 correlated features with the target_i
    # (excluding the target itself and the Date column)
    all_columns = df.columns.tolist()
    # Store the final selected features in this dict
    
    selection_records = []


    # 3) Train a model for each CMF using its selected features
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
            train_start_date=pd.Timestamp("2018-01-01"),
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

    # 4) Merge all CMFs' OOS data
    if len(merged_results) == 0:
        print("No OOS results for any CMF.")
        return pd.DataFrame()

    final_df = reduce(
        lambda left, right: pd.merge(left, right, on=["Date", "FoldID"], how="outer"),
        merged_results
    )
    final_df.sort_values(["FoldID", "Date"], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

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