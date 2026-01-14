from __future__ import annotations
import pandas as pd
import numpy as np
import json, hashlib
def _sha1(d): 
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
# def calculate_features():
#     file_path = "vix_pkg/data/vixprices.xlsx"
#     df = pd.read_excel(file_path)

#     # Remove first row and name column 1 as 'Date'
#     df = df.iloc[1:]
#     df.rename(columns={df.columns[0]: 'Date'}, inplace=True)


#     # Rename columns for simplicity
#     df.rename(columns={
#         'SPVIX2ME': 'CMF2', 'SPVIX3ME': 'CMF3', 'SPVIX4ME': 'CMF4',
#         'SPVIX6ME': 'CMF6', 'SPVXMP': 'CMF5', 'SPVXSP': 'CMF1'
#     }, inplace=True)


#     # List of approximated time to maturity (using trading days)
#     maturity_days = {
#         'CMF1': 21,  # Approx. 1 month
#         'CMF2': 42,  # Approx. 2 months
#         'CMF3': 63,  # Approx. 3 months
#         'CMF4': 84,  # Approx. 4 months
#         'CMF5': 105, # Approx. 5 months
#         'CMF6': 126  # Approx. 6 months
#     }

#     # Calculate Roll
#     for i in range(2, 7):  # CMF2 to CMF6
#         current_cmf = f'CMF{i}'
#         prev_cmf = f'CMF{i-1}'
#         T_current = maturity_days[current_cmf]
#         T_prev = maturity_days[prev_cmf]
        
#         # Calculate Roll
#         df[f'Roll_{i}'] = (df[current_cmf] - df[prev_cmf]) / (df[prev_cmf] * (T_current - T_prev) * (1/252))


#     # Calculate Delta Roll (change in roll yield)
#     for i in range(2, 7):  # CMF2 to CMF6
#         df[f'Delta_Roll_{i}'] = df[f'Roll_{i}'].diff()



#     # Calculate change in CMF (mu)
#     for i in range(2, 7):  # CMF2 to CMF6
#         current_cmf = f'CMF{i}'
#         prev_cmf = f'CMF{i-1}'
#         df[f'mu_{i}'] = df[current_cmf] - df[prev_cmf]

#     # Create lag columns for CMF variables (CMF1 to CMF6) for 1, 2, and 3 days
#     for cmf in ['CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']:
#         df[f'lag_{cmf}_01'] = df[cmf].shift(1)
#         df[f'lag_{cmf}_02'] = df[cmf].shift(2)
#         df[f'lag_{cmf}_03'] = df[cmf].shift(3)

#     # Create deltalag columns
#     for cmf in ['CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']:
#         # deltalag_CMFX_01: Today's price minus yesterday's price
#         df[f'deltalag_{cmf}_01'] = df[cmf] - df[f'lag_{cmf}_01']
        
#         # deltalag_CMFX_02: Yesterday's price minus the price from 2 days ago
#         df[f'deltalag_{cmf}_02'] = df[f'lag_{cmf}_02'] - df[f'lag_{cmf}_01']
        
#         # deltalag_CMFX_03: The price from 3 days ago minus the price from 2 days ago
#         df[f'deltalag_{cmf}_03'] = df[f'lag_{cmf}_03'] - df[f'lag_{cmf}_02']


#     # Create Delta_Roll_X_Y columns for all pairs of CMFs (with Y > X)
#     for x in range(1, 7):
#         for y in range(x + 1, 7):
#             cmf_x = f'CMF{x}'
#             cmf_y = f'CMF{y}'
#             T_x = maturity_days[cmf_x]
#             T_y = maturity_days[cmf_y]
#             # Calculate generalized roll yield for the pair (X, Y)
#             # This formula is analogous to the one used for consecutive maturities,
#             # but now uses the difference in maturity for any pair.
#             df[f'Delta_Roll_{x}_{y}'] = (df[cmf_y] - df[cmf_x]) / (df[cmf_x] * (T_y - T_x) * (1/252))



#     # rearrange columns
#     cmf_columns = ['Date', 'CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']
#     other_columns = [col for col in df.columns if col not in cmf_columns]
#     df = df[cmf_columns + other_columns]
#     df.head()


#     # rearrange date column and set to datetime
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.sort_values('Date', inplace=True)


#     # calculate next day returns (target variable)
#     cmf_cols = ['CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']
#     for col in cmf_cols:
#         df[f'Return_{col}'] = df[col].pct_change().shift(-1) # Next-day return

#     # drop rows with nan (first and last row)
#     df.dropna(inplace=True)

#     df.to_excel("vix_pkg/data/vix_features_calculated.xlsx", index=False)



from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ---- Paths (this file is .../vix_pkg/vix_pkg/calculate_features.py) ----
INNER = Path(__file__).resolve().parent            # .../vix_pkg/vix_pkg
PRICES_XLSX   = INNER / "data" / "vixprices.xlsx"                  # read here
FEATURES_XLSX = INNER / "data" / "vix_features_calculated.xlsx"    # write here

def calculate_features():
    # ---------- 1) Load prices ----------
    df = pd.read_excel(PRICES_XLSX, engine="openpyxl")
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Map Yahoo/SP tickers to CMF1..CMF6
    cmf_map = {
        "SPVXSP": "CMF1",
        "SPVIX2ME": "CMF2",
        "SPVIX3ME": "CMF3",
        "SPVIX4ME": "CMF4",
        "SPVXMP": "CMF5",
        "SPVIX6ME": "CMF6",
    }
    missing = [k for k in cmf_map if k not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in prices file: {missing} ({PRICES_XLSX})")

    base_cols = ["Date"] + list(cmf_map.keys())
    base = df[base_cols].copy().rename(columns=cmf_map)

    # Optional passthroughs your model uses
    passthrough_cols = ["SPX", "TLT US Equity", "VIX Index"]
    present_pt = [c for c in passthrough_cols if c in df.columns]
    if present_pt:
        base = base.merge(df[["Date"] + present_pt], on="Date", how="left")
    else:
        print("[WARN] No passthrough columns found among:", passthrough_cols)

    # ---------- 2) Feature engineering (all info ≤ t only) ----------
    # Trading-day approximations for maturities
    maturity_days = {
        "CMF1": 21, "CMF2": 42, "CMF3": 63, "CMF4": 84, "CMF5": 105, "CMF6": 126
    }

    # (a) Roll yields between adjacent CMFs (2..6 vs previous)
    for i in range(2, 7):
        c, p = f"CMF{i}", f"CMF{i-1}"
        Tc, Tp = maturity_days[c], maturity_days[p]
        base[f"Roll_{i}"] = (base[c] - base[p]) / (base[p] * (Tc - Tp) * (1/252))

    # (b) Delta Roll (changes)
    for i in range(2, 7):
        base[f"Delta_Roll_{i}"] = base[f"Roll_{i}"].diff()

    # (c) Level spreads (mu)
    for i in range(2, 7):
        c, p = f"CMF{i}", f"CMF{i-1}"
        base[f"mu_{i}"] = base[c] - base[p]

    # (d) Lags and delta-lags for all CMFs
    for k in range(1, 7):
        cmf = f"CMF{k}"
        base[f"lag_{cmf}_01"] = base[cmf].shift(1)
        base[f"lag_{cmf}_02"] = base[cmf].shift(2)
        base[f"lag_{cmf}_03"] = base[cmf].shift(3)

        base[f"deltalag_{cmf}_01"] = base[cmf] - base[f"lag_{cmf}_01"]
        base[f"deltalag_{cmf}_02"] = base[f"lag_{cmf}_02"] - base[f"lag_{cmf}_01"]
        base[f"deltalag_{cmf}_03"] = base[f"lag_{cmf}_03"] - base[f"lag_{cmf}_02"]

    # (e) Generalized roll across any pair X<Y
    for x in range(1, 7):
        for y in range(x+1, 7):
            X, Y = f"CMF{x}", f"CMF{y}"
            Tx, Ty = maturity_days[X], maturity_days[Y]
            base[f"Delta_Roll_{x}_{y}"] = (base[Y] - base[X]) / (base[X] * (Ty - Tx) * (1/252))

    # ---------- 3) Targets (training only): next-day log returns ----------
    # Keep them in the file, but DO NOT drop the last row because targets are NaN there.
    for k in range(1, 7):
        cmf = f"CMF{k}"
        base[f"Return_{cmf}"] = np.log(base[cmf].shift(-1) / base[cmf])

    # ---------- 4) Row filtering: drop only if FEATURE INPUTS are missing ----------
    # Your training script currently uses these features (see features_cmf* in copy_train_*):
    # ["CMF{i}" (or CMF{i}_Lag1), "Delta_Roll_2..6", "TLT US Equity", "deltalag_CMF{i}_02"]
    # Build a union of all required feature inputs:
    feature_inputs = set()
    for i in range(1, 7):
        feature_inputs.add(f"CMF{i}")                   # you currently use unlagged CMF in training
        feature_inputs.add(f"deltalag_CMF{i}_02")
    feature_inputs.update([f"Delta_Roll_{k}" for k in range(2, 7)])
    if "TLT US Equity" in base.columns:   # may be missing early history
        feature_inputs.add("TLT US Equity")

    must_have = ["Date"] + sorted(feature_inputs)
    # Drop rows that miss any of the inputs above, but keep last date even if Return_* are NaN
    feat = base.dropna(subset=must_have).copy()

    # ---------- 5) Save (keep targets even if NaN at the end) ----------
    FEATURES_XLSX.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Max price date in file: ", base["Date"].max().date())
    print("[INFO] Max feature date (inputs):", feat["Date"].max().date())
    feat.to_excel(FEATURES_XLSX, index=False, engine="openpyxl")

    # Quick re-open check
    try:
        chk = pd.read_excel(FEATURES_XLSX, engine="openpyxl")
        chk["Date"] = pd.to_datetime(chk["Date"]).dt.date
        print("[OK] Features file last Date:", chk["Date"].max())
    except Exception as e:
        print("[WARN] Wrote features but could not re-open:", e)

    try:
        feat_dbg = pd.read_excel(FEATURES_XLSX, engine="openpyxl")
        feat_dbg["Date"] = pd.to_datetime(feat_dbg["Date"])
        last = feat_dbg.iloc[-1].copy()
        sig_cols = [c for c in feat_dbg.columns if c.startswith(("CMF","Delta_Roll","deltalag_"))] + ["TLT US Equity"]
        row_sig = {k: (None if pd.isna(last.get(k)) else float(last.get(k))) for k in ["Date", *sig_cols] if k in last}
        print("[FEATURE SNAPSHOT] last_date=", str(row_sig["Date"].date()),
            " sha1=", _sha1(row_sig),
            " non_null=", int(pd.Series(row_sig).notna().sum())-1,
            " nulls=", int(pd.Series(row_sig).isna().sum()))
    except Exception as e:
        print("[FEATURE SNAPSHOT] failed:", e)

if __name__ == "__main__":
    calculate_features()