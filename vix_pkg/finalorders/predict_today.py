# vix_pkg/finalorders/predict_today.py
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import os, json, hashlib, datetime as dt
def _sha1(d): return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
def _mtime(p): return (dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat() if os.path.exists(p) else "MISSING")

# ==== BEGIN PATCH: helpers ====
from typing import List
import joblib
import pandas as pd

def load_model_and_columns(path: str, cmf_idx: int):
    """
    Load sklearn model and recover its feature list.
    Priority:
      1) attribute feature_names_in_ (if present)
      2) sidecar JSON "[model].cols.json" (list of strings)
      3) EXPECTED_FEATS fallback by CMF index
    """
    mdl = joblib.load(path)

    # 1) try sklearn attribute
    if hasattr(mdl, "feature_names_in_") and mdl.feature_names_in_ is not None:
        try:
            return mdl, list(mdl.feature_names_in_)
        except Exception:
            pass

    # 2) try sidecar json
    sidecar = os.fspath(Path(path).with_suffix(Path(path).suffix + ".cols.json"))
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                cols = json.load(f)
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return mdl, cols
        except Exception:
            pass

    # 3) fallback to hard-coded mapping
    if cmf_idx in FEATURES_BY_CMF:
        print(f"[WARN] feature_names_in_ missing on {Path(path).name}. Using EXPECTED_FEATS[{cmf_idx}].")
        return mdl, FEATURES_BY_CMF[cmf_idx][:]

    raise RuntimeError(f"Cannot determine feature list for {path}. Provide a sidecar JSON with columns.")

def ensure_predict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure lagged and alias columns exist BEFORE we slice the 'today' row.
    Build on full history so shift() has data.
    """
    # Harmonize TLT column name if needed
    if "TLT US Equity" not in df.columns:
        for alt in ("TLT", "TLT_US_Equity", "TLT_US"):
            if alt in df.columns:
                df["TLT US Equity"] = df[alt]
                break

    # CMF5/6 used Lag1 in training — create if missing
    for k in (5, 6):
        base = f"CMF{k}"
        lag1 = f"CMF{k}_Lag1"
        if lag1 not in df.columns and base in df.columns:
            df[lag1] = df[base].shift(1)

    # Build deltalag_CMFx_02 if missing (match your training definition!)
    for k in range(1, 7):
        src = f"CMF{k}"
        out = f"deltalag_CMF{k}_02"
        if out not in df.columns and src in df.columns:
            # Adjust this logic if your training used a different formula:
            df[out] = df[src] - df[src].shift(2)

    return df
# ==== END PATCH: helpers ====


# --- MODEL DIAGNOSTICS (constant-pred check) ---
def _dump_model_diagnostics(i, model, X, feats):
    try:
        # Linear models (ElasticNet/LinearRegression/etc.)
        if hasattr(model, "coef_"):
            w = np.asarray(model.coef_).ravel()
            nz = int(np.count_nonzero(w))
            l1 = float(np.abs(w).sum())
            l2 = float(np.linalg.norm(w))
            icpt = float(getattr(model, "intercept_", 0.0))
            print(f"[CMF{i}] COEF nz={nz}/{len(w)}  L1={l1:.3e}  L2={l2:.3e}  intercept={icpt:.6e}")

            # show top 5 |w*x| contributions for today's row
            contrib = {feats[j]: float(w[j] * X[0, j]) for j in range(len(feats))}
            top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            print(f"[CMF{i}] TOP CONTRIBUTIONS (w*x): {top}")

        # Tree/boosted models
        elif hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).ravel()
            nz = int(np.count_nonzero(imp))
            print(f"[CMF{i}] IMPORTANCES nz={nz}/{len(imp)}  sum={float(imp.sum()):.3f}")
            top = sorted(zip(feats, imp), key=lambda kv: kv[1], reverse=True)[:5]
            print(f"[CMF{i}] TOP FEATURES: {top}")

        else:
            print(f"[CMF{i}] (no coef_/feature_importances_ on {type(model).__name__})")
    except Exception as e:
        print(f"[CMF{i}] diag failed: {e}")

# Must match what you trained on (see copy_train_six_cmf_model.py)
FEATURES_BY_CMF = {
    1: ["CMF1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF1_02"],
    2: ["CMF2", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF2_02"],
    3: ["CMF3", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF3_02"],
    4: ["CMF4", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF4_02"],
    5: ["CMF5_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF5_02"],
    6: ["CMF6_Lag1", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6", "Delta_Roll_2", "TLT US Equity", "deltalag_CMF6_02"],
}

MODELS_DIR = Path("vix_pkg/models")
FEATURES_XLSX = Path("vix_pkg/data/vix_features_calculated.xlsx")

def predict_today():
    df = pd.read_excel(FEATURES_XLSX, engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    # --- NEW PATCH: ensure required columns exist before slicing ---
    df = ensure_predict_features(df)

    # now safely take the latest row (t)
    row = df.iloc[-1].copy()
    date_t = row["Date"].date()

    print("\n========== PREDICT_TODAY ==========")
    print("Models dir:", MODELS_DIR.resolve())
    print(f"[DATE] features Date=t => {date_t}")

    preds = {}
    for i in range(1, 7):
        model_p  = MODELS_DIR / f"model_cmf{i}.pkl"
        scaler_p = MODELS_DIR / f"scaler_cmf{i}.pkl"
        model, feat_cols = load_model_and_columns(model_p, i)
        # ensure all features exist
        if any(col not in df.columns for col in feat_cols):
            raise KeyError(f"Missing features for CMF{i}: {feat_cols}")
        
        # fail fast if a trained feature is missing
        missing = [c for c in feat_cols if c not in df.columns]
        print(f"[CMF{i}] available cols:", sorted(df.columns.tolist())[:10], "... total", len(df.columns))
        if missing:
            raise KeyError(f"Missing features for CMF{i}: {missing}")

        x_series = row[feat_cols]
        nan_count = int(x_series.isna().sum())
        payload_sig = _sha1({"feats": feat_cols, "vals": x_series.where(pd.notna(x_series), None).to_dict()})
        scaler_p = MODELS_DIR / f"scaler_cmf{i}.pkl"
        model_p  = MODELS_DIR / f"model_cmf{i}.pkl"
        print(f"[CMF{i}] NaNs={nan_count} Xshape={(1,len(feat_cols))} sig={payload_sig} "
              f"scaler={_mtime(scaler_p)} model={_mtime(model_p)}")

        if nan_count > 0:
            raise ValueError(f"CMF{i} has NaNs in X_today: {list(x_series[x_series.isna()].index)}")

        x = x_series.astype(float).values.reshape(1, -1)
        with open(scaler_p, "rb") as f: scaler = pickle.load(f)
        with open(model_p,  "rb") as f: model  = pickle.load(f)

        X = scaler.transform(x)
        xstd = float(np.std(X))
        print(f"[CMF{i}] Xstd(after-scale)={xstd:.6e}")
        if xstd == 0.0:
            raise AssertionError(f"CMF{i} scaled features are constant; investigate feature build for {feat_cols}")

        _dump_model_diagnostics(i, model, X, feat_cols)

        yhat = float(model.predict(X)[0])   # next-day (t+1) prediction
        preds[f"Pred_CMF{i}"] = yhat

    # summary
    s = pd.Series(preds, dtype="float64")
    v = s.values
    print("[PRED SUMMARY] n=6 mean=", float(np.mean(v)), " std=", float(np.std(v)),
          " min=", float(np.min(v)), " max=", float(np.max(v)))
    print("Ranked:\n", s.sort_values(ascending=False).to_string())

    out = {"Date": pd.Timestamp(date_t)}
    out.update(preds)
    return pd.DataFrame([out])

if __name__ == "__main__":
    print(predict_today())
