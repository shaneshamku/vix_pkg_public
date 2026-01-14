# vix_pkg/finalorders/feature_engineering.py
import sys
import subprocess
from pathlib import Path
import argparse
import pandas as pd

def main(debug: bool = False):
    """
    Runs the features calculation step and verifies the output.

    Layout assumed:
        New VIX/
          vix_pkg/
            vix_pkg/
              calculate_features.py
              data/vix_features_calculated.xlsx
            finalorders/
              feature_engineering.py  <-- this file
    """

        # This file lives at .../vix_pkg/vix_pkg/finalorders/feature_engineering.py
    INNER = Path(__file__).resolve().parents[1]   # => .../vix_pkg/vix_pkg
    PY = sys.executable

    calc_script = INNER / "calculate_features.py"
    out_excel   = INNER / "data" / "vix_features_calculated.xlsx"

    if not calc_script.exists():
        raise FileNotFoundError(f"calculate_features.py not found at: {calc_script}")

    print(f" Running: {[PY, str(calc_script)]}")
    cmd = [PY, str(calc_script)]
    print("  Running:", cmd)
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        # ALWAYS dump child output on error so we can see the real exception
        print("----- calculate_features.py STDOUT -----")
        print(res.stdout)
        print("----- calculate_features.py STDERR -----")
        print(res.stderr)
        res.check_returncode()  # raise after printing
    else:
        # Optional: show something on success (last line of stdout if any)
        if res.stdout.strip():
            last = res.stdout.strip().splitlines()[-1]
            print("  ->", last)

    if not out_excel.exists():
        raise FileNotFoundError(f"Expected output not found: {out_excel}")

    # Log the last feature date
    df = pd.read_excel(out_excel, engine="openpyxl")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        print(f"Features written: {out_excel}")
        print(f"   Last feature date: {df['Date'].max()}")
    else:
        print(f"Features written: {out_excel} (no 'Date' column)")

if __name__ == "__main__":
    # Quick feature sanity check before training (zero-variance or all-NaN)
    try:
        df = pd.read_excel("vix_pkg/data/vix_features_calculated.xlsx", engine="openpyxl")
        for col in ["deltalag_CMF5_02", "deltalag_CMF6_02", "Delta_Roll_2", "Delta_Roll_3", "Delta_Roll_4", "Delta_Roll_5", "Delta_Roll_6"]:
            if col in df.columns:
                if float(df[col].std()) == 0.0 or df[col].isna().all():
                    print(f"[WARN] {col} has zero variance or all-NaN in latest feature file.")
    except Exception as e:
        print("[WARN] feature variance check failed:", e)
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    main(debug=args.debug)