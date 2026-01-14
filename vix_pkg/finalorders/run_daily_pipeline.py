# vix_pkg/finalorders/run_daily_pipeline.py
import sys
import subprocess
from pathlib import Path
import pandas as pd

# ---------- Paths ----------
THIS_FILE = Path(__file__).resolve()
FINALORDERS_DIR = THIS_FILE.parent                # .../vix_pkg/vix_pkg/finalorders
PKG_ROOT = FINALORDERS_DIR.parents[1]             # .../vix_pkg  (outer folder)
INNER_PKG = PKG_ROOT / "vix_pkg"                  # .../vix_pkg/vix_pkg

print(f"current script path: {THIS_FILE}")
print(f"   FINALORDERS_DIR: {FINALORDERS_DIR}")
print(f"   PKG_ROOT:        {PKG_ROOT}")
print(f"   INNER_PKG:      {INNER_PKG}")

# Scripts we invoke
FETCH_SCRIPT   = FINALORDERS_DIR / "fetchdata.py"
FEAT_SCRIPT    = FINALORDERS_DIR / "feature_engineering.py"  # this calls inner calculate_features.py
TRAIN_SCRIPT   = FINALORDERS_DIR / "run_modeltraining.py"

# Data artifacts (adjust if your writers differ)
PRICES_XLSX    = INNER_PKG / "data"     / "vixprices.xlsx"                   # from fetchdata.py
FEATURES_XLSX  = INNER_PKG / "data"    / "vix_features_calculated.xlsx"     # from calculate_features.py
PREDICT_XLSX   = INNER_PKG / "data"     / "all_cmfs_oos_predictions.xlsx"    # from training step

PY = sys.executable


def run_script(path: Path, name: str):
    """Run a child script with the same Python interpreter; echo stdout/stderr if it fails."""
    print(f"\[RUNNING] Running {name}: {path}")
    res = subprocess.run([PY, str(path)], capture_output=True, text=True)
    if res.returncode != 0:
        print("----- STDOUT -----\n", res.stdout)
        print("----- STDERR -----\n", res.stderr)
        res.check_returncode()
    else:
        # print short stdout to help with tracing without spamming logs
        out = (res.stdout or "").strip()
        if out:
            print(out.splitlines()[-1][:300])


def show_last_date(label: str, path: Path, date_col: str = "Date"):
    """Log the last available date in an Excel file if it exists."""
    if not path.exists():
        print(f"[WARN] {label}: file not found at {path}")
        return
    try:
        df = pd.read_excel(path, engine="openpyxl")
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col]).dt.date
            print(f"[Success] {label}: last {date_col} = {df[date_col].max()}")
        else:
            print(f"[Success] {label}: loaded ({len(df):,} rows) — no '{date_col}' column")
    except Exception as e:
        print(f"[WARN]  {label}: failed to read {path} — {e}")


def main():
    # 1) Fetch latest market data
    run_script(FETCH_SCRIPT, "fetchdata.py")
    show_last_date("Prices", PRICES_XLSX)

    # 2) Feature engineering (this calls inner calculate_features.py and verifies output)
    run_script(FEAT_SCRIPT, "feature_engineering.py")
    show_last_date("Features", FEATURES_XLSX)

    # 3) Train / produce predictions
    run_script(TRAIN_SCRIPT, "run_modeltraining.py")
    show_last_date("Predictions", PREDICT_XLSX)

    # 4) Generate signals + email (import after training so dependencies are present)
    sys.path.insert(0, str(FINALORDERS_DIR))  # allow imports from finalorders/
    from generate_signals import generate_signals
    from send_email import send_signal_email

    signal = generate_signals()  # your function should read the latest predictions/prices
    print(" Signal payload:", signal)

    # --- Email: Gmail over SSL:465 (matches your working test) ---
    recipients    = ["shane.shamku@gmail.com", "casselrobson19@gmail.com"]
    # recipients    = ["shane.shamku@gmail.com"]

    smtp_server   = "smtp.gmail.com"
    smtp_port     = 465
    smtp_user     = "coolshane10@gmail.com"
    smtp_password = "iltwljyvfpsxduzs"  # <-- consider moving to an env var

    send_signal_email(
        signal=signal,
        recipients=recipients,
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        debug=False,  # flip True if you need SMTP conversation logs
    )

    print("\n[Success] Daily pipeline finished.")


if __name__ == "__main__":
    main()