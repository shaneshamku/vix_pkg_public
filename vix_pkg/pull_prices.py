# pull_prices.py  ──────────────────────────────────────────────────────────
"""
Fetch latest closes for:
  • VIX-term indices -> data/vixprices.xlsx
  • Volatility ETFs  -> data/etf_prices.xlsx

Run:
    python pull_prices.py         # fetch & append both sets

Requirements:
    pip install yfinance pandas openpyxl
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

# ─── Symbol sets & destinations ─────────────────────────────────────────
INDEX_TICKERS = [
    "SPVIX2ME", "SPVIX3ME", "SPVIX4ME",
    "SPVIX6ME", "SPVXMP", "SPVXSP"
]
REF_MAP = {
    "SPY": "SPY",
    "TLT": "TLT",
    "VIX Index": "^VIX",
}
ETF_TICKERS = [
    "IAU", "SVIX", "SVXY", "UVXY", "VIXM", "VXX", "VXZ"
]

DATA_DIR = Path(__file__).with_suffix('').parent / "data"
VIX_PATH = DATA_DIR / "vixprices.xlsx"
ETF_PATH = DATA_DIR / "etf_prices.xlsx"

# ─── Helpers ────────────────────────────────────────────────────────────
def fetch_cmf_prices(tickers, out_path):
    # Load existing data if available
    try:
        existing = pd.read_excel(out_path, parse_dates=["Date"])
        last_date = existing["Date"].max()
    except FileNotFoundError:
        existing = pd.DataFrame()
        last_date = None

    # Determine start date for new data
    if last_date is not None:
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = "2000-01-01"  # or earliest date you want

    end_date = datetime.now().strftime("%Y-%m-%d")

    # Download missing data for each ticker
    new_data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)["Close"]
        new_data[ticker] = df

    new_data.index = pd.to_datetime(new_data.index)
    new_data.reset_index(inplace=True)
    new_data.rename(columns={"index": "Date"}, inplace=True)

    # Merge with existing data, avoiding duplicates
    if not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
    else:
        combined = new_data

    # Reorder columns
    combined = combined[["Date"] + tickers]
    combined.sort_values("Date", inplace=True)
    combined.to_excel(out_path, index=False)

def upsert_row(series: pd.Series, path: Path, tickers: list[str]) -> None:
    """Insert `series` at top (row 2) of Excel sheet; replace if date exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Date", *tickers]

    if path.exists():
        df = pd.read_excel(path, engine="openpyxl")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[df["Date"] != series["Date"]]      # drop any existing row
    else:
        df = pd.DataFrame(columns=cols)

    new_row = pd.DataFrame([series[cols]])
    updated = pd.concat([new_row, df], ignore_index=True)
    updated.to_excel(path, index=False, engine="openpyxl")
    print(f"{path.name}: wrote row for {series['Date']}")

# ─── Main routine ────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # VIX-term indices (need caret prefix)
        vix_series = fetch_latest_closes(INDEX_TICKERS, add_caret=True)
        upsert_row(vix_series, VIX_PATH, INDEX_TICKERS)

        # Vol-ETFs / gold (no caret)
        etf_series = fetch_latest_closes(ETF_TICKERS, add_caret=False)
        upsert_row(etf_series, ETF_PATH, ETF_TICKERS)

    except Exception as e:
        print("Error:", e)
