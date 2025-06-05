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
ETF_TICKERS = [
    "IAU", "SVIX", "SVXY", "UVXY", "VIXM", "VXX", "VXZ"
]

DATA_DIR = Path(__file__).with_suffix('').parent / "data"
VIX_PATH = DATA_DIR / "vixprices.xlsx"
ETF_PATH = DATA_DIR / "etf_prices.xlsx"

# ─── Helpers ────────────────────────────────────────────────────────────
def fetch_latest_closes(tickers: list[str], add_caret: bool = False) -> pd.Series:
    """Return latest close for each ticker plus 'Date' (as datetime.date)."""
    closes: dict[str, float] = {}
    latest_date: datetime.date | None = None

    for tk in tickers:
        yf_symbol = "^" + tk if add_caret else tk
        hist = yf.Ticker(yf_symbol).history(period="5d", interval="1d")["Close"].dropna()
        if hist.empty:
            raise ValueError(f"No recent data for {tk}")
        closes[tk] = float(hist.iloc[-1])
        latest_date = hist.index[-1].date()
        print(f"Close for {tk} is {closes[tk]} on {latest_date}")

    closes["Date"] = latest_date
    return pd.Series(closes)

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
