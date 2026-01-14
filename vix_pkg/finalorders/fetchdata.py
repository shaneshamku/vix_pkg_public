"""
Append the most recent trading day's values for:
  ["^SPVXSP","^SPVIX2ME","^SPVIX3ME","^SPVIX4ME","^SPVXMP","^SPVIX6ME","^SPX","TLT","^VIX"]
to Excel at: vix_pkg/data/vixprices.xlsx, using custom column names.

Run:
    python pull_latest_vix_row.py

Req:
    pip install yfinance pandas openpyxl
"""
# %%
from pathlib import Path
import pandas as pd
import yfinance as yf
# %%
tickers = ["^SPVXSP", "^SPVIX2ME", "^SPVIX3ME", "^SPVIX4ME", "^SPVXMP", "^SPVIX6ME", "SPY", "TLT", "^VIX"]
# %%
# Download last 5 days of data for all tickers
df = yf.download(
    tickers=tickers,
    period="5d",
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    threads=True,
    progress=True,
)
# %%
print("Raw yfinance DataFrame:")
print(df)
# %%
# Only show the Close prices
close_prices = df["Close"] if "Close" in df.columns else df.xs("Close", axis=1, level=1)
print(close_prices)
# %%
# --- Config ------------------------------------------------------------------
TICKER_TO_COL = {
    "^SPVXSP": "SPVXSP",
    "^SPVIX2ME": "SPVIX2ME",
    "^SPVIX3ME": "SPVIX3ME",
    "^SPVIX4ME": "SPVIX4ME",
    "^SPVXMP": "SPVXMP",
    "^SPVIX6ME": "SPVIX6ME",
    "SPY": "SPX",
    "TLT": "TLT US Equity",
    "^VIX": "VIX Index",
}
EXCEL_PATH = Path("vix_pkg/data/vixprices.xlsx")

# ETF needs Adj Close, everything else Close
FIELD_MAP = {"TLT": "Adj Close"}

# --- Helpers -----------------------------------------------------------------
def _extract_series(df: pd.DataFrame, ytk: str, field: str) -> pd.Series:
    """Return a clean Series with dates as index."""
    try:
        ser = df[(ytk, field)].dropna()
        if not ser.empty:
            return ser
    except Exception:
        pass

    if field in df.columns and ytk in df[field].columns:
        ser = df[field][ytk].dropna()
        if not ser.empty:
            return ser

    if ytk in df.columns:
        ser = df[ytk].dropna()
        if not ser.empty:
            return ser

    return pd.Series(dtype=float)


def download_all() -> dict[str, pd.Series]:
    """Download all tickers in TICKER_TO_COL with appropriate field."""
    tickers = list(TICKER_TO_COL.keys())
    df = yf.download(
        tickers=tickers,
        period="15d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out = {}
    for tk in tickers:
        field = FIELD_MAP.get(tk, "Close")
        ser = _extract_series(df, tk, field)
        if ser.empty:
            raise ValueError(f"No recent {field} data for {tk}")
        out[tk] = ser
    return out


def latest_common_date(values: dict[str, pd.Series]) -> pd.Timestamp:
    """Most recent date present in all series."""
    common_idx = None
    for s in values.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common trading date across requested symbols.")
    return max(common_idx)


def upsert_row(row: pd.Series, path: Path, col_order: list[str]) -> None:
    """Insert/replace by Date at top of Excel file with fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Date", *col_order]

    if path.exists():
        df = pd.read_excel(path, engine="openpyxl")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            df = df[df["Date"] != row["Date"]]
        else:
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)

    new_row = pd.DataFrame([row[cols]])
    updated = pd.concat([new_row, df], ignore_index=True)
    updated.to_excel(path, index=False, engine="openpyxl")
    print(f"{path.name}: wrote row for {row['Date']}")


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    
    try:
        series_map = download_all()
        common_last = latest_common_date(series_map)

        # Build output row with mapped column names
        row = {"Date": common_last.date()}
        for tk, col in TICKER_TO_COL.items():
            row[col] = float(series_map[tk].loc[common_last])

        # Write in the mapped order
        col_order = list(TICKER_TO_COL.values())
        upsert_row(pd.Series(row), EXCEL_PATH, col_order)

    except Exception as e:
        print("Error:", e)

