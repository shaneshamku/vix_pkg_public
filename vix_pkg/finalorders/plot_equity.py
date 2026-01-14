# vix_pkg/finalorders/plot_equity.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict_today import predict_today


# --------- Config ----------
PRED_XLSX  = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
PRICE_XLSX = Path("vix_pkg/data/vixprices.xlsx")
OUT_DIR    = Path("vix_pkg/reports")
OUT_CURVE  = OUT_DIR / "equity_curve.xlsx"
OUT_PNG    = OUT_DIR / "equity_curve.png"

# Portfolio starting value for an intuitive y-axis
START_EQUITY = 100_000.0
WINDOW = 20  # rolling window for vol + drawdown logic (matches generate_signals idea)

# Pred → index mapping (same as your live code)   # :contentReference[oaicite:1]{index=1}
PRED_TO_INDEX = {
    "Pred_CMF1": "SPVXSP",
    "Pred_CMF2": "SPVIX2ME",
    "Pred_CMF3": "SPVIX3ME",
    "Pred_CMF4": "SPVIX4ME",
    "Pred_CMF5": "SPVXMP",
    "Pred_CMF6": "SPVIX6ME",
}

INDEX_COLS = list(PRED_TO_INDEX.values())

def load_data():
    pred = pd.read_excel(PRED_XLSX, parse_dates=["Date"])
    prices = pd.read_excel(PRICE_XLSX, parse_dates=["Date"])
    keep = ["Date"] + INDEX_COLS
    prices = prices[keep].dropna()
    df = pred.merge(prices, on="Date", how="inner").sort_values("Date").set_index("Date")
    return df

def daily_returns(df):
    """Log returns of the six indices."""
    rets = np.log(df[INDEX_COLS]).diff()
    return rets

def size_weights_for_day(rets_window, long_ticker, short_ticker):
    """Vol-based 2-leg sizing + simple drawdown throttle (as in your live logic)."""
    vol = rets_window.std()
    wl = 1.0 / vol[long_ticker]
    ws = 1.0 / vol[short_ticker]
    scale = 1.0 / (wl + ws)
    long_w = scale * wl
    short_w = -scale * ws

    # Drawdown throttle on the window equity (approximation used in your live code)
    combo_last_row = (pd.Series({long_ticker: long_w, short_ticker: short_w}) * rets_window.iloc[-1]).cumsum().fillna(0)
    peak = combo_last_row.cummax()
    dd = (combo_last_row - peak) / peak
    exposure = 1.0
    if dd.iloc[-1] < -0.10:
        exposure = 0.5
    if dd.iloc[-1] < -0.20:
        exposure = 0.0
    return long_w * exposure, short_w * exposure

def backtest():
    df = load_data()
    rets = daily_returns(df)

    # We’ll loop dates where we have a full WINDOW of history and a next-day return available.
    dates = df.index
    start_idx = WINDOW  # need WINDOW days of history before first signal
    # Store results per day t (the return realized on t+1 gets attributed to date t+1)
    rows = []

    for i in range(start_idx, len(dates) - 1):  # stop at len-2 because we need t+1
        t = dates[i]
        t1 = dates[i + 1]

        # Long/short choice from predictions at time t
        preds_t = df.filter(like="Pred_CMF").loc[t]
        long_cmf = preds_t.idxmax()
        short_cmf = preds_t.idxmin()
        long_ticker = PRED_TO_INDEX[long_cmf]
        short_ticker = PRED_TO_INDEX[short_cmf]

        # Build WINDOW lookback rets up to t (inclusive)
        window_rets = rets.iloc[i - WINDOW + 1 : i + 1]  # WINDOW rows ending at t
        if window_rets[[long_ticker, short_ticker]].isna().any().any():
            continue  # skip if insufficient history for either leg

        # Sizing for day t using the window
        lw, sw = size_weights_for_day(window_rets, long_ticker, short_ticker)

        # Realized next-day portfolio return (t -> t+1)
        r_long = rets.loc[t1, long_ticker]
        r_short = rets.loc[t1, short_ticker]
        if np.isnan(r_long) or np.isnan(r_short):
            continue
        port_ret = lw * r_long + sw * r_short

        rows.append({
            "Date": t1,                     # attribute P&L to the next day
            "Long": long_ticker,
            "Short": short_ticker,
            "W_long": lw,
            "W_short": sw,
            "Ret_long": r_long,
            "Ret_short": r_short,
            "Port_ret": port_ret,
        })

    if not rows:
        raise RuntimeError("No rows produced — check input files or WINDOW size.")
    perf = pd.DataFrame(rows).set_index("Date").sort_index()

    # Equity curve
    equity = START_EQUITY * np.exp(perf["Port_ret"].cumsum())
    perf["Equity"] = equity

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perf.to_excel(OUT_CURVE, engine="openpyxl")

    # Plot
    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140)
    plt.close()

    print("[OK] Wrote:", OUT_CURVE)
    print("[OK] Saved plot:", OUT_PNG)
    print("[INFO] Final equity:", float(equity.iloc[-1]))
    print("[INFO] CAGR approx (%):", round((equity.iloc[-1]/START_EQUITY)**(252/len(perf))-1, 4)*100)
    print("[INFO] Max DD approx (%):", round(100*(equity/ equity.cummax() - 1).min(), 2))

if __name__ == "__main__":
    backtest()
