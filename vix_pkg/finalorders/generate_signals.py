# vix_pkg/finalorders/generate_signals.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from vix_weights import weights_for_index, build_vx_strip

PORTFOLIO_NOTIONAL = 250_000  # starting portfolio size

# --- NEW: live prediction helper ---
# This expects vix_pkg/finalorders/predict_today.py (from earlier instructions).
# If it's missing, we'll fall back to the old OOS file.
try:
    from predict_today import predict_today  # returns a 1-row DF: Date=t, Pred_CMF1..6 for t+1
    _HAS_LIVE = True
except Exception:
    _HAS_LIVE = False


def _size_and_exposure(
    prices_df: pd.DataFrame,
    long_ticker: str,
    short_ticker: str,
    window: int = 20,
    start_live_date: str | None = None,  # e.g., "2025-10-01"
):
    """
    Vol-parity sizing on the last `window` days, then a drawdown throttle
    computed from a *time series* of portfolio returns.

    If start_live_date is provided, the drawdown is computed from that date forward.
    """
    idx_cols = ["SPVXSP", "SPVIX2ME", "SPVIX3ME", "SPVIX4ME", "SPVXMP", "SPVIX6ME"]
    prices_clean = prices_df.dropna(subset=idx_cols).copy().sort_values("Date")
    if len(prices_clean) < window + 1:
        raise ValueError(f"Not enough history for sizing window={window} (have {len(prices_clean)})")

    # full return series on the indices
    rets = np.log(prices_clean[idx_cols]).diff()

    # last `window` days up to the latest available date for vol sizing
    rets_win = rets.iloc[-window:]
    vol = rets_win[[long_ticker, short_ticker]].std()
    wl = 1.0 / vol[long_ticker]
    ws = 1.0 / vol[short_ticker]
    scale = 1.0 / (wl + ws)
    long_weight = float(scale * wl)
    short_weight = float(-scale * ws)

    # ----- Correct drawdown over time -----
    # Use a lookback for DD. If you're going live today, supply start_live_date to reset baseline.
    if start_live_date is not None:
        mask = prices_clean["Date"] >= pd.to_datetime(start_live_date)
    else:
        # last ~2 months (~42 trading days) for a smoother throttle
        mask = rets.index >= (rets.index.max() - pd.tseries.offsets.BDay(42))

    rets_dd = rets.loc[mask, [long_ticker, short_ticker]].dropna()
    if len(rets_dd) == 0:
        # fallback: use window if filtering removed everything
        rets_dd = rets_win[[long_ticker, short_ticker]].dropna()

    port_log_ret = long_weight * rets_dd[long_ticker] + short_weight * rets_dd[short_ticker]
    cum_log = port_log_ret.cumsum()
    # Convert to equity and compute percentage DD
    equity = np.exp(cum_log)
    dd_pct = equity / equity.cummax() - 1.0
    last_dd = float(dd_pct.iloc[-1]) if not dd_pct.empty else 0.0

    exposure = 1.0
    if last_dd <= -0.20:
        exposure = 0.0
    elif last_dd <= -0.10:
        exposure = 0.5

    print("[SIZING] vol(win):", vol.to_dict())
    print("[SIZING] wl, ws, scale =>", wl, ws, scale)
    print("[SIZING] pre-exposure long_weight=", long_weight, " short_weight=", short_weight)
    print("[DRAWDOWN] window_len=", len(rets_dd), " last_dd=", last_dd, " exposure=", exposure)
    return long_weight * exposure, short_weight * exposure, exposure, last_dd


def generate_signals():
    price_file = Path("vix_pkg/data/vixprices.xlsx")
    prices = pd.read_excel(price_file, parse_dates=["Date"])

    # Ensure we have the six index columns
    idx_cols = ["SPVXSP", "SPVIX2ME", "SPVIX3ME", "SPVIX4ME", "SPVXMP", "SPVIX6ME"]
    prices = prices[["Date"] + idx_cols].dropna().sort_values("Date")
    if prices.empty:
        raise RuntimeError("No prices available.")

    # --- NEW: live predictions for Date=t (preds refer to t+1) ---
    mapper = {
        "Pred_CMF1": "SPVXSP",
        "Pred_CMF2": "SPVIX2ME",
        "Pred_CMF3": "SPVIX3ME",
        "Pred_CMF4": "SPVIX4ME",
        "Pred_CMF5": "SPVXMP",
        "Pred_CMF6": "SPVIX6ME",
    }

    if _HAS_LIVE:
        # Live: compute predictions on the latest features row (Date=t)
        live = predict_today()  # one row: Date=t, Pred_CMF*
        live["Date"] = pd.to_datetime(live["Date"])
        pred_today = live.filter(like="Pred_CMF").iloc[0]
        trade_date = live["Date"].iloc[0].date()  # this is t (you will trade MOC on t)
    else:
        # --- OLD APPROACH (commented below) used OOS file's last row.
        # We only keep it as a fallback if predict_today.py isn't present.
        pred_file = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
        pred = pd.read_excel(pred_file, parse_dates=["Date"])
        pred = pred.sort_values("Date")
        if pred.empty:
            raise RuntimeError("No predictions available and live helper missing.")
        pred_today = pred.filter(like="Pred_CMF").iloc[-1]
        trade_date = pred["Date"].iloc[-1].date()
        print("[WARN] Using last OOS row (likely t-1). Add predict_today.py for same-day predictions.")

    print("\n========== SIGNAL STAGE ==========")
    print(f"[DATES] trade_date(t)={trade_date}  latest_price_date={prices['Date'].max().date()}")
    print("[RAW PRED VECTOR]\n", pred_today.to_string(float_format=lambda x: f"{x:+.6f}"))

    # Choose long/short indices by max/min predicted next-day return
    long_cmf = pred_today.idxmax()
    short_cmf = pred_today.idxmin()
    long_ticker = mapper[long_cmf]
    short_ticker = mapper[short_cmf]
    print(f"[SELECTION] long={long_ticker}({long_cmf})  short={short_ticker}({short_cmf})")

    # Sizing using last 20D on index returns
    # long_weight, short_weight, exposure = _size_and_exposure(prices, long_ticker, short_ticker, window=20)
    long_weight, short_weight, exposure, last_dd = _size_and_exposure(prices, long_ticker, short_ticker, window=20, start_live_date="2025-10-15")
    
    print("[WEIGHTS] long=", long_ticker, " w=", long_weight, " | short=", short_ticker, " w=", short_weight)
    print("[WEIGHTS ROUNDED->EMAIL] long_w=", round(float(long_weight), 2), " short_w=", round(float(short_weight), 2))
   
    # Build signal payload
    signal = {
        "date": str(trade_date),  # Date=t (you will trade at MOC on this date)
        "long": long_ticker,
        "long_weight": round(float(long_weight), 3),
        "long_pred": round(float(pred_today[long_cmf]) * 100, 3),    # in %
        "short": short_ticker,
        "short_weight": round(float(short_weight), 3),
        "short_pred": round(float(pred_today[short_cmf]) * 100, 3),  # in %
        "risk": "Normal" if exposure == 1.0 else ("Reduced" if exposure == 0.5 else "No Trade"),
    }

    assert pd.Timestamp(trade_date) == prices["Date"].max().normalize(), \
    f"Date misalign: preds(t)={trade_date} vs prices_last={prices['Date'].max().date()}"

    # --- Dollar allocations by VX leg (no futures prices needed) ---
    today = date.today()
    strip = build_vx_strip(today, n_months=8)

    print("DEBUG strip:", [c.symbol for c in strip])
    print("DEBUG SPVXSP:", weights_for_index(today, strip, "SPVXSP"))
    print("DEBUG SPVIX6ME:", weights_for_index(today, strip, "SPVIX6ME"))

    per_contract_weight = {}
    for idx_name, idx_w in {
        signal["long"]: signal["long_weight"],
        signal["short"]: signal["short_weight"],
    }.items():
        for sym, w in weights_for_index(today, strip, idx_name).items():
            per_contract_weight[sym] = per_contract_weight.get(sym, 0.0) + idx_w * w

    vx_dollar_allocations = {
        sym: round(PORTFOLIO_NOTIONAL * w, 2)
        for sym, w in per_contract_weight.items()
        if abs(PORTFOLIO_NOTIONAL * w) >= 1  # drop dust
    }

    signal["portfolio_notional"] = PORTFOLIO_NOTIONAL
    signal["vx_dollar_allocations"] = dict(sorted(vx_dollar_allocations.items()))
    return signal


# -------------------------------
# OLD CODE (kept for easy revert):
# -------------------------------
#
# def generate_signals():
#     pred_file = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
#     price_file = Path("vix_pkg/data/vixprices.xlsx")
#     pred = pd.read_excel(pred_file, parse_dates=["Date"])
#     prices = pd.read_excel(price_file, parse_dates=["Date"])
#     prices = prices[["Date", "SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]].dropna()
#     df = pred.merge(prices, on="Date", how="inner").set_index("Date").sort_index()
#     etf_cols = ["SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]
#     ret = np.log(df[etf_cols]).diff().tail(20)
#     pred_today = df.filter(like="Pred_CMF").iloc[-1]
#     mapper = {"Pred_CMF1": "SPVXSP", "Pred_CMF2": "SPVIX2ME", "Pred_CMF3": "SPVIX3ME",
#               "Pred_CMF4": "SPVIX4ME", "Pred_CMF5": "SPVXMP", "Pred_CMF6": "SPVIX6ME"}
#     long_cmf = pred_today.idxmax()
#     short_cmf = pred_today.idxmin()
#     long_ticker = mapper[long_cmf]
#     short_ticker = mapper[short_cmf]
#     vol = ret.std()
#     wl = 1 / vol[long_ticker]
#     ws = 1 / vol[short_ticker]
#     scale = 1 / (wl + ws)
#     long_weight = scale * wl
#     short_weight = -scale * ws
#     log_equity = (pd.Series({long_ticker: long_weight, short_ticker: short_weight}) * ret.iloc[-1]).cumsum().fillna(0)
#     peak = log_equity.cummax()
#     drawdown = (log_equity - peak) / peak
#     exposure = 1.0
#     if drawdown.iloc[-1] < -0.10:
#         exposure = 0.5
#     if drawdown.iloc[-1] < -0.20:
#         exposure = 0.0
#     long_weight *= exposure
#     short_weight *= exposure
#     signal = {
#         "date": df.index[-1].strftime("%Y-%m-%d"),
#         "long": long_ticker,
#         "long_weight": round(float(long_weight), 2),
#         "long_pred": round(float(pred_today[long_cmf])*100, 2),
#         "short": short_ticker,
#         "short_weight": round(float(short_weight), 2),
#         "short_pred": round(float(pred_today[short_cmf])*100, 2),
#         "risk": "Normal" if exposure == 1.0 else ("Reduced" if exposure == 0.5 else "No Trade"),
#     }
#     today = date.today()
#     strip = build_vx_strip(today, n_months=8)
#     per_contract_weight = {}
#     for idx_name, idx_w in {
#         signal["long"]: signal["long_weight"],
#         signal["short"]: signal["short_weight"],
#     }.items():
#         for sym, w in weights_for_index(today, strip, idx_name).items():
#             per_contract_weight[sym] = per_contract_weight.get(sym, 0.0) + idx_w * w
#     vx_dollar_allocations = {
#         sym: round(PORTFOLIO_NOTIONAL * w, 2)
#         for sym, w in per_contract_weight.items()
#         if abs(PORTFOLIO_NOTIONAL * w) >= 1
#     }
#     signal["portfolio_notional"] = PORTFOLIO_NOTIONAL
#     signal["vx_dollar_allocations"] = dict(sorted(vx_dollar_allocations.items()))
#     return signal
#
# if __name__ == "__main__":
#     print(generate_signals())

if __name__ == "__main__":
    from pprint import pprint
    pprint(generate_signals())