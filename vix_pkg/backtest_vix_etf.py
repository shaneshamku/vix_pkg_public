# backtest_vix_etf.py  ----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# load & merge
PRED_FILE  = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
PRICE_FILE = Path("vix_pkg/data/vixprices.xlsx")

pred   = pd.read_excel(PRED_FILE,  parse_dates=["Date"])
prices = pd.read_excel(PRICE_FILE, parse_dates=["Date"])

prices = prices[["Date", "SPVXSP","SPVIX2ME","SPVIX3ME",
                 "SPVIX4ME","SPVXMP","SPVIX6ME"]].dropna()

df = (pred.merge(prices, on="Date", how="inner")
          .set_index("Date").sort_index())

#  daily ETF returns 
etf_cols = ["SPVXSP","SPVIX2ME","SPVIX3ME",
            "SPVIX4ME","SPVXMP","SPVIX6ME"]

ret   = np.log(df[etf_cols]).diff().loc["2010-01-04":]   # log-returns
pred  = df.filter(like="Pred_CMF").loc[ret.index]

# -------------------------------------------------------------------
# 3.  ---  generate signals -----------------------------------------
mapper = {"Pred_CMF1": "SPVXSP",
          "Pred_CMF2": "SPVIX2ME",
          "Pred_CMF3": "SPVIX3ME",
          "Pred_CMF4": "SPVIX4ME",
          "Pred_CMF5": "SPVXMP",
          "Pred_CMF6": "SPVIX6ME"}

long_etf  = pred.idxmax(axis=1).map(mapper)
short_etf = pred.idxmin(axis=1).map(mapper)

signals = pd.DataFrame({"Long": long_etf, "Short": short_etf},
                       index=ret.index)

# -------------------------------------------------------------------
# 4.  ---  position weights -----------------------------------------
vol     = ret.rolling(20).std()
inv_vol = (1 / vol).clip(upper=10)

def weight_row(row):
    wl = inv_vol.at[row.name, row["Long"]]
    ws = inv_vol.at[row.name, row["Short"]]
    scale = 1/(wl+ws)
    return pd.Series({row["Long"]:  scale*wl,
                      row["Short"]: -scale*ws})

weights = (signals.apply(weight_row, axis=1)
                  .reindex(columns=etf_cols).fillna(0))

# -------------------------------------------------------------------
# 5.  ---  draw-down exposure cut -----------------------------------
log_equity = (weights.shift()*ret).sum(axis=1).cumsum().fillna(0)
peak       = log_equity.cummax()
drawdown   = (log_equity-peak)/peak
exposure   = np.where(drawdown < -0.10, 0.50, 1.00)
weights    = weights.mul(exposure, axis=0)

# -------------------------------------------------------------------
# 6.  ---  back-test -------------------------------------------------
daily_pnl_log = (weights.shift()*ret).sum(axis=1)        # one-day lag
log_equity    = daily_pnl_log.cumsum()
# *****  add these four lines ↓  *****
start_capital = 100_000.0                                # initial NAV
NAV           = start_capital * np.exp(log_equity)       # convert log-P&L
nav           = NAV                                      # alias the same series
out           = pd.DataFrame({"NAV": NAV})               # if you still plot later
# ****************************************
# -------------------------------------------------------------------------
# 6)  Collect statistics in one place  ➜  summary  (a Pandas Series)
# -------------------------------------------------------------------------
summary = pd.Series({
    "Start Capital"      : start_capital,
    "End Capital"        : NAV.iloc[-1],
    "Cumulative Return"  : NAV.iloc[-1] / start_capital - 1,
    "CAGR"               : (NAV.iloc[-1] / start_capital) ** (252/len(nav)) - 1,
    "Ann. Volatility"    : nav.pct_change().std() * np.sqrt(252),
    "Ann. Sharpe"        : (summary["CAGR"] / summary["Ann. Volatility"])
                           if "summary" in locals() else 0,   # set just below
    "Max Draw-down"      : (NAV / NAV.cummax() - 1).min(),
    "Trade Days"         : len(nav)
})

# fill in the Sharpe now that everything else exists
summary["Ann. Sharpe"] = summary["CAGR"] / summary["Ann. Volatility"]

# -------------------------------------------------------------------------
# 7)  Pretty print
# -------------------------------------------------------------------------
def pretty(series):
    """Convert large/small floats to comma-separated or % strings."""
    out = series.copy()
    out["Start Capital"] = f"${series['Start Capital']:,.0f}"
    out["End Capital"]   = f"${series['End Capital']:,.0f}"
    out["Cumulative Return"] = f"{series['Cumulative Return']*100:,.2f}%"
    out["CAGR"]             = f"{series['CAGR']*100:,.2f}%"
    out["Ann. Volatility"]  = f"{series['Ann. Volatility']*100:,.2f}%"
    out["Ann. Sharpe"]      = f"{series['Ann. Sharpe']:.2f}"
    out["Max Draw-down"]    = f"{series['Max Draw-down']*100:,.2f}%"
    out["Trade Days"]       = f"{series['Trade Days']:,.0f}"
    return out

print("\n=== Strategy Performance Summary ===")
print(pretty(summary))

# # -------------------------------------------------------------------
# # 8.  ---  performance report ---------------------------------------
# def perf_stats(nav, freq=252):
#     period_yrs = (nav.index[-1]-nav.index[0]).days/365.25
#     daily_ret  = np.log(nav/nav.shift()).dropna()

#     stats = {
#         "Start Capital":      nav.iloc[0],
#         "End Capital":        nav.iloc[-1],
#         "Cumulative Return":  nav.iloc[-1]/nav.iloc[0]-1,
#         "CAGR":               (nav.iloc[-1]/nav.iloc[0])**(1/period_yrs)-1,
#         "Ann. Volatility":    daily_ret.std()*np.sqrt(freq),
#         "Ann. Sharpe":        daily_ret.mean()/daily_ret.std()*np.sqrt(freq),
#         "Max Draw-down":      (nav/nav.cummax()-1).min(),
#         "Trade Days":         len(nav)
#     }
#     return pd.Series(stats).round(4)

# print("\n=== Strategy Performance Summary ===")
# print(perf_stats(out["NAV"]).to_string())

# print(f"""
# === Strategy Performance Summary ===
# Start Capital      ${summary['Start Capital']:,.0f}
# End Capital        ${summary['End Capital']:,.0f}
# Cumulative Return   {summary['Cumulative Return']*100:,.2f} %
# CAGR                {summary['CAGR']*100:,.2f} %
# Annual Volatility   {summary['Ann. Volatility']*100:,.2f} %
# Annual Sharpe       {summary['Ann. Sharpe']:.2f}
# Max Draw-down       {summary['Max Draw-down']*100:,.2f} %
# Trade Days          {summary['Trade Days']:,.0f}
# """)