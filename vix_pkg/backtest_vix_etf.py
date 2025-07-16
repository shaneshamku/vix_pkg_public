# backtest_vix_etf.py  ----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------------------
# 🆕  Draw-down–aware exposure scaling  (linear taper to 0 %)
# -------------------------------------------------------------------
def apply_scaled_drawdown_constraint(weights: pd.DataFrame,
                                     ret: pd.DataFrame,
                                     *,
                                     full_risk_drawdown: float = -0.05,
                                     max_drawdown_threshold: float = -0.20
                                    ) -> pd.DataFrame:
    """
    Reduce gross exposure as the strategy falls into draw-down.

    • draw-down = cumulative log-equity – running peak, expressed in % of peak  
    • if DD ≥ full_risk_drawdown (e.g. –5 %)    → keep 100 % weights  
    • if DD ≤ max_drawdown_threshold (e.g. –20 %) → cut to   0 % weights  
    • between the two thresholds we **linearly** taper exposure.
    """

    # 1) running equity curve (log-P&L, shifted one day to avoid look-ahead)
    log_equity = (weights.shift() * ret).sum(axis=1).cumsum().fillna(0.0)

    # 2) draw-down series
    peak     = log_equity.cummax()
    drawdown = (log_equity - peak) / peak        # negative numbers

    # 3) scale factor ∈ [0, 1]   (vectorised, no loop)
    scale = np.where(
        drawdown >= full_risk_drawdown,               1.0,
        np.where(drawdown <= max_drawdown_threshold,  0.0,
                 (drawdown - max_drawdown_threshold) /
                 (full_risk_drawdown - max_drawdown_threshold))
    )

    # 4) apply row-wise
    return weights.mul(scale, axis=0)



# load & merge
PRED_FILE  = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
PRICE_FILE = Path("vix_pkg/data/vixprices.xlsx")

pred   = pd.read_excel(PRED_FILE,  parse_dates=["Date"])
prices = pd.read_excel(PRICE_FILE, parse_dates=["Date"])

prices = prices[["Date", "SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]].dropna()

df = (pred.merge(prices, on="Date", how="inner").set_index("Date").sort_index())

#  daily ETF returns 
etf_cols = ["SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]

ret   = np.log(df[etf_cols]).diff().loc["2010-01-04":]   # log-returns
pred  = df.filter(like="Pred_CMF").loc[ret.index]

# generate signals
mapper = {"Pred_CMF1": "SPVXSP",
          "Pred_CMF2": "SPVIX2ME",
          "Pred_CMF3": "SPVIX3ME",
          "Pred_CMF4": "SPVIX4ME",
          "Pred_CMF5": "SPVXMP",
          "Pred_CMF6": "SPVIX6ME"}

long_etf  = pred.idxmax(axis=1).map(mapper)
short_etf = pred.idxmin(axis=1).map(mapper)

signals = pd.DataFrame({"Long": long_etf, "Short": short_etf}, index=ret.index)

# position weights
vol = ret.rolling(20).std()
inv_vol = (1 / vol).clip(upper=10)

# assigns weights to each long/short position for every date
def weight_row(row):
    wl = inv_vol.at[row.name, row["Long"]]
    ws = inv_vol.at[row.name, row["Short"]]
    scale = 1/(wl+ws)
    return pd.Series({row["Long"]:  scale*wl,
                      row["Short"]: -scale*ws})

weights = (signals.apply(weight_row, axis=1)
                  .reindex(columns=etf_cols).fillna(0))





# draw-down exposure cut 
#  yesterdays weights x todays log-ret
log_equity = (weights.shift()*ret).sum(axis=1).cumsum().fillna(0)
peak       = log_equity.cummax()
drawdown   = (log_equity-peak)/peak
exposure   = np.where(drawdown < -0.10, 0.50, 1.00)
weights    = weights.mul(exposure, axis=0)

# weights = apply_scaled_drawdown_constraint(weights, ret,
#                                            full_risk_drawdown=-0.05,
#                                            max_drawdown_threshold=-0.20)

# back-test 
daily_pnl_log = (weights.shift()*ret).sum(axis=1)        # one-day lag
log_equity    = daily_pnl_log.cumsum()
start_capital = 100000.0                                # initial NAV
NAV           = start_capital * np.exp(log_equity)       # convert log-P&L
nav           = NAV                                      # alias the same series
out           = pd.DataFrame({"NAV": NAV})               # if you still plot later


# Collect statistics in one place  ➜  summary  (a Pandas Series)
cagr  = (NAV.iloc[-1] / start_capital) ** (252/len(nav)) - 1
vol   = nav.pct_change().std() * np.sqrt(252)
summary = pd.Series({
    "Start Capital"     : start_capital,
    "End Capital"       : NAV.iloc[-1],
    "Cumulative Return" : NAV.iloc[-1] / start_capital - 1,
    "CAGR"              : cagr,
    "Ann. Volatility"   : vol,
    "Ann. Sharpe"       : cagr/vol if vol else np.nan,
    "Max Draw-down"     : (NAV / NAV.cummax() - 1).min(),
    "Trade Days"        : len(nav)
})
# Sharpe
summary["Ann. Sharpe"] = summary["CAGR"] / summary["Ann. Volatility"]

# print stats
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

plt.figure(figsize=(10, 5))
plt.plot(log_equity, linewidth=1.2)
plt.title("Cumulative Log-Return of Long/Short VIX ETF Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative log-return")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n=== Strategy Performance Summary ===")
print(pretty(summary))

# save position weights to excel
weights_to_save = weights.copy()
weights_to_save.index.name = "Date"          
outfile = Path("vix_pkg/data/index_position_weights.xlsx")
weights_to_save.to_excel(outfile)
print(f"Position‐weight file written to: {outfile}")

