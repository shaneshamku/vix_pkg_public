# %%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

def plot_equity_curve():
    pred_file = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
    price_file = Path("vix_pkg/data/vixprices.xlsx")
    pred = pd.read_excel(pred_file, parse_dates=["Date"])
    prices = pd.read_excel(price_file, parse_dates=["Date"])
    prices = prices[["Date", "SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]].dropna()
    df = pred.merge(prices, on="Date", how="inner").set_index("Date").sort_index()
    etf_cols = ["SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]
    ret = np.log(df[etf_cols]).diff().fillna(0)

    # Generate signals for each day
    signals = []
    for i in range(20, len(df)):
        ret_window = ret.iloc[i-20:i]
        pred_today = df.filter(like="Pred_CMF").iloc[i]
        mapper = {"Pred_CMF1": "SPVXSP", "Pred_CMF2": "SPVIX2ME", "Pred_CMF3": "SPVIX3ME",
                  "Pred_CMF4": "SPVIX4ME", "Pred_CMF5": "SPVXMP", "Pred_CMF6": "SPVIX6ME"}
        long_cmf = pred_today.idxmax()
        short_cmf = pred_today.idxmin()
        long_ticker = mapper[long_cmf]
        short_ticker = mapper[short_cmf]
        vol = ret_window.std()
        wl = 1 / vol[long_ticker]
        ws = 1 / vol[short_ticker]
        scale = 1 / (wl + ws)
        long_weight = scale * wl
        short_weight = -scale * ws
        # Drawdown logic
        log_equity = (pd.Series({long_ticker: long_weight, short_ticker: short_weight}) * ret_window.iloc[-1]).cumsum().fillna(0)
        peak = log_equity.cummax()
        drawdown = (log_equity - peak) / peak
        exposure = 1.0
        if drawdown.iloc[-1] < -0.10:
            exposure = 0.5
        if drawdown.iloc[-1] < -0.20:
            exposure = 0.0
        long_weight *= exposure
        short_weight *= exposure
        weights = pd.Series(0, index=etf_cols)
        weights[long_ticker] = long_weight
        weights[short_ticker] = short_weight
        signals.append(weights)

    # Convert signals to DataFrame and align with returns
    weights_df = pd.DataFrame(signals, index=ret.index[20:])
    pnl = (weights_df.shift().fillna(0) * ret.loc[weights_df.index]).sum(axis=1)
    equity = pnl.cumsum()

    plt.figure(figsize=(10,5))
    plt.plot(equity.index, equity.values, label="Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log-Return")
    plt.title("Strategy Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_equity_curve()