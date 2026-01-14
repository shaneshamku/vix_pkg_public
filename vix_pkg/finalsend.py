import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
import subprocess

# --- 1. Data Acquisition ---
def fetch_cmf_prices(tickers, out_path):
    try:
        existing = pd.read_excel(out_path, parse_dates=["Date"])
    except FileNotFoundError:
        existing = pd.DataFrame()

    # Prepare new row with today's date
    new_row = {"Date": None}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="2d")["Close"]
            if not data.empty:
                # Get last date and price
                last_date = data.index[-1].date()
                last_price = data.iloc[-1]
                new_row["Date"] = last_date  # Set date for the row
                new_row[ticker] = last_price
            else:
                new_row[ticker] = np.nan
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            new_row[ticker] = np.nan

    # Only append if this date is not already present
    if new_row["Date"] is not None:
        if not existing.empty and new_row["Date"] in existing["Date"].values:
            print("Latest date already present. Skipping append.")
            combined = existing
        else:
            combined = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        combined.to_excel(out_path, index=False)
    else:
        print("No new data to append.")

# --- 2. Feature Engineering ---
def run_feature_engineering():
    subprocess.run(["python", "vix_pkg/vix_pkg/calculate_features.py"], check=True)

# --- 3. Model Training & Prediction ---
def run_model_training():
    subprocess.run(["python", "vix_pkg/vix_pkg/copy_train_six_cmf_model.py"], check=True)

# --- 4. Signal Generation ---
def generate_signals():
    pred_file = Path("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
    price_file = Path("vix_pkg/data/vixprices.xlsx")
    pred = pd.read_excel(pred_file, parse_dates=["Date"])
    prices = pd.read_excel(price_file, parse_dates=["Date"])
    prices = prices[["Date", "SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]].dropna()
    df = pred.merge(prices, on="Date", how="inner").set_index("Date").sort_index()
    etf_cols = ["SPVXSP","SPVIX2ME","SPVIX3ME", "SPVIX4ME","SPVXMP","SPVIX6ME"]
    ret = np.log(df[etf_cols]).diff().tail(20)
    pred_today = df.filter(like="Pred_CMF").iloc[-1]
    mapper = {"Pred_CMF1": "SPVXSP", "Pred_CMF2": "SPVIX2ME", "Pred_CMF3": "SPVIX3ME",
              "Pred_CMF4": "SPVIX4ME", "Pred_CMF5": "SPVXMP", "Pred_CMF6": "SPVIX6ME"}
    long_cmf = pred_today.idxmax()
    short_cmf = pred_today.idxmin()
    long_ticker = mapper[long_cmf]
    short_ticker = mapper[short_cmf]
    vol = ret.std()
    wl = 1 / vol[long_ticker]
    ws = 1 / vol[short_ticker]
    scale = 1 / (wl + ws)
    long_weight = scale * wl
    short_weight = -scale * ws
    # Drawdown logic (simplified for last 20 days)
    log_equity = (pd.Series({long_ticker: long_weight, short_ticker: short_weight}) * ret.iloc[-1]).cumsum().fillna(0)
    peak = log_equity.cummax()
    drawdown = (log_equity - peak) / peak
    exposure = 1.0
    if drawdown.iloc[-1] < -0.10:
        exposure = 0.5
    if drawdown.iloc[-1] < -0.20:
        exposure = 0.0
    long_weight *= exposure
    short_weight *= exposure
    return {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "long": long_ticker,
        "long_weight": round(long_weight, 2),
        "long_pred": round(pred_today[long_cmf]*100, 2),
        "short": short_ticker,
        "short_weight": round(short_weight, 2),
        "short_pred": round(pred_today[short_cmf]*100, 2),
        "risk": "Normal" if exposure == 1.0 else ("Reduced" if exposure == 0.5 else "No Trade")
    }

# --- 5. Email Notification ---
def send_signal_email(signal, recipients):
    body = f"""
Subject: VIX CMF Strategy Signals for {signal['date']}

Long: {signal['long']}, Size: {signal['long_weight']}, Predicted Return: {signal['long_pred']}%
Short: {signal['short']}, Size: {signal['short_weight']}, Predicted Return: {signal['short_pred']}%

Risk Control: {signal['risk']}
Instructions: Submit Market-On-Close order for above tickers and sizes.
"""
    msg = MIMEText(body)
    msg["Subject"] = f"VIX CMF Strategy Signals for {signal['date']}"
    msg["From"] = "your_email@example.com"
    msg["To"] = ", ".join(recipients)
    with smtplib.SMTP("smtp.example.com") as server:
        server.login("your_email@example.com", "your_password")
        server.sendmail(msg["From"], recipients, msg.as_string())

# --- Main Routine ---
def main():
    tickers = ["^SPVXSP","^SPVIX2ME","^SPVIX3ME", "^SPVIX4ME","^SPVXMP","^SPVIX6ME","^SPX","TLT","^VIX"]
    fetch_cmf_prices(tickers, "vix_pkg/data/vixprices.xlsx")
    # run_feature_engineering()
    # run_model_training()
    # signal = generate_signals()
    # send_signal_email(signal, ["team@example.com"])

if __name__ == "__main__":
    main()