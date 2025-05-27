# %%
import yfinance as yf
import pandas as pd
import os
from datetime import date, timedelta
import numpy as np
import warnings
warnings.simplefilter('ignore')
# %%


# %%
def pullHistoricalTickerPrice(ticker_symbol, start_date, end_date):
    """
    Pulls the close price for a list of tickers, between start_date and end_date
    
    Parameters:
        ticker_symbol (list): The list of stock symbols to fetch
        start_date (date): Start date of the close prices
        end_date (date): Last date for close prices
    Returns:
        multi_data: dataframe of tickers and closing prices from start date to end date
    """
    # pulls close price for list of ticker_symbol, between start_date and end_date
    multi_data = yf.download(ticker_symbol, start=start_date, end=end_date)['Close']
    return multi_data


def AppendNewClosePrice():
    """
    Pulls new close prices if todays date is further than the last pulled date, 
    and appends new data to the old data, printing to vix_prices
    
    Parameters:
    
    Returns:
        : 
    """

    vix_prices = pd.read_excel(r'vix_pkg/vix_pkg/data/vixprices.xlsx')
    vix_prices = vix_prices.drop(index=0).reset_index().drop(columns='index').rename(columns={vix_prices.columns[0]: 'Date'})


    # Get last date in dataframe
    last_pulled_date = pd.Timestamp(vix_prices['Date'].iloc[0]).date()

    # Compare with today's date
    if date.today() > last_pulled_date:
        tickers = ['^SPVIX2ME','^SPVIX3ME','^SPVIX4ME','^SPVIX6ME','^SPVXMP','^SPVXSP',"SPY", 'TLT', '^VIX']
        
        latest_prices = {}

        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period='1d')
                if not data.empty:
                    latest_prices[ticker] = data['Close'].iloc[0]
                else:
                    latest_prices[ticker] = None
            except Exception as e:
                latest_prices[ticker] = None
                print(f"Error fetching {ticker}: {e}")

        # Create new row with today's date and fetched prices
        new_row = {'Date': date.today()}
        for ticker, price in latest_prices.items():
            new_row[ticker] = price

        newData = pd.DataFrame([new_row])

    else:
        print(f"Data is up to date as of {last_pulled_date}")


    # reformat pulled data to properly merge with old data
    newData.columns = [x.replace("^", "") for x in newData.columns]
    newData  = newData.rename(columns={'TLT':"TLT US Equity", "VIX": "VIX Index", "SPY": "SPX"})
    
    vix_prices['Date'] = pd.to_datetime(vix_prices['Date']).dt.date
  
    # merge with previous set of vix_prices
    vix_prices = pd.concat([vix_prices, newData])
    vix_prices = vix_prices.sort_values(by='Date', ascending=False)
    vix_prices = vix_prices.reset_index(drop=True)
    vix_prices.to_excel(r'vix_pkg/vix_pkg/data/vixprices.xlsx', index=False)
    print('done appending close prices')

if __name__ == "__main__":
    AppendNewClosePrice()
    
    
    