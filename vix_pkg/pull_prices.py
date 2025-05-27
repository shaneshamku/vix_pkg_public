# %%
import yfinance as yf
import pandas as pd
import os
from datetime import date, timedelta
import numpy as np
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
    and appends new data to the old data, printing to etf_prices
    
    Parameters:
    
    Returns:
        : 
    """

    etf_prices = pd.read_excel('vix_pkg\data\etf_prices.xlsx')
    etf_prices = etf_prices.sort_values(by='Date', ascending=False)

    # check if last date in excel is up to date
    if (date.today()) > pd.Timestamp(etf_prices['Date'][0]).date():

        # pull data from last pulled date + 1, up to today
        start_date = pd.Timestamp(etf_prices['Date'][0]).date() + timedelta(days=1)
        end_date = (date.today())

        newData = pullHistoricalTickerPrice(['IAU','SVIX','SVXY','UVXY','VIXM','VXX','VXZ'], start_date, end_date)
    else:
        print(f"Data is up to date as of " + pd.Timestamp(etf_prices['Date'][0]).date())


    # reformat pulled data to properly merge with old data
    newData['^XIV'] = np.nan
    newData = newData.reset_index()
    newData.columns.name = None

    # merge with previous set of etf_prices
    etf_prices = pd.concat([etf_prices, newData])
    etf_prices = etf_prices.sort_values(by='Date', ascending=False)
    etf_prices = etf_prices.reset_index(drop=True)
    etf_prices.to_excel("vix_pkg\data\etf_prices.xlsx", index=False)
    print('done appending close prices')

if __name__ == "__main__":
    AppendNewClosePrice()
    
    
    