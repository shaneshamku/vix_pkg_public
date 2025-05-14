from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
# import os
# from dotenv import load_dotenv
# load_dotenv()
# ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
# ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
# ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
VIX_ALPACA_API_KEY = "PKIXS8H5ETCIAG1LRAVK"
VIX_ALPACA_API_SECRET = "NYLz0oY4vsdsSRU6HfHfyOPbUcR1cOHBSp2IOcaI"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/"

# Initialize Alpaca API client
api = REST(VIX_ALPACA_API_KEY, VIX_ALPACA_API_SECRET, ALPACA_BASE_URL)

def get_account_info():
    account = api.get_account()
    return float(account.equity)  # Equivalent to Net Liquidation

def is_shortable(symbol):
    """
    Checks if a stock is shortable (easy to borrow) through Alpaca API.
    
    Parameters:
        symbol (str): The stock symbol to check
        
    Returns:
        bool: True if the stock is shortable, False otherwise
    """
    try:
        # Get asset information from Alpaca
        asset = api.get_asset(symbol)
        
        # Check if the asset is shortable (easy to borrow)
        return asset.easy_to_borrow
    except Exception as e:
        print(f"Error checking if {symbol} is shortable: {e}")
        return False


def place_orders(df):
    for _, row in df.iterrows():
        symbol = row["Symbol"]
        allocation = row["Dollar Allocation"]

        try:
            # Get the latest closing price
            bars = api.get_bars(symbol, TimeFrame.Day, limit=5).df
            if bars.empty:
                print(f"Skipping {symbol}, no price data available.")
                continue

            price = bars.iloc[-1].close
            shares = int(abs(allocation) // price)

            if shares == 0:
                print(f"Skipping {symbol}, allocation too small for even 1 share.")
                continue

            if allocation > 0:
                # Long position
                side = "buy"
            else:
                # Short position
                if not is_shortable(symbol):
                    print(f"Cannot short {symbol}, not easy to borrow.")
                    continue
                side = "sell"

            api.submit_order(
                symbol=symbol,
                qty=shares,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            print(f"{side.upper()} {shares} shares of {symbol} at approx ${price:.2f}.")

        except Exception as e:
            print(f"Error placing order for {symbol}: {e}")


def close_positions(new_portfolio_df):
    try:
        # Get all currently held positions
        positions = api.list_positions()
        new_portfolio_symbols = set(new_portfolio_df["Symbol"])  # Stocks in new portfolio

        for position in positions:
            symbol = position.symbol

            # If stock is not in the new portfolio, close it
            if symbol not in new_portfolio_symbols:
                qty = abs(int(float(position.qty)))  # Ensure quantity is positive
                side = "sell" if position.side == "long" else "buy"

                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Closing {side.upper()} order for {qty} shares of {symbol} (not in new portfolio).")

        print("Unwanted positions have been closed.")
    except Exception as e:
        print(f"Error closing positions: {e}")



def send_weekly_basket():
    pf = get_account_info()
    lookback = 22
    winners_from_low_vol, losers_from_high_vol, low_vol_from_winners, high_vol_from_losers = calculate_portfolios(lookback, pf)
    close_positions(winners_from_low_vol)
    place_orders(winners_from_low_vol, 'long')



def send_daily_orders(df):
    long_df = pd.DataFrame({
    'Symbol': df['Top_ETF'].values,
    'Dollar Allocation': df['Long Pos Notional'].values
    })

    short_df = pd.DataFrame({
        'Symbol': df['Bottom_ETF'].values,
        'Dollar Allocation': -df['Short Pos Notional'].values  # negate short position
    })

    # Combine both
    result_df = pd.concat([long_df, short_df], ignore_index=True)
    print(result_df)
    close_positions(result_df)
    place_orders(result_df)