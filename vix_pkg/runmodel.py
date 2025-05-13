from copy_train_six_cmf_model import train_six_cmf_models
from etf_predicted_returns import *

if __name__ == "__main__":
    final_preds = train_six_cmf_models("vix_pkg/data/vix_features_calculated.xlsx")
    print("Final shape:", final_preds.shape)
    print(final_preds.head(10))
    
    cmf_predictions = preprocess_predictions()

    #SVIX is -1x Short VIX short-term futures contracts CMF 1-2
    #Need to use same prediction model to get next day returns for gold etf
    #VIXM is 1x Long VIX mid-term futures contracts CMF 4-7
    vix_etf_predictions, start_date, end_date = predict_vix_etf_next_day_returns(cmf_predictions)
    start_date = pd.to_datetime(start_date).tz_localize("America/New_York")
    end_date = pd.to_datetime(end_date).tz_localize("America/New_York") 

    #gold_prices = get_gold_prices_spdr(start_date, end_date)

    # vix_etf_predictions = pd.merge(vix_etf_predictions, gold_prices[["Date", " GLD Close"]], on="Date", how="left")
    etf_prices = read_etf_prices()
    etf_prices = etf_prices.set_index("Date")
    etf_prices = etf_prices.sort_index()
    simple_returns = etf_prices.pct_change()
    etf_prices = np.log(etf_prices/etf_prices.shift(1))
    etf_prices = etf_prices.reset_index() # log returns
    etf_prices = etf_prices.rename(columns={"SVIX": "Act_SVIX", "UVXY": "Act_UVXY", "VIXM": "Act_VIXM"}).dropna()

    vix_etf_predictions = pd.merge(vix_etf_predictions, etf_prices, on="Date", how="left").dropna()

    vix_etf_predictions = vix_etf_predictions[['Date', 'Act_SVIX','Pred_SVIX', 'Act_UVXY','Pred_UVXY', 'Act_VIXM','Pred_VIXM', 'SPX']]
    vix_etf_predictions.to_excel("vix_pkg/data/all_etfs_oos_predictions.xlsx")
    
    etf_predictions = pd.read_excel("vix_pkg/data/all_etfs_oos_predictions.xlsx")
    

    etf_mapping = {
        'Pred_SVIX': 'Act_SVIX',
        'Pred_UVXY': 'Act_UVXY',
        'Pred_VIXM': 'Act_VIXM'
    }

    # Step 1: Identify top and bottom predicted ETFs
    pred_cols = list(etf_mapping.keys())

    # Get the ETF with the highest predicted return
    etf_predictions['Top_ETF'] = etf_predictions[pred_cols].idxmax(axis=1)

    # Get the ETF with the lowest predicted return
    etf_predictions['Bottom_ETF'] = etf_predictions[pred_cols].idxmin(axis=1)

    # Step 2: Compute the actual return for each (long the top, short the bottom)
    def calc_strategy_return(row):
        long_return = row[etf_mapping[row['Top_ETF']]]
        short_return = row[etf_mapping[row['Bottom_ETF']]]
        return long_return - short_return

    etf_predictions['Return'] = etf_predictions.apply(calc_strategy_return, axis=1)
    

    
    
    etf_predictions = etf_predictions.set_index("Date")
    
    capital = 100000
    
    
    etf_predictions = volatility_weighted_position_sizing(etf_predictions, capital)
    etf_predictions = apply_scaled_drawdown_constraint(etf_predictions, max_drawdown_threshold=-0.2)
    
    sizing = etf_predictions[['Top_ETF','Long Pos Notional', 'Bottom_ETF', 'Short Pos Notional']]
    
    daily_sizing = sizing.tail(1)
    print(daily_sizing)
    
    
    