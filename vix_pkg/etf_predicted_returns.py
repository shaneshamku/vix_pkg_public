import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
# Initialize the Alpaca client (replace with your own keys)


def preprocess_predictions():

    cmf_predictions = pd.read_excel("vix_pkg/data/all_cmfs_oos_predictions.xlsx")
    #cmf_predictions = cmf_predictions[["Date", 'Pred_CMF1', 'Pred_CMF2', 'Pred_CMF3', 'Pred_CMF4', 'Pred_CMF5', 'Pred_CMF6', 'SPX']]
    cmf_predictions = cmf_predictions.sort_values("Date")
    return cmf_predictions    

def predict_vix_etf_next_day_returns(cmf_predictions):
    svix_weights = {
        'month 1': 0.6364, # average these out later to get better estimate of weights
        'month 2': 0.3636
    }
    uvxy_weights = {
        'month 1': 0.5714,
        'month 2': 0.4286
    }
    vixm_weights = { # should technically be 4-7 but we only have 4-6
        'month 4': 0.1905,
        'month 5': 0.3333,
        'month 6': 0.3333
    }
    
    cmf_predictions['Pred_SVIX'] = -1 * (cmf_predictions['Pred_CMF1'] * svix_weights['month 1'] + cmf_predictions['Pred_CMF2'] * svix_weights['month 2'])
    cmf_predictions['Pred_UVXY'] = 1.5 * (cmf_predictions['Pred_CMF1'] * uvxy_weights['month 1'] + cmf_predictions['Pred_CMF2'] * uvxy_weights['month 2'])
    cmf_predictions['Pred_VIXM'] = 1 * (cmf_predictions['Pred_CMF4'] * vixm_weights['month 4'] + cmf_predictions['Pred_CMF5'] * vixm_weights['month 5'] + cmf_predictions['Pred_CMF6'] * vixm_weights['month 6'])
    return cmf_predictions, cmf_predictions.loc[0, "Date"], cmf_predictions.loc[len(cmf_predictions)-1, "Date"]


def read_etf_prices():
    etf_prices = pd.read_excel('vix_pkg/data/etf_prices.xlsx').drop(columns=['^XIV', 'SVXY', 'VXX', 'VXZ'])
    return etf_prices

def print_performance_metrics(results):
    """Calculates and prints performance metrics"""
    returns = results['Adjusted_Return']
    
    # Annual metrics
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative / rolling_max - 1)
    max_drawdown = drawdowns.min() * 100
    
    # Win rate
    win_rate = (returns > 0).mean() * 100
    
    
    # Calmar ratio
    calmar_ratio = annual_return / (abs(max_drawdown/100)) if max_drawdown != 0 else 0
    
    # Positive/negative days stats
    pos_ret_mean = returns[returns > 0].mean()
    neg_ret_mean = returns[returns < 0].mean()
    gain_loss_ratio = abs(pos_ret_mean / neg_ret_mean) if neg_ret_mean != 0 else float('inf')
    
    # Final portfolio value (starting with $100,000 as done for our L/S)
    initial_capital = 100000
    final_value = initial_capital * cumulative.iloc[-1]
    
    # Print summary
    print("=" * 50)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {cumulative.iloc[-1] - 1:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Gain/Loss Ratio: {gain_loss_ratio:.2f}")
    print("=" * 50)

def plot_etfs_vs_vix(results, df, initial_capital=100000):
    results.index = pd.to_datetime(results.index)
    df.index = pd.to_datetime(df.index)
    """Plot CMVO vs S&P 500 benchmark with drawdown"""
    # Calculate CMVO portfolio value
    cmvo_cum_return = (1 + results['Adjusted_Return']).cumprod()
    cmvo_portfolio = initial_capital * cmvo_cum_return
    
    # Checking if SPX column exists in the dataframe
    if 'VIX' not in df.columns:
        print("SPX column not found in the data. Plotting only CMVO performance.")
        # Create figure with two subplots (equity curve and drawdown)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Upper plot - CMVO equity curve
        ax1.plot(cmvo_portfolio.index, cmvo_portfolio.values, label="Long Short ETF Portfolio", color="darkblue", lw=2)
        ax1.set_title("Long Short ETF Strategy Performance", fontsize=16)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, ls="--", alpha=0.7)
        
        # Lower plot - drawdown chart
        rolling_max = cmvo_portfolio.cummax()
        drawdown = (cmvo_portfolio / rolling_max - 1) * 100
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='darkred', alpha=0.3)
        ax2.set_title("Long Short ETFs vs VIX Drawdown (%))", fontsize=14)
        ax2.set_xlabel("Date", fontsize=14)
        ax2.set_ylabel("Drawdown %", fontsize=14)
        ax2.grid(True, ls="--", alpha=0.7)
        
    else:
        # Extract S&P 500 data
        spx_data = df[['VIX']].copy().loc[results.index[0]:results.index[-1]]
        
        # Calculate returns and portfolio value for VIX benchmark
        spx_data['daily_return'] = spx_data['VIX'].pct_change().fillna(0)
        spx_data['VIX_portfolio'] = initial_capital * (1 + spx_data['daily_return']).cumprod()
        
        # Create figure with two subplots (equity curve and drawdown)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Upper plot - compare equity curves
        ax1.plot(cmvo_portfolio.index, cmvo_portfolio.values, label="Long Short ETFs Portfolio", color="darkblue", lw=2)
        ax1.plot(spx_data.index, spx_data['VIX_portfolio'], label="VIX Benchmark", color="green", lw=2, ls="--")
        
        ax1.set_title("ETFs vs VIX Benchmark", fontsize=16)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, ls="--", alpha=0.7)
        
        # Lower plot - drawdown chart
        rolling_max = cmvo_portfolio.cummax()
        drawdown = (cmvo_portfolio / rolling_max - 1) * 100
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='darkred', alpha=0.3)
        ax2.set_title("Long Short ETFs vs VIX Drawdown (%)", fontsize=14)
        ax2.set_xlabel("Date", fontsize=14)
        ax2.set_ylabel("Drawdown %", fontsize=14)
        ax2.grid(True, ls="--", alpha=0.7)
        
        # Calculate and display correlation and beta
        common_idx = results.index.intersection(spx_data.index)
        if len(common_idx) > 0:
            cmvo_returns = results.loc[common_idx, 'Adjusted_Return']
            spx_returns = spx_data.loc[common_idx, 'daily_return']
            
            correlation = cmvo_returns.corr(spx_returns)
            cov_matrix = np.cov(cmvo_returns, spx_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
            
            text_info = f"Correlation: {correlation:.2f}\nBeta: {beta:.2f}"
            ax1.annotate(text_info, xy=(0.02, 0.02), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=10)
    
    plt.tight_layout()
    plt.savefig('charts/etfs_vs_vix.png', dpi=300)
    plt.show()

def volatility_weighted_position_sizing(df, capital):
    lookback = 10  # rolling window
    vol = df[['Act_SVIX', 'Act_UVXY', 'Act_VIXM']].rolling(lookback, closed='left').std()

    # Inverse volatility weights (normalized)
    inv_vol = 1 / vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Use weights for top/bottom ETF picks
    df['Long_Weight'] = df.index.to_series().map(lambda d: weights.loc[d, df.loc[d, 'Top_ETF'].replace("Pred_", "Act_")])
    df['Short_Weight'] = df.index.to_series().map(lambda d: weights.loc[d, df.loc[d, 'Bottom_ETF'].replace("Pred_", "Act_")])
    def get_actual_return(row, etf_col_prefix='Act_'):
        long_ret = row[etf_col_prefix + row['Top_ETF'].replace('Pred_', '')]
        short_ret = row[etf_col_prefix + row['Bottom_ETF'].replace('Pred_', '')]
        return row['Long_Weight'] * long_ret - row['Short_Weight'] * short_ret
    
    def get_return_on_notional(row, etf_col_prefix='Act_'):
        long_ret = row[etf_col_prefix + row['Top_ETF'].replace('Pred_', '')]
        short_ret = row[etf_col_prefix + row['Bottom_ETF'].replace('Pred_', '')]
        return row['Long Pos Notional'] * long_ret - (row['Short Pos Notional'] * short_ret)
    
    df['Long Pos Notional'] = (capital/2)*df['Long_Weight']
    df['Short Pos Notional'] = (capital/2)*df['Short_Weight']
    
    df['Vol_Adjusted_Return'] = df.apply(get_actual_return, axis=1)
    # df['$ Return On Position'] = df.apply(get_return_on_notional, axis=1)
    # df['Real Return on Position'] = ((df['$ Return On Position']+df['Long Pos Notional'] + df['Short Pos Notional'])/(df['Long Pos Notional'] + df['Short Pos Notional']))-1
    return df

def apply_scaled_drawdown_constraint(df, max_drawdown_threshold=-0.2, full_risk_drawdown=-0.05):
    """
    Scales return exposure based on drawdown:
    - Full exposure when drawdown > full_risk_drawdown (e.g. -5%)
    - Linearly reduces exposure as drawdown increases
    - Zero exposure when drawdown <= max_drawdown_threshold (e.g. -20%)
    """
    df = df.sort_index()
    df['Cumulative_Return'] = (1 + df['Vol_Adjusted_Return']).cumprod()
    df['Running_Max'] = df['Cumulative_Return'].cummax()
    df['Drawdown'] = df['Cumulative_Return'] / df['Running_Max'] - 1

    # Scaling factor: linearly decreases from 1 to 0 between full_risk_drawdown and max_drawdown_threshold
    def get_scaling_factor(drawdown):
        if drawdown > full_risk_drawdown:
            return 1.0
        elif drawdown <= max_drawdown_threshold:
            return 0.0
        else:
            return (drawdown - max_drawdown_threshold) / (full_risk_drawdown - max_drawdown_threshold)

    df['Scaling_Factor'] = df['Drawdown'].apply(get_scaling_factor)
    df['Adjusted_Return'] = df['Vol_Adjusted_Return'] * df['Scaling_Factor']
    df['Adjusted_Cumulative_Return'] = (1 + df['Adjusted_Return']).cumprod()
    
    df['Long Pos Notional'] = df['Long Pos Notional'] * df['Scaling_Factor']
    df['Short Pos Notional'] = df['Short Pos Notional'] * df['Scaling_Factor']
    return df.drop(columns=['Running_Max', 'Drawdown'])


def apply_max_drawdown_constraint(df, max_drawdown_threshold=-0.2):
    # Ensure the dataframe is sorted by date
    df = df.sort_index()

    # Calculate the cumulative return (assuming 'Return' is daily return, you can adjust if needed)
    df['Cumulative_Return'] = (1 + df['Vol_Adjusted_Return']).cumprod()

    # Calculate the running maximum of the cumulative return (for calculating drawdown)
    df['Running_Max'] = df['Cumulative_Return'].cummax()

    # Calculate the drawdown as the difference between running maximum and the current cumulative return
    df['Drawdown'] = df['Cumulative_Return'] - df['Running_Max']

    # Apply the max drawdown constraint
    df['Adjusted_Return'] = np.where(df['Drawdown'] < max_drawdown_threshold, 0, df['Vol_Adjusted_Return'])

    # Optionally, re-calculate cumulative return with the adjusted returns
    df['Adjusted_Cumulative_Return'] = (1 + df['Adjusted_Return']).cumprod()

    # Drop temporary columns
    df = df.drop(columns=['Running_Max', 'Drawdown'])
    
    return df
