
import pandas as pd
import numpy as np

def calculate_features():
    file_path = "vix_pkg/data/vixprices.xlsx"
    df = pd.read_excel(file_path)

    # Remove first row and name column 1 as 'Date'
    df = df.iloc[1:]
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)


    # Rename columns for simplicity
    df.rename(columns={
        'SPVIX2ME': 'CMF2', 'SPVIX3ME': 'CMF3', 'SPVIX4ME': 'CMF4',
        'SPVIX6ME': 'CMF6', 'SPVXMP': 'CMF5', 'SPVSP': 'CMF1'
    }, inplace=True)


    # List of approximated time to maturity (using trading days)
    maturity_days = {
        'CMF1': 21,  # Approx. 1 month
        'CMF2': 42,  # Approx. 2 months
        'CMF3': 63,  # Approx. 3 months
        'CMF4': 84,  # Approx. 4 months
        'CMF5': 105, # Approx. 5 months
        'CMF6': 126  # Approx. 6 months
    }

    # Calculate Roll
    for i in range(2, 7):  # CMF2 to CMF6
        current_cmf = f'CMF{i}'
        prev_cmf = f'CMF{i-1}'
        T_current = maturity_days[current_cmf]
        T_prev = maturity_days[prev_cmf]
        
        # Calculate Roll
        df[f'Roll_{i}'] = (df[current_cmf] - df[prev_cmf]) / (df[prev_cmf] * (T_current - T_prev) * (1/252))


    # Calculate Delta Roll (change in roll yield)
    for i in range(2, 7):  # CMF2 to CMF6
        df[f'Delta_Roll_{i}'] = df[f'Roll_{i}'].diff()



    # Calculate change in CMF (mu)
    for i in range(2, 7):  # CMF2 to CMF6
        current_cmf = f'CMF{i}'
        prev_cmf = f'CMF{i-1}'
        df[f'mu_{i}'] = df[current_cmf] - df[prev_cmf]

    # Create lag columns for CMF variables (CMF1 to CMF6) for 1, 2, and 3 days
    for cmf in ['CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']:
        df[f'lag_{cmf}_01'] = df[cmf].shift(1)
        df[f'lag_{cmf}_02'] = df[cmf].shift(2)
        df[f'lag_{cmf}_03'] = df[cmf].shift(3)

    # Create deltalag columns
    for cmf in ['CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']:
        # deltalag_CMFX_01: Today's price minus yesterday's price
        df[f'deltalag_{cmf}_01'] = df[cmf] - df[f'lag_{cmf}_01']
        
        # deltalag_CMFX_02: Yesterday's price minus the price from 2 days ago
        df[f'deltalag_{cmf}_02'] = df[f'lag_{cmf}_02'] - df[f'lag_{cmf}_01']
        
        # deltalag_CMFX_03: The price from 3 days ago minus the price from 2 days ago
        df[f'deltalag_{cmf}_03'] = df[f'lag_{cmf}_03'] - df[f'lag_{cmf}_02']


    # Create Delta_Roll_X_Y columns for all pairs of CMFs (with Y > X)
    for x in range(1, 7):
        for y in range(x + 1, 7):
            cmf_x = f'CMF{x}'
            cmf_y = f'CMF{y}'
            T_x = maturity_days[cmf_x]
            T_y = maturity_days[cmf_y]
            # Calculate generalized roll yield for the pair (X, Y)
            # This formula is analogous to the one used for consecutive maturities,
            # but now uses the difference in maturity for any pair.
            df[f'Delta_Roll_{x}_{y}'] = (df[cmf_y] - df[cmf_x]) / (df[cmf_x] * (T_y - T_x) * (1/252))



    # rearrange columns
    cmf_columns = ['Date', 'CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']
    other_columns = [col for col in df.columns if col not in cmf_columns]
    df = df[cmf_columns + other_columns]
    df.head()


    # rearrange date column and set to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)


    # calculate next day returns (target variable)
    cmf_cols = ['CMF1', 'CMF2', 'CMF3', 'CMF4', 'CMF5', 'CMF6']
    for col in cmf_cols:
        df[f'Return_{col}'] = df[col].pct_change().shift(-1) # Next-day return

    # drop rows with nan (first and last row)
    df.dropna(inplace=True)

    print(df.columns)
    exit()
    df.to_excel("vix_pkg/data/vix_features_calculated.xlsx", index=False)
