import pandas as pd
import numpy as np

def load_and_prepare_data(path='data/lpg_dataset.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['district', 'branch', 'date']).reset_index(drop=True)
    
    # Demand target
    df['demand'] = df['refill_bookings']
    
    # Calendar features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    
    # Categorical encoding (useful for future model improvement)
    df['district_code'] = df['district'].astype('category').cat.codes
    df['branch_code']  = df['branch'].astype('category').cat.codes
    
    # Lags
    for lag in [1, 7, 30]:
        df[f'demand_lag_{lag}'] = df.groupby(['district','branch'])['demand'].shift(lag)
    
    # Rolling mean
    df['demand_roll_mean_30'] = df.groupby(['district','branch'])['demand'].transform(
        lambda x: x.rolling(30, min_periods=1).mean())
    
    # Drop rows with NaN (early periods without enough history)
    df = df.dropna().reset_index(drop=True)
    
    return df