import pandas as pd
import numpy as np
import itertools
import re
import math
from math import floor
import sys
import warnings


def calculate_log_return_2d(arr, calling_function=''):
    # Calculate the ratio between each element and its previous element along the rows
    try:
        # Temporarily promote warnings to errors within the context block
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            # Perform the division (This may raise a RuntimeWarning)
            ratios = arr[:, 1:] / arr[:, :-1]
    except RuntimeWarning as e:
        # Catch the RuntimeWarning and print the arr value
        print("RuntimeWarning occurred for the following array:")
        print(calling_function)
        print([a for a in arr if 0 in a])
        print("Exception message:", e)
        # Re-raise the exception to stop the code execution
        raise
    
    # Calculate the log returns using np.log and handle division by zero by setting result to NaN
    log_returns = np.where(ratios > 0, np.log(ratios), np.nan)
    
    return log_returns

def identify_fair_value_gaps_optimized(df, timeframes, get_high_low=True):
    df_with_fvg = df.copy(deep=True)
    for timeframe in timeframes:
        if timeframe == '4H':
            resampled_df = df.resample(timeframe, offset='2H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        else:
            resampled_df = df.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        high_shifted = resampled_df['high'].shift(2)
        low_shifted = resampled_df['low'].shift(2)
        gap_up = (high_shifted < resampled_df['low'])
        gap_down = (low_shifted > resampled_df['high'])
        fair_value_gaps = pd.Series(np.where(gap_up, 1, np.where(gap_down, -1, np.nan)), index=gap_up.index).shift(-1) #(gap_up | gap_down).astype(int)
        
        fvg_col_name = f'fair_value_gap_{timeframe}'     
        df_with_fvg[fvg_col_name] = fair_value_gaps.reindex(df_with_fvg.index).fillna(0).astype('int8')
        df_with_fvg[fvg_col_name] = df_with_fvg[fvg_col_name].shift(2) #avoid lookahead bias
        
        if get_high_low:
            fvg_high = np.maximum(gap_up * resampled_df['low'], gap_down * low_shifted).shift(-1)
            fvg_low = np.maximum(gap_up * high_shifted, gap_down * resampled_df['high']).shift(-1)
            
            fvg_high_col_name = f'fair_value_gap_{timeframe}_high'
            fvg_low_col_name = f'fair_value_gap_{timeframe}_low'
            
            df_with_fvg[fvg_high_col_name] = fvg_high.reindex(df_with_fvg.index).fillna(0).astype('int8')
            df_with_fvg[fvg_low_col_name] = fvg_low.reindex(df_with_fvg.index).fillna(0).astype('int8')
            
            df_with_fvg[fvg_high_col_name] = df_with_fvg[fvg_col_name].shift(2) #avoid lookahead bias
            df_with_fvg[fvg_low_col_name] = df_with_fvg[fvg_col_name].shift(2) #avoid lookahead bias
    
    return df_with_fvg

# old function, not being used
def identify_fair_value_gaps(df, timeframes):
    df_with_fvg = df#.copy(deep=True)
    for timeframe in timeframes:
        fair_value_gaps = []
        resampled_df = df.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        for i in range(2, len(resampled_df)):
            if resampled_df['high'].iloc[i-2] < resampled_df['low'].iloc[i]:
                fair_value_gaps.append(1)
            elif resampled_df['low'].iloc[i-2] > resampled_df['high'].iloc[i]:
                fair_value_gaps.append(1)
            else:
                fair_value_gaps.append(0)
        fvg_col_name = f'fair_value_gap_{timeframe}'
        df_with_fvg[fvg_col_name] = pd.Series(fair_value_gaps, index=resampled_df.index[2:]).fillna(False)
        df_with_fvg[fvg_col_name].fillna(0, inplace=True)
    return df_with_fvg

# this function combines fvg_to_feature and split_column_fvg into one function that is much faster
def fvg_to_feature_optimized(df, lookback, log_returns=True):
    final_fvg_cols = [c for c in df.columns if re.match(r'fair_value_gap_\d{1,4}[TDW]$', c)]
    
    if log_returns:
        lookback_fvg = lookback + 1 
    else:
        lookback_fvg = lookback
        
    for c in final_fvg_cols:
        mask = df[df[c].abs() == 1]
        if log_returns:
            windows = np.lib.stride_tricks.sliding_window_view(mask[1:].index, lookback) # we need {lookback+1} to do the log_returns for fvg_high/fvg_low even though the length stays {lookback}. The first row value is always NaN, since there is no previous value, so we ignore that in the sliding view.
        else:
            windows = np.lib.stride_tricks.sliding_window_view(mask.index, lookback)
        deltas = (windows[:, -1:] - windows[:, :]) / np.timedelta64(1, 's') // 60
        deltas_index = [pd.to_datetime(w[-1]) for w in windows]
        fvg_high = np.lib.stride_tricks.sliding_window_view(mask[f'{c}_high'], lookback_fvg)
        fvg_low = np.lib.stride_tricks.sliding_window_view(mask[f'{c}_low'], lookback_fvg)
        
        if log_returns:
            fvg_high = calculate_log_return_2d(fvg_high, f'{c}, fvg_high')
            fvg_low = calculate_log_return_2d(fvg_low, f'{c}, fvg_low')            

        col_names = [[f'{c}_{i+1}' for i in range(lookback)], [f'{c}_{i+1}_high' for i in range(lookback)], [f'{c}_{i+1}_low' for i in range(lookback)]]
        col_names = [item for sublist in col_names for item in sublist]
        df = df.join(pd.DataFrame(np.column_stack([deltas, fvg_high, fvg_low]), index=deltas_index, columns=col_names).reindex(df.index, method='ffill'))
    return df

# get and add past {lookback} fvg array for each row
def fvg_to_feature(df, lookback):
    final_fvg_cols = [c for c in df.columns if re.match(r'fair_value_gap_\d{1,4}[TDW]$', c)]
    #initialize past # lookback fvg columns
    for c in final_fvg_cols:
        df[f'{c}_{lookback}'] = np.zeros(len(df))

    for c in final_fvg_cols:
        deltas_list = []
        for i in range(len(df)):
            temp_df = df[c][:i]
            deltas = [(temp_df.index[-1] - date).seconds//60 for date in temp_df[temp_df == 1].index]
            deltas = [np.nan] * (lookback - len(deltas)) + deltas
            deltas_list.append(deltas[-lookback:])
        df[f'{c}_{lookback}'] = deltas_list
        #print(f'Added {c} fvg features')
        
    return df

# Define function to split a column into separate columns
def split_column_fvg(df, lookback):
    dfs = []
    for col in df.columns:
        if re.match(r'fair_value_gap_\d{1,4}[TDW]_\d+$', col):
            col_names = [f'{col}_{i+1}' for i in range(lookback)]
            split_columns = pd.DataFrame(df[col].tolist(), columns=col_names, index=df.index)
            dfs.append(split_columns)
        else:
            dfs.append(df[col])
    return pd.concat(dfs, axis=1)
