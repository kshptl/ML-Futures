import pandas as pd
import numpy as np
import itertools
import re
import math
from math import floor
import sys

past_swings = []

def calculate_log_return_2d(arr):
    # Calculate the ratio between each element and its previous element along the rows
    ratios = arr[:, 1:] / arr[:, :-1]
    
    # Calculate the log returns using np.log and handle division by zero by setting result to NaN
    log_returns = np.where(ratios > 0, np.log(ratios), np.nan)
    
    return log_returns

def identify_swing_points_optimized(df, timeframes, get_swing_values=True, interpolation_method='interpolate'):
    
    # Iterate through each timeframe
    for timeframe in timeframes:
        # Resample the DataFrame to the specified timeframe
        if timeframe == '4H':
            resampled_df = df.resample(timeframe, offset='2H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        else:
            resampled_df = df.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

        # Calculate swing highs: A point is a swing high if its high is greater than the high to the left and right
        swing_highs = ((resampled_df['high'] > resampled_df['high'].shift(1)) & (resampled_df['high'] > resampled_df['high'].shift(-1))).shift(2).replace(np.nan, False)

        # Calculate swing lows: A point is a swing low if its low is lower than the low to the left and right
        swing_lows = ((resampled_df['low'] < resampled_df['low'].shift(1)) & (resampled_df['low'] < resampled_df['low'].shift(-1))).shift(2).replace(np.nan, False)

        if get_swing_values:
            # Create new columns for swing high and low prices
            df[f'swing_high_{timeframe}'] = resampled_df.loc[swing_highs, 'high'].reindex(df.index).astype('float32')
            df[f'swing_low_{timeframe}'] = resampled_df.loc[swing_lows, 'low'].reindex(df.index).astype('float32')
            
        else:
            # Create new columns for swing high and low prices
            #df[f'swing_high_{timeframe}'] = swing_highs.reindex(df.index).astype('float32')
            #df[f'swing_low_{timeframe}'] = swing_lows.reindex(df.index).astype('float32')
            swings = pd.Series(np.where(swing_highs, 1, np.where(swing_lows, -1, np.nan)), index=swing_highs.index)
            df[f'swing_{timeframe}'] = swings.reindex(df.index).astype('float32')
            
        
    # Interpolate or forward fill the NaN values based on the specified interpolation method
    columns_to_shift = df.filter(regex=r'^(?!open$|high$|low$|close$).+$').columns
    
    if interpolation_method == 'interpolate':
        df.interpolate(method='linear', inplace=True)
    elif interpolation_method == 'ffill':
        df.loc[:, columns_to_shift] = df.loc[:, columns_to_shift]
        df.fillna(method='ffill', inplace=True)
    else:
        pass#df.loc[:, columns_to_shift] = df.loc[:, columns_to_shift]
    return df


# get past 20 pivot points
def pivots(df, c, lookback, i = None, get_next_pivot=False):
    all_data = df[c]
    past_data = df[c][:i+1]
    
    past_pivots = get_uniques(past_data)
    
    if get_next_pivot:
        all_pivots = get_uniques(all_data)
        future_pivots = all_pivots[len(past_pivots):]
        next_pivot = future_pivots[0] if len(future_pivots) != 0 else 0    
        # if math.isnan(next_pivot) and i > 1000 and '60' in c:
        #     print('all pivots:: ', all_pivots)
        #     print('past pivots:: ', past_pivots)
        #     print('future pivots: ', future_pivots)
        #     print('next value: ', next_pivot)
        #     sys.exit()
        return (0 if math.isnan(next_pivot) else next_pivot)
    else:
        past_pivots = past_pivots[-lookback:]
        pivot_list = [np.nan] * (lookback - len(past_pivots)) + past_pivots #pad the list to keep it a length of {lookback}

        return list(pd.Series(pivot_list).fillna(0))

# this function combines pivots_to_feature and split_column_pivots into one function that is much faster
def swings_to_features_optimized(df, lookback, log_returns=True):
    global past_swings
    
    cols = df.filter(regex=r'^(?!open$|high$|low$|close$|open_log_return$|high_log_return$|low_log_return$|close_log_return$).+$').columns

    if log_returns:
        lookback_past_swings = lookback + 1 
    else:
        lookback_past_swings = lookback

    
    for c in cols:
        df_temp = df[~df[c].isna()]

        if log_returns:
            past_swings_index = [w[-1] for w in np.lib.stride_tricks.sliding_window_view(df_temp[c].index[1:], lookback)]
        else:
            past_swings_index = [w[-1] for w in np.lib.stride_tricks.sliding_window_view(df_temp[c].index, lookback)]
        
        past_swings = np.lib.stride_tricks.sliding_window_view(df_temp[c], lookback_past_swings)
        
        if log_returns:
            past_swings = calculate_log_return_2d(past_swings)
            
        col_names = [f'{c}_{i+1}' for i in range(lookback)]
        df = df.join(pd.DataFrame(past_swings, index=past_swings_index, columns=col_names).reindex(df.index, method='ffill'))
    return df
    
# get and add past {lookback} pivot points array for each row
def pivots_to_feature(df, lookback, get_next_pivot=False):

    #final_pivot_cols = [c for c in df.columns if re.match(r'zigzag_\d{1,4}[TDW]$', c)] #zigzag_##[T or D]
    final_pivot_cols = [c for c in df.columns if re.match(r'swing_points_\d{1,4}$', c)] 
    
    if get_next_pivot:
        for c in final_pivot_cols:
            next_pivot = []
            for i in range(len(df)):
                next_pivot.append(pivots(df, c, lookback, i, True))
            df[f'{c}_next_swing'] = next_pivot
            
            #print(f'Added {c} next swing')
        return df

    else:
        #initialize past # lookback pivots columns
        for c in final_pivot_cols:
            df[f'{c}_{lookback}'] = np.zeros(len(df))
        for c in final_pivot_cols:
            pivots_list = []
            for i in range(len(df)):
                pivots_list.append(pivots(df, c, lookback, i, False)) #should return size {lookback} array of past pivot ponits
            df[f'{c}_{lookback}'] = pivots_list
            
            #print(f'Added {c} swing features')
            #print('After adding the array to the df: ', type(df.zigzag_1T_20[0]), '\n')

        return df