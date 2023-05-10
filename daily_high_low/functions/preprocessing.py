import pandas as pd
import numpy as np

def preprocess_dataframe(df, log_returns=True, price_columns = ['open', 'high', 'low', 'close']):
    # Combine the date and time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    
    # Drop the unnecessary columns
    df = df.drop(columns=['<DATE>', '<TIME>', '<TICKVOL>', '<VOL>', '<SPREAD>'])
    
    # Rename columns to remove symbols and make them lowercase
    df.columns = [col.replace('<', '').replace('>', '').lower() for col in df.columns]
    df = df[['datetime', 'open', 'high', 'low', 'close']]
    
    # Set index to datetime column
    df.set_index('datetime', inplace=True)
    
    #shift the datetime index to EST timezone
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    
    if log_returns:
        df = calculate_log_returns(df)
    
    return df


def calculate_log_returns(df, price_columns = ['open', 'high', 'low', 'close']):
    """
    Calculate log returns for specified price columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing price data.
        price_columns (list): List of column names representing prices for which log returns are to be calculated.

    Returns:
        df_log_returns (pandas.DataFrame): Output DataFrame containing log returns for the specified price columns.
    """
    for column in price_columns:
        log_return_column = f'{column}_log_return'
        df[log_return_column] = np.log(df[column] / df[column].shift(1))
    
    # Drop the first row as it will have NaN values due to the shift operation
    df = df[1:]
    return df