import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# Function to calculate ATR
def calculate_atr(df, period=200):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


# FVG/IFVG Detection with True/False flag
def detect_ifvg_with_true_false_flag(df, atr_multiplier=0.25, lookback=5):
    atr = calculate_atr(df, period=200).fillna(0) * atr_multiplier

    bull_fvg = []  # To track bullish FVGs
    bear_fvg = []  # To track bearish FVGs
    inversions = []  # To store IFVGs

    # Initialize a column to track if an IFVG is detected (True/False)
    df['IFVG_Detected'] = False

    # Loop over the DataFrame by row index
    for i in range(lookback, len(df)):
        high_value = df['High'].iloc[i]
        low_value = df['Low'].iloc[i]

        # Bullish FVG: Low (current bar) > High (2 bars ago)
        if low_value > df['High'].iloc[i - 2]:
            if abs(low_value - df['High'].iloc[i - 2]) > atr.iloc[i]:
                bull_fvg.append({
                    'left_time': df.index[i - 1],  # The bar creating the FVG
                    'right_time': df.index[i],  # The current bar
                    'low': low_value,  # Current bar's low (FVG low)
                    'high': df['High'].iloc[i - 2],  # 2 bars ago high (FVG high)
                    'mid': (low_value + df['High'].iloc[i - 2]) / 2,
                    'direction': 1,  # bullish
                    'state': 0  # Not yet inverted
                })

        # Bearish FVG: High (current bar) < Low (2 bars ago)
        if high_value < df['Low'].iloc[i - 2]:
            if abs(df['Low'].iloc[i - 2] - high_value) > atr.iloc[i]:
                bear_fvg.append({
                    'left_time': df.index[i - 1],  # The bar creating the FVG
                    'right_time': df.index[i],  # The current bar
                    'low': df['Low'].iloc[i - 2],  # 2 bars ago low (FVG low)
                    'high': high_value,  # Current bar's high (FVG high)
                    'mid': (high_value + df['Low'].iloc[i - 2]) / 2,
                    'direction': -1,  # bearish
                    'state': 0  # Not yet inverted
                })

        # Bullish IFVG check: Price fully enters and crosses the gap below the FVG low
        for fvg in bull_fvg[:]:
            if df['Low'].iloc[i] < fvg['high']:  # Low must enter and fill the gap
                fvg['inversion_time'] = df.index[i]
                fvg['state'] = 1  # Mark as inverted
                fvg['direction'] = -1  # Reverse the direction
                inversions.append(fvg)  # Store the IFVG
                df.at[df.index[i], 'IFVG_Detected'] = True  # Mark as True
                bull_fvg.remove(fvg)

        # Bearish IFVG check: Price fully enters and crosses the gap above the FVG high
        for fvg in bear_fvg[:]:
            if df['High'].iloc[i] > fvg['low']:  # High must enter and fill the gap
                fvg['inversion_time'] = df.index[i]
                fvg['state'] = 1  # Mark as inverted
                fvg['direction'] = 1  # Reverse the direction
                inversions.append(fvg)  # Store the IFVG
                df.at[df.index[i], 'IFVG_Detected'] = True  # Mark as True
                bear_fvg.remove(fvg)

    return df, inversions


# Function to track highs and lows of macro sessions
def track_ict_macros(df):
    # Initialize columns to store macro data
    df['London_High'] = np.nan
    df['London_Low'] = np.nan
    df['NY1_High'] = np.nan
    df['NY1_Low'] = np.nan
    df['NY3_High'] = np.nan
    df['NY3_Low'] = np.nan
    df['NY4_High'] = np.nan
    df['NY4_Low'] = np.nan

    # Define the macro session times (London and New York)
    macro_sessions = [
        {"name": "London", "start_hour": 2, "start_min": 33, "end_hour": 3, "end_min": 0},
        {"name": "NY1", "start_hour": 8, "start_min": 50, "end_hour": 9, "end_min": 10},
        {"name": "NY3", "start_hour": 13, "start_min": 10, "end_hour": 13, "end_min": 40},
        {"name": "NY4", "start_hour": 15, "start_min": 15, "end_hour": 15, "end_min": 45},
    ]

    # Loop over each session to track macro highs, lows, and mids
    for session in macro_sessions:
        session_name = session["name"]
        session_start = df.index[(df.index.hour == session["start_hour"]) & (df.index.minute == session["start_min"])]
        session_end = df.index[(df.index.hour == session["end_hour"]) & (df.index.minute == session["end_min"])]

        if len(session_start) > 0 and len(session_end) > 0:
            start_idx = session_start[0]
            end_idx = session_end[0]

            # Filter the data for the macro time range
            macro_df = df.loc[start_idx:end_idx]

            # Track highs, lows, mids
            session_high = macro_df['High'].max()
            session_low = macro_df['Low'].min()

            # Update the DataFrame for the session's high, low, mid
            df.loc[start_idx:end_idx, f'{session_name}_High'] = session_high
            df.loc[start_idx:end_idx, f'{session_name}_Low'] = session_low

    return df


# Long Condition Check
def check_long_condition(df):
    """
    Checks if the long condition is met:
    - Price is below the macro level (e.g., previous London/NY session low).
    - An IFVG (Inversion Fair Value Gap) is detected.

    Returns:
    pd.DataFrame: The DataFrame with an additional column 'Long_Condition_Met' (True/False).
    """
    df['Long_Condition_Met'] = False

    for i in range(len(df)):
        if pd.notna(df['London_Low'].iloc[i]) and df['Close'].iloc[i] < df['London_Low'].iloc[i] and df['IFVG_Detected'].iloc[i]:
            df.at[df.index[i], 'Long_Condition_Met'] = True  # Mark the long condition as True

    return df


# Short Condition Check
def check_short_condition(df):
    """
    Checks if the short condition is met:
    - Price is above the macro level (e.g., previous London/NY session high).
    - An IFVG (Inversion Fair Value Gap) is detected.

    Returns:
    pd.DataFrame: The DataFrame with an additional column 'Short_Condition_Met' (True/False).
    """
    df['Short_Condition_Met'] = False

    for i in range(len(df)):
        session_high = max(df['London_High'].iloc[i], df['NY1_High'].iloc[i], df['NY3_High'].iloc[i], df['NY4_High'].iloc[i])
        if pd.notna(session_high) and df['Close'].iloc[i] > session_high and df['IFVG_Detected'].iloc[i]:
            df.at[df.index[i], 'Short_Condition_Met'] = True  # Mark the short condition as True

    return df


# Ensure full display of DataFrame in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Example usage
gld = yf.download("GC=F", interval='5m', period='5d')

# Run FVG/IFVG detection
gld_with_ifvg, ifvgs = detect_ifvg_with_true_false_flag(gld)

# Track highs and lows for each macro session
gld_with_macro = track_ict_macros(gld_with_ifvg)

# Check long and short conditions
gld_with_conditions = check_long_condition(gld_with_macro)
gld_with_conditions = check_short_condition(gld_with_conditions)

# Output the DataFrame with macro highs, lows, and long/short conditions
print(gld_with_conditions[['Close', 'London_High', 'London_Low', 'NY1_High', 'NY1_Low', 'NY3_High', 'NY3_Low', 'NY4_High', 'NY4_Low', 'Long_Condition_Met', 'Short_Condition_Met']])

# Optional: Save the result to a CSV file
# gld_with_conditions.to_csv('output_conditions.csv')
