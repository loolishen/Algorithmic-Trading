import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt  # Fixed import


# Function to calculate ATR
def calculate_atr(df, period=200):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


# FVG/IFVG Detection Version 1, to detect IFVG in the latest bar
# def detect_fvg_ifvg(df, atr_multiplier=0.25, lookback=5, signal_preference='Close'):
#     atr = calculate_atr(df, period=200).fillna(0) * atr_multiplier
#
#     bull_fvg = []
#     bear_fvg = []
#     inversions = []
#
#     # Use range(len(df)) to loop over DataFrame by row index
#     for i in range(lookback, len(df)):
#         # FVG Up (Bullish) Condition
#         if (df['Low'].iloc[i] > df['High'].iloc[i - 2]) and (df['Close'].iloc[i - 1] > df['High'].iloc[i - 2]):
#             if abs(df['Low'].iloc[i] - df['High'].iloc[i - 2]) > atr.iloc[i]:
#                 bull_fvg.append({
#                     'left_time': df.index[i - 1],
#                     'right_time': df.index[i],
#                     'low': df['Low'].iloc[i],
#                     'high': df['High'].iloc[i - 2],
#                     'mid': (df['Low'].iloc[i] + df['High'].iloc[i - 2]) / 2,
#                     'direction': 1,  # bullish
#                     'state': 0
#                 })
#
#         # FVG Down (Bearish) Condition
#         if (df['High'].iloc[i] < df['Low'].iloc[i - 2]) and (df['Close'].iloc[i - 1] < df['Low'].iloc[i - 2]):
#             if abs(df['Low'].iloc[i - 2] - df['High'].iloc[i]) > atr.iloc[i]:
#                 bear_fvg.append({
#                     'left_time': df.index[i - 1],
#                     'right_time': df.index[i],
#                     'low': df['Low'].iloc[i - 2],
#                     'high': df['High'].iloc[i],
#                     'mid': (df['High'].iloc[i] + df['Low'].iloc[i - 2]) / 2,
#                     'direction': -1,  # bearish
#                     'state': 0
#                 })
#
#         # Check for inversions (IFVG)
#         for fvg in bull_fvg[:]:
#             if df['Low'].iloc[i] < fvg['low']:  # Inversion in bullish FVG
#                 fvg['inversion_time'] = df.index[i]
#                 fvg['state'] = 1  # Mark it as inverted
#                 fvg['direction'] = -1  # Reverse the direction
#                 inversions.append(fvg)
#                 bull_fvg.remove(fvg)
#
#         for fvg in bear_fvg[:]:
#             if df['High'].iloc[i] > fvg['high']:  # Inversion in bearish FVG
#                 fvg['inversion_time'] = df.index[i]
#                 fvg['state'] = 1  # Mark it as inverted
#                 fvg['direction'] = 1  # Reverse the direction
#                 inversions.append(fvg)
#                 bear_fvg.remove(fvg)
#
#     # Check if there's any inversion on the last data point
#     if inversions and inversions[-1]['inversion_time'] == df.index[-1]:
#         return True  # Inversion detected at the latest bar
#     else:
#         return False  # No inversion detected

# FVG/IFVG Detection Version 2, to detect every single IFVG within the scope of the chart
# Version 3: taking into account of candle stick wicks
def detect_ifvg_with_true_false_flag(df, atr_multiplier=0.25, lookback=5):
    atr = calculate_atr(df, period=200).fillna(0) * atr_multiplier

    bull_fvg = []  # To track bullish FVGs
    bear_fvg = []  # To track bearish FVGs
    inversions = []  # To store IFVGs

    # Initialize a column to track if an IFVG is detected (True/False)
    df['IFVG_Detected'] = False

    # Use range(len(df)) to loop over DataFrame by row index
    for i in range(lookback, len(df)):
        # Now using the highs and lows explicitly (wicks)
        high_value = df['High'].iloc[i]
        low_value = df['Low'].iloc[i]

        # Bullish FVG: Low (current bar) > High (2 bars ago)
        if low_value > df['High'].iloc[i - 2]:
            # Ensure the gap size is significant using ATR as a filter
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

        # Now we check if any of the tracked FVGs have been inverted
        # Bullish IFVG check: Price *must fully enter and cross the gap* below the FVG low
        for fvg in bull_fvg[:]:
            # Make sure that the price crosses the entire gap for inversion
            if df['Low'].iloc[i] < fvg['high']:  # The low must fully enter and fill the gap
                fvg['inversion_time'] = df.index[i]
                fvg['state'] = 1  # Mark as inverted
                fvg['direction'] = -1  # Reverse the direction
                inversions.append(fvg)  # Store the IFVG
                df.at[df.index[i], 'IFVG_Detected'] = True  # Mark as True in the DataFrame
                bull_fvg.remove(fvg)  # Remove it from the list once inverted

        # Bearish IFVG check: Price *must fully enter and cross the gap* above the FVG high
        for fvg in bear_fvg[:]:
            # Make sure that the price crosses the entire gap for inversion
            if df['High'].iloc[i] > fvg['low']:  # The high must fully enter and fill the gap
                fvg['inversion_time'] = df.index[i]
                fvg['state'] = 1  # Mark as inverted
                fvg['direction'] = 1  # Reverse the direction
                inversions.append(fvg)  # Store the IFVG
                df.at[df.index[i], 'IFVG_Detected'] = True  # Mark as True in the DataFrame
                bear_fvg.remove(fvg)  # Remove it from the list once inverted

    # Return the modified DataFrame with True/False IFVG detection flag and the list of inversions
    return df, inversions


# Example usage:
# Download 5-minute data for the last 5 days
gld = yf.download("GC=F", interval='5m', period='5d')

# Run the IFVG detection function with True/False flag
gld_with_ifvg, ifvgs = detect_ifvg_with_true_false_flag(gld)

# Check if any IFVGs were detected
if ifvgs:
    print("IFVGs detected:")
    for inv in ifvgs:
        print(f"IFVG detected at {inv['inversion_time']}")
else:
    print("No IFVGs detected.")

# Example usage for trading: Check if an IFVG was detected in the most recent bar
if gld_with_ifvg['IFVG_Detected'].iloc[-1]:
    print("IFVG detected in the most recent bar!")
else:
    print("No IFVG in the most recent bar.")


# Define session start and end times
def get_session_times(df, start_hour, start_min, end_hour, end_min, tz_info):
    sessions = []
    for date in df.index.date:
        start_time = datetime(date.year, date.month, date.day, start_hour, start_min, tzinfo=tz_info)
        end_time = datetime(date.year, date.month, date.day, end_hour, end_min, tzinfo=tz_info)
        sessions.append((start_time, end_time))
    return sessions


# Create sessions times for London and New York
tz_info = gld.index[0].tzinfo
london_sessions = get_session_times(gld, 2, 33, 3, 0, tz_info)
ny_sessions = get_session_times(gld, 8, 50, 9, 10, tz_info)

# Initialize session highs/lows
london_high, london_low, ny_high, ny_low = 0, float('inf'), 0, float('inf')
gld['London_High'], gld['London_Low'], gld['NewYork_High'], gld['NewYork_Low'] = np.nan, np.nan, np.nan, np.nan

# Loop through each 5-minute interval to capture highs/lows within sessions
for idx, row in gld.iterrows():
    current_time = idx

    for london_start, london_end in london_sessions:
        if london_start <= current_time <= london_end:
            london_high = max(london_high, row['High'])
            london_low = min(london_low, row['Low'])
        elif current_time > london_end and london_high > 0:
            gld.at[idx, 'London_High'] = london_high
            gld.at[idx, 'London_Low'] = london_low
            london_high, london_low = 0, float('inf')  # Reset

    for ny_start, ny_end in ny_sessions:
        if ny_start <= current_time <= ny_end:
            ny_high = max(ny_high, row['High'])
            ny_low = min(ny_low, row['Low'])
        elif current_time > ny_end and ny_high > 0:
            gld.at[idx, 'NewYork_High'] = ny_high
            gld.at[idx, 'NewYork_Low'] = ny_low
            ny_high, ny_low = 0, float('inf')  # Reset


# Long Condition Check
def check_long_condition(df):
    """
    Checks if the long condition is met:
    - Price is below the macro level (e.g., previous London/NY session low).
    - An IFVG (Inversion Fair Value Gap) is detected.

    Args:
    df (pd.DataFrame): The DataFrame containing the price data, macro levels,
                       and IFVG detection column.

    Returns:
    pd.DataFrame: The DataFrame with an additional column 'Long_Condition_Met' (True/False).
    """
    # Initialize a new column to store whether the long condition is met (True/False)
    df['Long_Condition_Met'] = False

    # Loop through the DataFrame and check the long condition for each row
    for i in range(len(df)):
        # Example: Check if price is below both the previous session lows (London/NY) and IFVG is detected
        if df['Close'].iloc[i] < df['London_Low'].iloc[i] and df['IFVG_Detected'].iloc[i]:
            df.at[df.index[i], 'Long_Condition_Met'] = True  # Mark the long condition as True

    return df


# Run the long condition check
gld_with_long_condition = check_long_condition(gld_with_ifvg)

# Output the DataFrame to review the results
print(gld_with_long_condition[['Close', 'London_Low', 'IFVG_Detected', 'Long_Condition_Met']])

# # Forward fill NaN values to maintain continuous session lines
# gld['London_High'].ffill(inplace=True)
# gld['London_Low'].ffill(inplace=True)
# gld['NewYork_High'].ffill(inplace=True)
# gld['NewYork_Low'].ffill(inplace=True)
#
# # Plot the original OHLC and the London/New York session high/low
# plt.figure(figsize=(14, 7))
#
# # Plot OHLC data
# plt.plot(gld.index, gld['Close'], label='Close Price', color='blue')
#
# # Plot London session highs and lows
# plt.plot(gld.index, gld['London_High'], label='London High', color='green', linestyle='--')
# plt.plot(gld.index, gld['London_Low'], label='London Low', color='red', linestyle='--')
#
# # Plot New York session highs and lows
# plt.plot(gld.index, gld['NewYork_High'], label='New York High', color='orange', linestyle='-.')
# plt.plot(gld.index, gld['NewYork_Low'], label='New York Low', color='purple', linestyle='-.')
#
# # Add labels and legend
# plt.title('Gold Futures (GC=F) - 5-Minute OHLC with London & New York Session Highs and Lows')
# plt.xlabel('Datetime')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
#
# # Show the plot
# plt.show()A
