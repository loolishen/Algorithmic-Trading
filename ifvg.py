import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

gld = yf.download("GC=F")
day = np.arange(1, len(gld) + 1)
gld['Day'] = day
gld.drop(columns=['Volume', 'Adj Close'], inplace=True)

chart_date = gld.index[1]  # this is to get the date in the dataframe from the first column

# Initialize empty lists to store the timestamps for each row
london_times = []
new_york1_times = []
new_york3_times = []
new_york4_times = []

# Loop over each date in the DataFrame
for chart_date in gld.index:
    # Extract year, month, and day from the chart's date
    year = chart_date.year
    month = chart_date.month
    dayofmonth = chart_date.day

    # Create timestamps for each time slot
    londonStartTime = datetime(year, month, dayofmonth, 2, 33)
    londonEndTime = datetime(year, month, dayofmonth, 3, 0)

    newYork1StartTime = datetime(year, month, dayofmonth, 8, 50)
    newYork1EndTime = datetime(year, month, dayofmonth, 9, 10)

    newYork3StartTime = datetime(year, month, dayofmonth, 13, 10)
    newYork3EndTime = datetime(year, month, dayofmonth, 13, 40)

    newYork4StartTime = datetime(year, month, dayofmonth, 15, 15)
    newYork4EndTime = datetime(year, month, dayofmonth, 15, 45)

    # Store the timestamps in their respective lists
    london_times.append((londonStartTime, londonEndTime))
    new_york1_times.append((newYork1StartTime, newYork1EndTime))
    new_york3_times.append((newYork3StartTime, newYork3EndTime))
    new_york4_times.append((newYork4StartTime, newYork4EndTime))

# Convert the lists to DataFrame columns if needed
gld['LondonStartTime'], gld['LondonEndTime'] = zip(*london_times)
gld['NewYork1StartTime'], gld['NewYork1EndTime'] = zip(*new_york1_times)
gld['NewYork3StartTime'], gld['NewYork3EndTime'] = zip(*new_york3_times)
gld['NewYork4StartTime'], gld['NewYork4EndTime'] = zip(*new_york4_times)

# Print the updated DataFrame with new time columns
print(gld)
