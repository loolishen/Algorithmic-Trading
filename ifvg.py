import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import yfinance as yf

gld = yf.download("GC=F")

day = np.arange(1, len(gld) + 1)
gld['Day'] = day

gld.drop(columns=['Volume', 'Adj Close'], inplace = True)

print(gld)