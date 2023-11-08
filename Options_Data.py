import yfinance as yf
import datetime
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Set path for data file
path = './Data/options.csv'

spx = yf.Ticker('AAPL')

#-- Write ticker names of desired options data --
identifier = ['AAPL', '^SPX']
tickers = []

for i in identifier:           
    tickers.append(yf.Ticker(i))

today = datetime.date.today()
_30_days_after_today = today + datetime.timedelta(days=30)
next_friday = str(_30_days_after_today + datetime.timedelta(days=(calendar.FRIDAY - _30_days_after_today.weekday())))
print(next_friday)


exps = spx.option_chain(date=next_friday)
print(exps)


