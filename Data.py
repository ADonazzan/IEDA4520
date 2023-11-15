import yfinance as yf
from yahooquery import Ticker
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Price_Models as pm

#-- Write ticker names of desired options data --
options_identifiers = '^SPX AAPL AMZN NFLX GOOG'
#-- Write ticker names of desired stock data --
stock_identifier = ['^SPX', 'AAPL', 'AMZN', 'NFLX', 'GOOG']

#-- Set paths for data files
stock_path = './Data/stock.csv'
option_path = './Data/options.csv'

# The following function pulls data from yfinance and builds the stock database
def update_stock(start_date, df_end):
    tickers = []    
    # Creates the yf ticker objects for each stock_identifier entered in "stock_identifier"
    for i in stock_identifier:           
        tickers.append(yf.Ticker(i))

    # vars()[i] = yf.Ticker(i)  # Use function to create single variables for each ticker
    #-- Create dataframe with the trading dates in the interval --
    df = tickers[0].history(start = start_date,end = df_end)[['Close']] 
    df = df.drop(columns=['Close'])

    #-- Populates dataframe with data from all the stock_identifiers entered in "stock_identifier" --
    for i in range(len(tickers)):
        stock = tickers[i]
        temp_data = stock.history(start = start_date,end = df_end)[['Close']]
        df[stock_identifier[i]] = temp_data

    # Polishing date format...
    df = df.reset_index() 
    df.index = df.Date.dt.date
    df = df.drop(columns=['Date'])
    # Export to CSV
    df.to_csv(stock_path)

    return df


# The following function pulls data from yahooquery and builds the options database
def update_options():
    tickers = Ticker(options_identifiers) # sets tickers to stocks identified

    df = tickers.option_chain

    currency = list(set(df['currency']))

    if(currency == ['USD']):
        df2 = df.drop(columns='currency')
        df2 = df2.drop(columns=['contractSymbol'])
        inTheMoney = df2.inTheMoney
        lastTradeDate = df2.lastTradeDate
        df2 = df2.iloc[:,:2]
        df2['lastTradeDate']=lastTradeDate
        df2['inTheMoney']=inTheMoney
        
    else:
        raise Exception("Unexpected currency found.")
    return df2


def getoptions(update = False):
    if update == True:
        df = update_options()
    else:
        df = pd.read_csv(option_path,index_col=[0,1])

    df = df.reset_index()
    df['expiration'] = pd.to_datetime(df["expiration"], format= '%Y-%m-%d').dt.date
    df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate']).dt.date
    df.set_index(['symbol','optionType'], inplace=True)  
    df.to_csv(option_path)
    return df


def getstock(start_date, df_end, update = False):
    if update == True:
        df = update_stock(start_date, df_end)
    else:
        df = pd.read_csv(stock_path, parse_dates=['Date'])
        df.index = df.Date.dt.date
        df = df.drop(columns=['Date'])

        #Select date interval
        df = df[df.index >= start_date]
        df = df[df.index < df_end]
    return df

