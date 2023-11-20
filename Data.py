import yfinance as yf
from yahooquery import Ticker
# from datetime import date, timedelta
import pandas as pd
import numpy as np

#-- Write ticker names of desired options data --
options_identifiers = '^SPX ^NDX ^RUT ^VIX AAPL AMZN NFLX GOOG TSLA MSFT META'
#-- Write ticker names of desired stock data --
stock_identifier = ['^SPX', '^VIX', '^NDX', '^RUT',  'AAPL', 'AMZN', 'NFLX', 'GOOG', 'TSLA', 'MSFT', 'META']

# OPTIONS WITH EXISTING DATA: ^NDX, ^RUT, ^VIX, 
# Stocks tickers: , 'NKE', 'MCD', 'V', 'JPM', 'MS', 'GS', 'BAC'
#NKE MCD V JPM MS GS BAC'
# , 'A','A', 'A', 'A', 'A','A', 'A'

#-- Write option methods (American = 'A', European = 'E')
methods = ['E','E','E', 'E', 'A', 'A', 'A', 'A','A', 'A', 'A']
df_methods = pd.DataFrame({'symbol': stock_identifier, 'method': methods})

#-- Set paths for data files
stock_path = './Data/stock.csv'
option_path = './Data/options.csv'
old_option_path = './Data/options_1511.csv'
merged_path = './Data/tmp.csv'

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
        temp_data = temp_data[:1265]
        temp_data = temp_data.set_index(df.index)
        df[stock_identifier[i]] = temp_data

    # Polishing date format...
    df = df.reset_index() 
    df.index = df.Date.dt.date
    df = df.drop(columns=['Date'])

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


def getoptions(update = False, OldOptions = False):
    if update == True:
        df = update_options()
    elif OldOptions == False:
        df = pd.read_csv(option_path,index_col=[0,1])
    elif OldOptions == True:
        df1 = pd.read_csv(option_path)
        df2 = pd.read_csv(old_option_path)
        df = pd.concat([df1, df2], ignore_index = True)
    #print(df)
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

    # Export to CSV
    df.to_csv(stock_path)
    return df

def GetData(start_date, df_end, trade_days, Update, OldOptions = False):
    # Get data from Data file
    df_stock = getstock(start_date, df_end, Update)
    df_options = getoptions(Update, OldOptions)

    # Add maturity column to options database
    maturity = (pd.to_datetime(df_options['expiration']) - pd.to_datetime(df_options['lastTradeDate'])).dt.days
    df_options = df_options.assign(maturity = maturity)

    # Create long version of stock database
    df_stock_long = df_stock.reset_index()
    df_stock_long = pd.melt(df_stock_long,id_vars='Date',var_name ='stock_name', value_name ='S0')

    # Create Volatility column for dataframe
    stock_names = df_stock_long['stock_name'].unique()
    volatility = []
    returns = []

    for i in stock_names:
        stock_series = df_stock_long[df_stock_long['stock_name'] == i]['S0']
        log_ret = np.log(stock_series) - np.log(stock_series.shift(1))
        daily_std = log_ret.expanding().std()
        annual_std = daily_std.array * np.sqrt(trade_days)
        volatility.extend(annual_std)

        daily_ret = log_ret.expanding().mean()
        annual_ret = (1+daily_ret)**trade_days
        continuous_ret = np.log(annual_ret)
        returns.extend(continuous_ret)

    df_stock_long['sigma'] = volatility
    df_stock_long['returns'] = returns    

    # Merge databases
    df_merged = pd.merge(df_options.reset_index(), df_stock_long, left_on =['symbol','lastTradeDate'], right_on=['stock_name','Date'])
    df_merged = df_merged.drop(columns=['Date','stock_name'])

    df_merged = pd.merge(df_merged, df_methods, left_on =['symbol'], right_on=['symbol'])

    df_merged.to_csv(merged_path)
    return df_merged

