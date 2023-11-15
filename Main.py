from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Price_Models as pm
import Data
import LSMC

# Set to true to update database from online data, if false will pull data from csv files
Update = False 
trade_days = 256
est_price_path = './Data/est_prices.csv'

#-- Desired interval examined --
start_date = date(2018,11,7)
end_date = date(2023,11,14)
df_end = end_date + timedelta(days=1) #Add one day as yf functions don't include last day

def GetData():
    # Get data from Data file
    df_stock = Data.getstock(start_date, df_end, Update)
    df_options = Data.getoptions(Update)

    # Add maturity column to options database
    maturity = (pd.to_datetime(df_options['expiration']) - pd.to_datetime(df_options['lastTradeDate'])).dt.days
    df_options = df_options.assign(maturity = maturity)

    # Create long version of stock database
    df_stock_long = df_stock.reset_index()
    df_stock_long = pd.melt(df_stock_long,id_vars='Date',var_name ='stock_name', value_name ='S0')

    # Create Volatility column for dataframe
    stock_names = df_stock_long['stock_name'].unique()
    volatility = []

    for i in stock_names:
        stock_series = df_stock_long[df_stock_long['stock_name'] == i]['S0']
        log_ret = np.log(stock_series) - np.log(stock_series.shift(1))
        daily_std = log_ret.expanding().std()
        annual_std = daily_std.array * np.sqrt(trade_days)
        volatility.extend(annual_std)

    df_stock_long['sigma'] = volatility

    # Merge databases
    df_merged = pd.merge(df_options.reset_index(), df_stock_long, left_on =['symbol','lastTradeDate'], right_on=['stock_name','Date'])
    df_merged = df_merged.drop(columns=['Date','stock_name'])
    return df_merged

df = GetData()
stock = 'AAPL'
r = 0.07
sim_price = []

df = df[df['symbol'] == stock]

est_price = [] 
# iterates through every option in option chain given a certain stock 
for i in range(len(df)):
    S0 = df.iloc[i].S0       # Sets S0
    K = df.iloc[i].strike
    T = df.iloc[i].maturity / 365
    sigma = df.iloc[i].sigma
    type = df.iloc[i].optionType

    computed_price_LSMC = LSMC.LSMC(S0, K, T, sigma, r, type)

    est_price.append(computed_price)
    
df = df.assign(est_price = est_price)

df.to_csv(est_price_path)