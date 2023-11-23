import yfinance as yf
from yahooquery import Ticker
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Price_Models as pm
import Data

def getdata(OldData = False):
    # Set to true to update database from online data, if false will pull data from csv files
    Update = False
    trade_days = 256
    est_price_path = './Data/tmp.csv'

    #-- Desired interval examined --
    start_date = date(2018,11,7)
    end_date = datetime.today()
    df_end = end_date + timedelta(days=1) #Add one day as yf functions don't include last day
    df_end = df_end.date()

    # Importing data
    df = Data.GetData(start_date, df_end, trade_days, Update, OldData)
    df = df.drop(columns=['symbol', 'expiration', 'lastTradeDate', 'inTheMoney'])

    def binary_vec_type(strings):
        return [int(s == "calls") for s in strings]
    def binary_vec_method(strings):
        return [int(s == "A") for s in strings]

    df.optionType = binary_vec_type(df.optionType)
    df.method = binary_vec_method(df.method)
    df.maturity = df.maturity/365
    lastprice = df.lastPrice
    df = df.drop(columns='lastPrice')
    df = df.assign(lastPrice = lastprice)
    return df