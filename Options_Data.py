from yahooquery import Ticker
import datetime
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Update = False # Set to true to update database from online data, update is necessary if following values have been changed

# Set path for data file
path = './Data/options.csv'

#-- Write ticker names of desired options data --
identifiers = 'fb aapl amzn nflx goog'


def update_data():
    tickers = Ticker(identifiers) # sets tickers to stocks identified

    df = tickers.option_chain

    currency = list(set(df['currency']))

    if(currency == ['USD']):
        df2 = df.drop(columns='currency')
        df2 = df2.drop(columns=['contractSymbol'])
        inTheMoney = df2.inTheMoney
        df2 = df2.iloc[:,:2]
        df2['inTheMoney']=inTheMoney
    else:
        raise Exception("Unexpected currency found.")

    df2.to_csv(path)
    return df2


# Calls function to update database or loads old database
if Update == True:
    df = update_data()
else:
    df = pd.read_csv(path,index_col=[0,1,2])

print(df)


