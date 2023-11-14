import yfinance as yf
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Price_Models as pm

Update = False # Set to true to update database from online data, update is necessary if following values have been changed

#Set path for data file
path = './Data/stock.csv'

#-- Write ticker names of desired stock data --
identifier = ['AAPL', '^SPX']


#-- Desired interval examined --
start_date = date(2018,11,7)
end_date = date(2022,11,7)
df_end = end_date + timedelta(days=1) #Add one day as yf functions don't include last day

# The following function pulls data from yfinance and builds the database
def update_data():
    tickers = []    
    # Creates the yf ticker objects for each identifier entered in "identifier"
    for i in identifier:           
        tickers.append(yf.Ticker(i))

    # vars()[i] = yf.Ticker(i)  # Use function to create single variables for each ticker
    #-- Create dataframe with the trading dates in the interval --
    df = tickers[0].history(start = start_date,end = df_end)[['Close']] 
    df = df.drop(columns=['Close'])

    #-- Populates dataframe with data from all the identifiers entered in "identifier" --
    for i in range(len(tickers)):
        stock = tickers[i]
        temp_data = stock.history(start = start_date,end = df_end)[['Close']]
        df[identifier[i]] = temp_data

    # Polishing date format...
    df = df.reset_index() 
    df.index = df.Date.dt.date
    df = df.drop(columns=['Date'])
    # Export to CSV
    df.to_csv(path)

    return df


# Calls function to update database or loads old database
if Update == True:
    df = update_data()
else:
    df = pd.read_csv(path, parse_dates=['Date'])
    df.index = df.Date.dt.date
    df = df.drop(columns=['Date'])


S = df.at[end_date,'^SPX']
K = 4378.379883

returns = np.diff(df['^SPX']) # np.log(df['^SPX']) - np.log(df['^SPX'].shift(1)) for log returns

sigma = np.std(returns)

print(pm.eu_call_price(K,K,1,0.05,250,sigma/100))


#x = list(df.index)
#x.pop(0)
#plt.plot(x,returns)
#plt.show()

#bs.BS(S = )
#print(aapl.options) # list of dates 
#DF_calls = aapl.option_chain(aapl.options[1])
#DF_calls = pd.DataFrame(DF_calls[1])