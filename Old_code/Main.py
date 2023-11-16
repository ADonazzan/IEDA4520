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

df = Data.GetData(start_date, df_end, trade_days, Update)

stock = '^SPX'
r = 0.1
sim_price = []

df = df[df['symbol'] != stock]

est_price = [] 
# iterates through every option in option chain given a certain stock 
for i in range(len(df)):
    S0 = df.iloc[i].S0       # Sets S0
    K = df.iloc[i].strike
    T = df.iloc[i].maturity / 365
    sigma = df.iloc[i].sigma
    type = df.iloc[i].optionType

    computed_price_LSMC = LSMC.LSMC(S0, K, T, sigma, r, type)
    computed_price_ = pm.Merton_J_D(S0, K, T, sigma, r, type)

    est_price.append(computed_price_LSMC)
    
df = df.assign(est_price = est_price)

df.to_csv(est_price_path)