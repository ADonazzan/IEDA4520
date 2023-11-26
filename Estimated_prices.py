from datetime import date, timedelta, datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print('importing price models')
import Price_Models as pm
print('done')
import Data

# Set to true to update database from online data, if false will pull data from csv files
Update = True 
trade_days = 256
est_price_path = './Data/est_prices.csv'

#-- Desired interval examined --
start_date = date(2018,11,7)
end_date = datetime.today()
df_end = end_date + timedelta(days=1) #Add one day as yf functions don't include last day
df_end = df_end.date()

# Importing data
df = Data.GetData(start_date, df_end, trade_days, Update)
df = df[df.lastTradeDate >= date(2023, 11, 21)]
#df = df.sample(1000)  #Reduce data size for testing purposes
r = 0.0533

# Uncomment to select only one stock
# stock = '^SPX'
# df = df[df['symbol'] == stock]


LSMC_est_price = [] 
BIN_est_price = [] 
BS_est_price = [] 
MJD_est_price = []
DTR_est_price = []
XGBr_est_price = []

# iterates through every option in option chain given a certain stock 
for i in tqdm(range(len(df))):
    S0 = df.iloc[i].S0       # Sets S0
    K = df.iloc[i].strike
    T = df.iloc[i].maturity / 365
    sigma_ML = df.iloc[i].sigma
    sigma = sigma_ML - 0.1
    type = df.iloc[i].optionType
    method = df.iloc[i].method

    if method == 'A':
        computed_price_LSMC = pm.LSMC(S0, K, T, sigma, r, type)
        computed_price_BIN = pm.BinomialTree(S0, K, T, sigma, r, type)
        computed_price_BS = np.nan
        computed_price_MJD = np.nan
    elif method == 'E':
        computed_price_LSMC = np.nan
        computed_price_BIN = np.nan   
        computed_price_BS = pm.BS(S0, K, T, sigma, r, type)
        computed_price_MJD = pm.MJD(S0, K, T, sigma, r, type)
    else:
        raise Exception('Unexpected option method')
    r_ML = df.iloc[i].returns

    computed_price_DTR = pm.DTR(S0, K, T, sigma_ML, r_ML, type, method)
    computed_price_XGBr = pm.XGBr(S0, K, T, sigma_ML, r_ML, type, method)

    LSMC_est_price.append(computed_price_LSMC)
    BIN_est_price.append(computed_price_BIN)
    BS_est_price.append(computed_price_BS)
    MJD_est_price.append(computed_price_MJD)
    DTR_est_price.append(computed_price_DTR)
    XGBr_est_price.append(computed_price_XGBr)

mkt_price = df.lastPrice
mkt_price_A = df[df.method == 'A'].lastPrice
mkt_price_E = df[df.method == 'E'].lastPrice

LSMC_perc_error = (mkt_price - LSMC_est_price)/mkt_price
BIN_perc_error = (mkt_price - BIN_est_price)/mkt_price
BS_perc_error = (mkt_price - BS_est_price)/mkt_price
MJD_perc_error = (mkt_price - MJD_est_price)/mkt_price
DTR_perc_error = (mkt_price - DTR_est_price)/mkt_price
XGBr_perc_error = (mkt_price - XGBr_est_price)/mkt_price

df = df.assign(LSMC_est_price = LSMC_est_price)
df = df.assign(LSMC_perc_error = LSMC_perc_error)
df = df.assign(BIN_est_price = BIN_est_price)
df = df.assign(BIN_perc_error = BIN_perc_error)
df = df.assign(BS_est_price = BS_est_price)
df = df.assign(BS_perc_error = BS_perc_error)
df = df.assign(MJD_est_price = MJD_est_price)
df = df.assign(MJD_perc_error = MJD_perc_error)
df = df.assign(DTR_est_price = DTR_est_price)
df = df.assign(DTR_perc_error = DTR_perc_error)
df = df.assign(XGBr_est_price = XGBr_est_price)
df = df.assign(XGBr_perc_error = XGBr_perc_error)

df.to_csv(est_price_path)