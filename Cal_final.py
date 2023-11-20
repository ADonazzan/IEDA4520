from datetime import date, timedelta, datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Price_Models as pm
import Data

n = 100 #Sample size
iterations = 10 # max tries

# Set to true to update database from online data, if false will pull data from csv files
Update = False 
trade_days = 256
est_price_path = './Data/tmp.csv'

#-- Desired interval examined --
start_date = date(2018,11,7)
end_date = date(2023,11,14)
df_end = end_date + timedelta(days=1) #Add one day as yf functions don't include last day

# Importing data
df = Data.GetData(start_date, df_end, trade_days, Update)
df = df.sample(n)  #Reduce data size for testing purposes


r = 0.0553

# Uncomment to select only one stock
# stock = '^SPX'
# df = df[df['symbol'] == stock]

def compute_errors(dsigma, r):
    LSMC_est_price = [] 
    BIN_est_price = [] 
    BS_est_price = [] 
    MJD_est_price = []
    
    for i in range(len(df)):
        S0 = df.iloc[i].S0       # Sets S0
        K = df.iloc[i].strike
        T = df.iloc[i].maturity / 365
        sigma = df.iloc[i].sigma + dsigma
        type = df.iloc[i].optionType

        if df.iloc[i].method == 'A':
            computed_price_LSMC = pm.LSMC(S0, K, T, sigma, r, type)
            computed_price_BIN = pm.BinomialTree(S0, K, T, sigma, r, type)
            computed_price_BS = np.nan
            computed_price_MJD = np.nan
        elif df.iloc[i].method == 'E':
            computed_price_LSMC = np.nan
            computed_price_BIN = np.nan     
            computed_price_BS = pm.BS(S0, K, T, sigma, r, type)
            computed_price_MJD = pm.MJD(S0, K, T, sigma, r, type)
        else:
            raise Exception('Unexpected option method')

    LSMC_est_price.append(computed_price_LSMC)
    BIN_est_price.append(computed_price_BIN)
    BS_est_price.append(computed_price_BS)
    MJD_est_price.append(computed_price_MJD)

    LSMC_perc_error = (df.lastPrice - LSMC_est_price)/df.lastPrice
    BIN_perc_error = (df.lastPrice - BIN_est_price)/df.lastPrice
    BS_perc_error = (df.lastPrice - BS_est_price)/df.lastPrice
    MJD_perc_error = (df.lastPrice - MJD_est_price)/df.lastPrice

    averageLSMC = np.sqrt(np.mean((LSMC_perc_error)**2))
    averageBIN = np.sqrt(np.mean((BIN_perc_error)**2))
    averageBS = np.sqrt(np.mean((BS_perc_error)**2))
    averageMJD = np.sqrt(np.mean((MJD_perc_error)**2))

    return averageLSMC, averageBIN, averageBS, averageMJD

def testvalues():
    r = 0.05 # Starting r
    dsigma = -0.125 #Starting delta sigma
    errors = compute_errors(dsigma, r)
    df_errors_r = pd.DataFrame({'dsigma': dsigma, 'r': r, 'LSMCerror' : errors[0], 'BINerror' : errors [1], 'BSerror' : errors[2], 'MJDerror' : errors[3]},index=[0])

    for i in tqdm(range(iterations)):
        errors = compute_errors(dsigma, r)
        df_errors_r.loc[i+1] = [dsigma,r,errors[0],errors[1], errors[2], errors[3]]
        r += 0.03*(np.random.rand() - 0.5)
        r = abs(r)

    r = 0.05 # Starting r
    dsigma = -0.125 #Starting delta sigma
    errors = compute_errors(dsigma, r)
    df_errors_sigma = pd.DataFrame({'dsigma': dsigma, 'r': r, 'LSMCerror' : errors[0], 'BINerror' : errors [1], 'BSerror' : errors[2], 'MJDerror' : errors[3]},index=[0])

    for i in tqdm(range(iterations)):
        errors = compute_errors(dsigma, r)
        df_errors_sigma.loc[i+1] = [dsigma,r,errors[0],errors[1], errors[2], errors[3]]
        dsigma += 0.01*(np.random.randn())
        dsigma = abs(dsigma)

    return df_errors_r, df_errors_sigma

