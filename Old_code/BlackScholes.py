import numpy as np
import math
from scipy.stats import norm
"""..."""

fake_data = []
T = 256/365 # amount of trading days in a year
real_prices = []

S = [] # Underlying price of the stock
K = None # Strike price
r = None # Risk free rate, something like US treasury bonds or 5 year bonds
sigma = None # Stocks true volatility?


def BS(S, K, r, sigma, T, option_type):
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if option_type == "Call":
            price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        
        elif option_type == "Put":
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
    
    except:
        print("Invalid option type")

    return price

for i in range(256):
    z_i = np.random.normal(0, 5)
    S.append(30+z_i)

print(S)


for i in range(256):
    T = (256-i)/365
    # S = fetch price at timestep T
    # K is the same
    # Do we update volatility daily? and do we update risk free rate daily?
    print("Option price is: ", round(BS(S[i], 30, 0.03, 0.20, T, "Call"), 10), "$ |", "stock price", round(S[i], 3), "| at time left to expiration", round(T, 3))