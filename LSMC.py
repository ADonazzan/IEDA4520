import numpy as np
from scipy.stats import norm
import pandas as pd

# define the number of simulations and time steps
M = 10  # number of simulations

def LSMC_put(S0, K, T, sigma, r):
    # generate the stock price paths
    N = int(np.floor(T*252)) #Time steps with trading days
    dt = T / N
    z = np.random.randn(M, N)
    S = np.zeros((M, N+1))
    S[:,0] = S0
    for i in range(1, N+1):
        S[:,i] = S[:,i-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z[:,i-1])

    payoff = np.maximum(K - S, 0)

    # perform the least squares regression
    dis_cfl = np.zeros((M, N+1)) # discounted cashflow at every timestep 
    dis_cfl[:,N] = payoff[:,N] 
    exercise_flag = np.zeros((M,N)) # should we exercise
    cond = S[:,-1] < K # not in the money
    exercise_flag[cond, -1] = 1
    for i in range(N-1, 0, -1): # backward
        cond = S[:,i] < K
        X = np.column_stack([np.ones(M), S[:,i], S[:,i]**2])
        cond_x = X[cond, :]
        Y = np.exp(-r*dt) * dis_cfl[cond,i+1]
        beta = np.linalg.lstsq(cond_x, Y, rcond=None)[0]
        continue_val = np.dot(X, beta)
        continue_val[~cond] = 0
        cond_exercise = payoff[:,i] > continue_val
        exercise_flag[cond_exercise, i-1] = 1
        dis_cfl[:,i] = np.exp(-r*dt) * dis_cfl[:,i+1]
        dis_cfl[cond_exercise,i] = payoff[cond_exercise,i]

    stopping_criteria = np.argmax(exercise_flag, axis=1) # first exercise point

    actual_exercise = np.zeros_like(exercise_flag)
    actual_exercise[np.arange(M), stopping_criteria] = exercise_flag[np.arange(M), stopping_criteria]
    discount = (np.ones((M, N))*np.exp(-r*dt)).cumprod(axis=1)[::-1]
    exp_payoff = (actual_exercise * payoff[:,1:] * discount).sum() / M
    return exp_payoff


def LSMC_call(S0, K, T, sigma, r):
    # generate the stock price paths
    N = int(np.floor(T*252)) #Time steps with trading days
    dt = T / N
    t = np.linspace(0, T, N+1)
    z = np.random.randn(M, N)
    S = np.zeros((M, N+1))
    S[:,0] = S0
    for i in range(1, N+1):
        S[:,i] = S[:,i-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z[:,i-1])

    payoff = np.maximum(S - K, 0)

    # perform the least squares regression
    dis_cfl = np.zeros((M, N+1)) # discounted cashflow at every timestep 
    dis_cfl[:,N] = payoff[:,N] 
    exercise_flag = np.zeros((M,N)) # should we exercise
    cond = S[:,-1] > K # not in the money
    exercise_flag[cond, -1] = 1
    for i in range(N-1, 0, -1): # backward
        cond = S[:,i] > K
        X = np.column_stack([np.ones(M), S[:,i], S[:,i]**2])
        cond_x = X[cond, :]
        Y = np.exp(-r*dt) * dis_cfl[cond,i+1]
        beta = np.linalg.lstsq(cond_x, Y, rcond=None)[0]
        continue_val = np.dot(X, beta)
        continue_val[~cond] = 0
        cond_exercise = payoff[:,i] > continue_val
        exercise_flag[cond_exercise, i-1] = 1
        dis_cfl[:,i] = np.exp(-r*dt) * dis_cfl[:,i+1]
        dis_cfl[cond_exercise,i] = payoff[cond_exercise,i]

    stopping_criteria = np.argmax(exercise_flag, axis=1) # first exercise point

    actual_exercise = np.zeros_like(exercise_flag)
    actual_exercise[np.arange(M), stopping_criteria] = exercise_flag[np.arange(M), stopping_criteria]
    discount = (np.ones((M, N))*np.exp(-r*dt)).cumprod(axis=1)[::-1]
    exp_payoff = (actual_exercise * payoff[:,1:] * discount).sum() / M
    return exp_payoff


#print("American Price:", LSMC_call(135.43, 50, 3/365, 0.32, 0.06))
                                  
# pull strike price from data, calculate trading days from data, pull sigma from data, pull risk from data

def LSMC(S0, K, T, sigma, r, type):
    """
    S0 = stock price at first day
    K = strike
    T = time in years
    sigma = sigma
    r = risk free rate
    type = call/put
    """
    if type == "puts":
        price = LSMC_put(S0, K, T, sigma, r)
    elif type == "calls":
        price = LSMC_call(S0, K, T, sigma, r)
    else:
        raise Exception("Unexpected input")
    
    return price
    