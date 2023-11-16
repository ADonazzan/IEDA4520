import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import math

nPaths = 10

'''
----------------------------------------
Black Scholes model for european options
----------------------------------------
'''
def BS_path(S_0, r, sigma, T, nSteps):
    """
    Returns [nPaths, nSteps] matrix of nPaths paths of stock price following Black Scholes Model.
    :int T: Number of years the simulation lasts
    """
    delta_i = T/nSteps
    X = np.zeros((nPaths, nSteps+1))      #Create a null matrix of size nPaths, nSteps
    X[:,0] = S_0                          #Set entries on first column to S_0

    for i in range(nSteps):
        #for each step of the simulation, obtain n = nPath normally distributed random numbers
        Z = stats.norm.rvs(size = (nPaths))    
        #add to the result matrix the simulated price at time x + DeltaT according to BS model
        X[:,i+1] = X[:,i]*np.exp((r-0.5*(sigma**2))*(delta_i)+sigma*np.sqrt(delta_i)*Z)
    return X

def BS_call_price(S0, K, T, r, sigma):
    """
    Returns european call option price following Black Scholes Model.
    """
    nSteps = int(np.ceil(T*252)) #Time steps with trading days
    X = BS_path(S0, r, sigma, T, nSteps)
    S = np.mean(X[:,-1])
    Price = np.exp(-r*T)*max(0,S-K)
    return Price

def BS_put_price(S0, K, T, r, sigma):
    """
    Returns european call option price following Black Scholes Model.
    """
    nSteps = int(np.ceil(T*252)) #Time steps with trading days
    X = BS_path(S0, r, sigma, T, nSteps)
    S = np.mean(X[:,-1])
    Price = np.exp(-r*T)*max(0,K-S)
    return Price

def BS(S0, K, T, sigma, r, type):
    """
    S0 = stock price at first day
    K = strike
    T = time in years
    sigma = sigma
    r = risk free rate
    type = call/put
    """
    if type == "puts":
        price = BS_put_price(S0, K, T, sigma, r)
    elif type == "calls":
        price = BS_call_price(S0, K, T, sigma, r)
    else:
        raise Exception("Unexpected input")
    
    return price


'''
----------------------------------------
Binomial Tree model for american options
----------------------------------------
'''
def BinomialTree(S0, K, T, sigma, r, type):

    N = int(np.floor(T*252)) #Time steps with trading days
    dt = T / N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r*dt) - d) / (u - d)

    stock_tree = np.zeros((N+1, N+1))
    stock_tree[0, 0] = S0
    
    for i in range(1, N+1):
        stock_tree[i, 0] = stock_tree[i-1, 0] * u
        for j in range(1, i+1):
            stock_tree[i, j] = stock_tree[i-1, j-1] * d

    option_tree = np.zeros((N+1, N+1))
    for j in range(N+1):
        if type == "calls":
            option_tree[N, j] = max(stock_tree[N, j] - K, 0)
        elif type == "puts":
            option_tree[N, j] = max(K - stock_tree[N, j], 0)
        else:
            raise Exception("Unexpected input, please try again")
    
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option_tree[i, j] = np.exp(r*dt) * (q * option_tree[i+1, j] + (1-q) * option_tree[i+1, j+1])
    
    return option_tree[0, 0]

'''
---------------------------
Merton Jump Diffusion Model
---------------------------
'''
def MertonJD(S0, r, sigma, T, nSteps, lamb = 0.25, a = 0.2, b = 0.2):
    T_vec, dt = np.linspace(0, T, nSteps+1, retstep = True)
    S_arr = np.zeros([nPaths, nSteps+1])
    S_arr[:,0] = S0
    Z_1 = np.random.normal(size = (nPaths, nSteps))
    Z_2 = np.random.normal(size = (nPaths, nSteps))
    Pois = np.random.poisson(lamb*dt, (nPaths, nSteps))
    for i in range(nSteps):
        S_arr[:,i+1] = S_arr[:,i]*np.exp((r - sigma**2/2)*dt + sigma*np.sqrt(dt) * Z_1[:,i] + a*Pois[:,i] + b * np.sqrt(Pois[:,i]) * Z_2[:,i])
    return S_arr

def MJD(S0, K, T, sigma, r, type):
    """
    S0 = stock price at first day
    K = strike
    T = time in years
    sigma = sigma
    r = risk free rate
    type = call/put
    """
    nSteps = int(np.ceil(T*252)) #Time steps with trading days
    X = MertonJD(S0, r, sigma, T, nSteps)
    S = np.mean(X[:,-1])
    if type == "puts":
        price = np.exp(-r*T)*max(0,K-S)
    elif type == "calls":
        price = np.exp(-r*T)*max(0,S-K)
    else:
        raise Exception("Unexpected input")
    
    return price
