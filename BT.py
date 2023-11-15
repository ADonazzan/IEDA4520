import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from Price_Models import BS



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


# print(BinomialTree(187.44, 50, 3/365, 0.332, 0.06, 'calls'))
# print(BinomialTree(187.44, 187.5, 3/365, 0.332, 0.06, 'puts'))
# puts	2023-11-17	187.5	1.23	2023-11-14	TRUE	3	187.44000244140625	0.33206896365985056	1.6145500184892856

k = 5
nPaths = 1
M = 10
N = stats.norm.cdf
plt.style.use('ggplot')

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

def merton_jump_paths(S, T, r, sigma,  lam, m, v, steps, Npaths):
    size=(steps,Npaths)
    dt = T/steps 
    poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                         np.random.normal(m,v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=size)), axis=0)
    
    return np.exp(geo+poi_rv)*S


S0 = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam =1 # intensity of jump i.e. number of jumps per annum
steps =10000 # time steps
Npaths = 1 # number of paths to simulate
sigma = 0.2 # annaul standard deviation , for weiner process

j = merton_jump_paths(S0, T, r, sigma, lam, m, v, steps, Npaths)

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
# plt.show()

def merton_jump_call(S0, K, T, r, sigma, m , v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k)
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact))  * BS_CALL(S0, K, T, r_k, sigma_k)
    
    return p 


def merton_jump_put(S, K, T, r, sigma, m , v, lam):
    p = 0 # price of option
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k) # 
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact)) \
                    * BS_PUT(S, K, T, r_k, sigma_k)
    return p 