import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


def BS(S_0, r, sigma, T, nSteps, nPaths = 1000):
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

def eu_call_price(S0, K, T, r, sigma, nSteps):
    """
    Returns european call option price following Black Scholes Model.
    """
    X = BS(S0, r, sigma, T, nSteps)
    S = np.mean(X[:,-1])
    Price = np.exp(-r*T)*max(0,S0-K)

    # ---- Uncomment to plot results -------
    #for i in range(len(X[:,0])):
    #    plt.plot(range(len(X[1])),X[i])
    #plt.show()

    return Price

# Uncomment to test
# print(eu_call_price(100,1,1,0.05,100, 0.4))
def eu_put_price(S0, K, T, r, sigma, nSteps):
    """
    Returns european call option price following Black Scholes Model.
    """
    X = BS(S0, r, sigma, T, nSteps)
    S = np.mean(X[:,-1])
    Price = np.exp(-r*T)*max(0,K-S0)

    # ---- Uncomment to plot results -------
    #for i in range(len(X[:,0])):
    #    plt.plot(range(len(X[1])),X[i])
    #plt.show()

    return Price


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
k = 5
nPaths = 1
M = 10

plt.style.use('ggplot')

def Merton_J_D(S0, T, K, r, sigma,  lam, m, v, steps, Npaths, type):
    size=(steps,Npaths)
    dt = T/steps 
    poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                         np.random.normal(m,v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=size)), axis=0)
    
    inbetween = np.exp(geo+poi_rv)*S0

    if type == 'puts':
        p = 0 # price of option
        for k in range(40):
            r_k = r - lam*(m-1) + (k*np.log(m) ) / T
            sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
            k_fact = np.math.factorial(k) # 
            p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact)) \
                        * eu_put_price(S0, K, T, r_k, sigma_k, 1)
        return p 
    elif type == "calls":
        p = 0
        for k in range(40):
            r_k = r - lam*(m-1) + (k*np.log(m) ) / T
            sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
            k_fact = np.math.factorial(k)
            p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact))  * eu_call_price(S0, K, T, r_k, sigma_k, 1)
        
        return p 
    else:
        raise Exception("Unexpected input, try again")



S0 = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0.1 # meean of jump size
v = 0.3 # standard deviation of jump
lam =1 # intensity of jump i.e. number of jumps per annum
steps =10000 # time steps
Npaths = 1 # number of paths to simulate
sigma = 0.2 # annaul standard deviation , for weiner process

# j = Merton_J_D(S0, T, r, sigma, lam, m, v, steps, Npaths)


plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
# plt.show()


print(Merton_J_D(187.44, 50, 3/365, 0.06, 0.332, m, v, lam, type='calls'), "hello")
# print(BinomialTree(187.44, 50, 3/365, 0.332, 0.06, 'calls'))