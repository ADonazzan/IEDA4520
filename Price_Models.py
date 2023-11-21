import numpy as np
import scipy.stats as stats

nPaths = 1000
'''
----------------------------------------
Black Scholes model for european options
----------------------------------------
'''
def BS_path(S_0, r, sigma, T, nSteps, nPaths = nPaths):
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

def BS(S0, K, T, sigma, r, type):
    """
    S0 = stock price at first day
    K = strike
    T = time in years
    sigma = sigma
    r = risk free rate
    type = call/put
    """
    nSteps = int(np.ceil(T*252)) #Time steps with trading days
    X = BS_path(S0, r, sigma, T, nSteps)

    if type == "puts":
        Price = np.mean(np.exp(-r*T)*np.maximum(0,K-X[:,-1]))
    elif type == "calls":
        Price = np.mean(np.exp(-r*T)*np.maximum(0,X[:,-1]-K))
    else:
        raise Exception("Unexpected input")
    return Price

def BinomialTree(S0, K, T, sigma, r, type):
    '''
    ----------------------------------------
    Binomial Tree model for american options
    ----------------------------------------
    '''
    N = int(np.ceil(T*252)) #Time steps with trading days
    dt = T / N
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
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
        if type == "calls":
            for j in range(i+1):
                option_tree[i, j] = max(np.exp(r*dt) * (q * option_tree[i+1, j] + (1-q) * option_tree[i+1, j+1]), stock_tree[i, j] - K)
        elif type == "puts":
            for j in range(i+1):
                option_tree[i, j] = max(np.exp(r*dt) * (q * option_tree[i+1, j] + (1-q) * option_tree[i+1, j+1]), K - stock_tree[i, j])
    
    return option_tree[0, 0]

def MertonJD(S0, r, sigma, T, nSteps, lamb = 0.075, a = -0.1, b = 0.05):
    '''
    ---------------------------
    Merton Jump Diffusion Model
    ---------------------------
    '''
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
    
    if type == "puts":
        Price = np.mean(np.exp(-r*T)*np.maximum(0,K-X[:,-1]))
    elif type == "calls":
        Price = np.mean(np.exp(-r*T)*np.maximum(0,X[:,-1]-K))
    else:
        raise Exception("Unexpected input")
    return Price

M = nPaths  # number of simulations

def LSMC_put(S_0, K, T, sigma, r):
    '''
    ---------------------------
    Least Squares Monte Carlo Model
    ---------------------------
    '''
    # generate the stock price paths
    N = int(np.ceil(T*252)) #Time steps with trading days
    dt = T / N
    # z = np.random.randn(M, N)
    # S = np.zeros((M, N+1))
    # S[:,0] = S_0
    # for i in range(1, N+1):
    #     S[:,i] = S[:,i-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z[:,i-1])

    S = BS_path(S_0, r, sigma, T, N, nPaths = M)

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


def LSMC_call(S_0, K, T, sigma, r):
    # generate the stock price paths
    N = int(np.ceil(T*252)) #Time steps with trading days
    dt = T / N
    S = BS_path(S_0, r, sigma, T, N, nPaths = M)

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
    
