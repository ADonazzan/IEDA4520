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

def eu_call_price(S_0, K, T, r, nSteps, sigma):
    """
    Returns european call option price following Black Scholes Model.
    """
    X = BS(S_0, r, sigma, T, nSteps)
    S = np.mean(X[:,-1])
    Price = np.exp(-r*T)*max(0,S-K)

    # ---- Uncomment to plot results -------
    #for i in range(len(X[:,0])):
    #    plt.plot(range(len(X[1])),X[i])
    #plt.show()

    return Price

# Uncomment to test
# print(eu_call_price(100,1,1,0.05,100, 0.4))



#Simulation parameters
Exp_val_points = 10
X_0 = 0.7


#CIR parameters
k = 5
sigma = 0.2
mu = 0.7
nPaths = 1

#Euler approximation with absolute value
def CIR_1(nSteps):
    delta_i = 1/nSteps
    X = np.zeros((nPaths, nSteps+1))      #Create a null matrix of size nPaths, nSteps
    X[:,0] = X_0                          #Set entries on first column to X_0

    for i in range(nSteps):
        #for each step of the simulation, obtain n = nPath normally distributed random numbers
        Z = stats.norm.rvs(size = (nPaths))    
        #add to the result matrix the newly computed steps
        X[:,i+1] = X[:,i] + k*delta_i*(mu-X[:,i]) + sigma*np.sqrt(delta_i*abs(X[:,i]))*Z    

    return X  