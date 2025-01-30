import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rnd
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
import pandas_datareader as pd_data
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf
from scipy.interpolate import interp1d
from numpy import arange, array, exp
import Inputs

# Created penalty functions 

def penalty_tensor(K,T,device):
    """
    Computes arbitrage penalty matrices for implied volatility surfaces.

    Parameters:
    -----------
    K : Tensor
        Moneyness grid (m = K/S, strike-to-spot ratio).
    T : Tensor
        Time-to-maturity grid (τ in years).
    device : str
        Computational device ('cpu' or 'cuda').

    Returns:
    --------
    P_T : Tensor
        Time-to-maturity arbitrage penalty matrix.
    P_K : Tensor
        Moneyness arbitrage penalty matrix.
    PB_K : Tensor
        Second derivative penalty matrix for smoothness in the moneyness dimension.
    """
    P_T = torch.zeros(size=(len(T),len(T)),dtype=torch.float,device = device)
    P_K = torch.zeros(size=(len(K),len(K)),dtype=torch.float,device = device)
    PB_K = torch.zeros(size=(len(K),len(K)),dtype=torch.float,device = device)
    
    # TODO: Time-to-Maturity Penalty (P_T)
    #P_T first, the last one is zero
    for j in tqdm(np.arange(0,len(T)-1,1)):
        P_T[j,j] = T[j]/(T[j+1]-T[j])
        P_T[j+1,j] = -T[j]/(T[j+1]-T[j])
    
    # TODO: Moneyness Penalty (P_K)
    #now P_K and then PB_K 
    for i in tqdm(np.arange(0,len(K)-1,1)):
        P_K[i,i] = -1/(K[i+1]-K[i])
        P_K[i,i+1] = 1/(K[i+1]-K[i])
    
    # TODO: Second Derivative Penalty for Smoothness (PB_K)
    #PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    #once we fix the grid we have to run this function only once so it doesn't matter much
    for i in tqdm(np.arange(1,len(K)-1,1)):
        PB_K[i,i-1] =  -(K[i+1]-K[i]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
        PB_K[i,i] = (K[i+1] - K[i-1]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
        PB_K[i,i+1] = -(K[i]-K[i-1]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))

    return P_T,P_K,PB_K


def penalty_mutau_tensor(mu,T,device):
    return penalty_tensor(mu,T,device)

# TODO
def arbitrage_penalty_tensor(C,P_T,P_K,PB_K):
    """
    Given the prices of calls C for a fixed grid (K,T)
    P_T, P_K, PB_K: the matrices calculated by penalty_matrices
    Returns matrix penalties (for each point) in order
    1) penalty for violating C being increasing in T
    2) penalty for violating C being decreasing in K
    3) penalty for violating C being convex in K
    plus
    4) the sum of all penalties together (scalar)
    """
    """
    Compute arbitrage penalties for a given call option price grid C(K,T).
    
    Parameters:
    -----------
    C : Tensor
        A tensor of call option prices across a (strike, time-to-maturity) grid.
    P_T : Tensor
        Matrix enforcing the no-arbitrage condition that call prices must be increasing in time-to-maturity (T).
    P_K : Tensor
        Matrix enforcing the no-arbitrage condition that call prices must be decreasing in strike price (K).
    PB_K : Tensor
        Matrix enforcing the convexity condition for call prices in K.
    
    Returns:
    --------
    P1 : Tensor
        Penalty for violating the increasing condition in T.
    P2 : Tensor
        Penalty for violating the decreasing condition in K.
    P3 : Tensor
        Penalty for violating the convexity condition in K.
    Total penalty (scalar):
        The sum of all penalties across the grid.
    """

    # Penalty 1: Ensure call prices C(K,T) are increasing in time-to-maturity T
    P1 = torch.max(torch.tensor(0.0),torch.matmul(C,P_T))
    # Penalty 2: Ensure call prices C(K,T) are decreasing in strike price K
    P2 = torch.max(torch.tensor(0.0),torch.matmul(P_K,C))
    # Penalty 3: Ensure call prices C(K,T) are convex in strike price K
    P3 = torch.max(torch.tensor(0.0),torch.matmul(PB_K,C))
    # Compute total arbitrage penalty (sum over all elements in P1, P2, P3 by sum over batch dimensions)
    return P1,P2,P3,torch.sum(P1,dim=(1,2)) + torch.sum(P2,dim=(1,2)) + torch.sum(P3,dim=(1,2))



# BS model

def smallBS_tensor(m,tau,sigma,r):
    """
    relative call: tensor
    
    This function is used to compute relative Black-Scholes call option prices 
    based on the generated implied volatility surfaces
    """
    norm = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    d1 = (-torch.log(m)+tau*(r+0.5*sigma*sigma))/(sigma*torch.sqrt(tau))
    d2 = d1-sigma*torch.sqrt(tau)
    price = norm.cdf(d1)-m*norm.cdf(d2)*torch.exp(-r*tau)
    ####avoiding numerical errors
    price[price<=0] = 10**(-10) 
    return price
