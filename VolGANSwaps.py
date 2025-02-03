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

##############################
#####  DATA  PROCESSING  #####
##############################

def SwapsData(datapath, surfacespath):
    """
    function to read the pre-processed Swaptions implied vol data
    """

    # SETUP RETURNS ON UNDERLYING 
    # In rates world the returns are just the difference, not the log difference
    returns = pd.read_excel(datapath, skiprows = 2).set_index("Ticker").sort_index().diff().dropna()
    
    # GET DATES
    dates_dt = returns.index
    
    # PROCESS VOLATILITY SURFACE DATA
    volatilities = pd.read_excel(surfacespath, skiprows=2).set_index("Ticker")
    mat_n_ten = Inputs.maturity_tenor(surfacespath).T  

    # Initialize matrix
    surfaces_transform = np.empty((len(dates_dt), 144))  # Preallocate memory

    # Process dates
    for i, date in enumerate(dates_dt):
        d1 = pd.DataFrame(volatilities.loc[date])
        df = d1.join(mat_n_ten)  # Join precomputed maturity/tenor
        df.columns = ["Values", "Tenor", "Maturity"]
        surfaces_transform[i, :] = df["Values"].values  # Directly assign to matrix

    # SETUP THE TENOR AND TAU MATRICES
    # t \in {\frac{1}{12}, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30}
    tenor = np.array([1/12, 1/4, 1/2, 3/4, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30])
    # \tau \in {1, 2, 3, 4, 5, 6, 7, 8, 9}
    tau = np.linspace(1, 9, 9) 
    
    taus, tenors = np.meshgrid(tau,tenor)

    # NOTE: each COLUMN in returns is the full time series of return increments of a specific underlying asset with specific tenor and maturity
    # The set of assets with differing tenors and maturity for a specific date can be found in each ROW 
    # This means the COLUMN INDEX is a specific asset with some tenor and maturity 
    # The ROW INDEX is the date / time series of realised vol of that asset

    # This is shape (456, 144)
    # Same case for surfaces_transform, it is also shape (456, 144)
    
    return surfaces_transform, returns, tenor, tau, tenors, taus, dates_dt

    # Originally the penalty matrices in m and tau
    # dates = data['date'].unique()
    # data_temp = data[data['date']==dates[0]]
    # days = data_temp['days'].unique()
    # dates_format = np.copy(dates)
    # for i in tqdm(range(len(dates_format))):
    #     dates_format[i] = dates[i][6:]+'-'+dates[i][3:5]+'-'+dates[i][0:2]
    # tau = np.copy(days[1:9]) / 365
    # dtm = np.copy(days[1:])
    # m = np.linspace(0.6,1.4,10)
    # mP_t,mP_k,mPb_K = penalty_mutau(m,dtm) 
    # These values were not utilized in original VolGAN bc smoothness constraints were originally enforcing the aribtrage conditions
    # We may need to include / modify this in the future if our model does not successfully enforce arbitrage conditions for swaptions

    return surfaces_transform, log_return, tenor, tau, tenors, taus, dates_dt

def DataPreprocesssing(datapath, surfacepath):
    """
    function for preparing the data for VolGAN
    later to be split into train, val, test
    """
    surfaces_transform, returns, tenor, tau, tenors, taus, dates_dt = SwapsData(datapath,surfacepath)

    # PRODUCE REALIZED VOLATILITIES
    # Our returns are across different tenors and maturities (as if we have more than one underlying asset compared to just SPX)
    # This means that the realized volatilities we produce will actually be a matrix 

    #Realised volatility at time t-1
    realised_vol_tm1 = np.zeros((returns.shape[0]-22, returns.shape[1]))

    for i in range(len(realised_vol_tm1)):
            realised_vol_tm1[i] = np.sqrt(252 / 21) * np.sqrt(returns.iloc[i:(i+21)] ** 2).sum()
    # COLUMN INDEX is a specific asset with some tenor and maturity 
    # ROW INDEX is the date / time series of realised vol of that asset
    # This has shape (434, 144)

    #shift the time
    dates_t = dates_dt[22:]
    
    # SEPARATE OUR RETURNS
    #Return increment at t, t-1, t-2
    return_t = returns.iloc[22:].values
    return_tm1 = 252 * returns.iloc[21:-1].values
    return_tm2 = 252 * returns.iloc[20:-2].values
    # COLUMN index is a specific asset with some tenor and maturity, row index is the date / time series of realised vol of that asset

    # GET LOG IMPLIED VOLS
    #log implied vol at t and t-1
    log_iv_t = np.log(surfaces_transform[22:])
    log_iv_tm1 = np.log(surfaces_transform[21:-1])
    #we want to simulate the increment at time t (t - t-1)
    log_iv_inc_t = log_iv_t - log_iv_tm1
    # same here the time series dates are the rows, the COLUMN is a specific underlying asset with specific tenor and maturity 
    
    # SET UP NORMALIZATION PARAMETERS (I think this is used later in model training, it might be useful to keep)
    
    # calculate normalisation parameters in case it is needed
    # underlying returns
    m_ret = np.mean(return_t[0:100], axis=0)
    sigma_ret = np.std(return_t[0:100], axis=0)
    # shape (144,)

    #realised vol
    m_rv = np.mean(realised_vol_tm1[0:100], axis=0)
    sigma_rv = np.std(realised_vol_tm1[0:100], axis=0)
    # shape (144,)

    #log implied vol
    m_liv = np.mean(log_iv_t[0:100], axis=0)
    sigma_liv = np.std(log_iv_t[0:100], axis=0)
    # shape (144,)

    #log implied vol increment
    m_liv_inc = np.mean(log_iv_inc_t[0:100], axis=0)
    sigma_liv_inc = np.std(log_iv_inc_t[0:100], axis=0)
    # shape (144,)

    m_in = np.concatenate((m_ret,m_ret,m_rv,m_liv))
    sigma_in = np.concatenate((sigma_ret,sigma_ret,sigma_rv,sigma_liv))
    
    #the output of the generator is the return of SPX and increment of log-iv
    m_out = np.concatenate((m_ret,m_liv_inc))
    sigma_out = np.concatenate((sigma_ret,sigma_liv_inc))
    
    #condition for generator and discriminator
    condition = np.concatenate((np.expand_dims(return_tm1,axis=2),np.expand_dims(return_tm2,axis=2),np.expand_dims(realised_vol_tm1,axis=2),np.expand_dims(log_iv_tm1, axis=2)),axis=2)
    # shape (434, 144, 4)
    # each asset at each time has the condition vector

    #true: what we are trying to predict, increments at time t
    return_t_annualized = 252 * return_t
    true = np.concatenate((np.expand_dims(return_t_annualized,axis=2),np.expand_dims(log_iv_inc_t, axis=2)),axis=2)
    # shape (434, 144, 2)
    # each asset at each time has the predicted annualized log return and log implied vol increment

    return true, condition, m_in,sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus

def DataTrainTest(datapath,surfacepath, tr, device = 'cpu'):
    """
    function to split the data into train, test
    tr are the proportions to use for testing
    tr is specifically the percentage of data to use for training
    """

    true, condition, m_in,sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus = DataPreprocesssing(datapath, surfacepath)

    data_tt = torch.from_numpy(m_in)
    m_in = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(m_out)
    m_out = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(sigma_in)
    sigma_in = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(sigma_out)
    sigma_out = data_tt.to(torch.float).to(device)

    n = true.shape[0]
    data_tt = torch.from_numpy(true)
    true_tensor = data_tt.to(torch.float).to(device)

    data_tt = torch.from_numpy(condition)
    condition_tensor = data_tt.to(torch.float).to(device)

    true_train = true_tensor[0:int(tr * n), :, :]
    true_test = true_tensor[int(tr * n):, :, :]

    condition_train = condition_tensor[0:int(tr * n), :, :]
    condition_test = condition_tensor[int(tr * n):, :, :]

    return true_train, true_test, condition_train,  condition_test,  m_in,sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus