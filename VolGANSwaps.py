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

# DATA PROCESSING

def SwapsData(datapath, surfacespath):
    """
    function to read the pre-processed Swaptions implied vol data
    """

    # SETUP LOG RETURNS ON UNDERLYING 
    log_rtn = np.log(pd.read_excel(datapath, skiprows = 2).set_index("Ticker").sort_index()).diff().dropna()
    
    # GET DATES
    dates_dt = log_rtn.index
    
    # PROCESS VOLATILITY DATA
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

    # NOTE: each COLUMN in log_rtn is the time series of realised vol of a specific underlying asset with specific tenor and maturity
    # Asset is in each ROW 
    # shape (456, 144)
    # Same case for surfaces_transform
    # also shape (456, 144)
    return surfaces_transform, log_rtn, tenor, tau, tenors, taus, dates_dt

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
    surfaces_transform, log_rtn, tenor, tau, tenors, taus, dates_dt = SwapsData(datapath,surfacepath)

    # PRODUCE REALIZED VOLATILITIES
    # Our log returns are across different tenors and maturities (as if we have more than one underlying asset compared to just SPX)
    # The realized volatilities we produce will actually be a matrix 

    #Realised volatility at time t-1
    realised_vol_tm1 = np.zeros((log_rtn.shape[0]-22, log_rtn.shape[1]))

    for i in range(len(realised_vol_tm1)):
            realised_vol_tm1[i] = np.sqrt(252 / 21) * np.sqrt(log_rtn.iloc[i:(i+21)] ** 2).sum()
    # COLUMN index is a specific asset with some tenor and maturity, row index is the date / time series of realised vol of that asset
    # shape (434, 144)

    #shift the time
    dates_t = dates_dt[22:]
    
    # SEPARATE OUR LOG RETURNS
    #log-return at t, t-1, t-2
    log_rtn_t = log_rtn.iloc[22:].values
    log_rtn_tm1 = np.sqrt(252) * log_rtn.iloc[21:-1].values
    log_rtn_tm2 = np.sqrt(252) * log_rtn.iloc[20:-2].values
    # COLUMN index is a specific asset with some tenor and maturity, row index is the date / time series of realised vol of that asset

    # GET LOG IMPLIED VOLS
    #log implied vol at t and t-1
    log_iv_t = np.log(surfaces_transform[22:])
    log_iv_tm1 = np.log(surfaces_transform[21:-1])
    #we want to simulate the increment at time t (t - t-1)
    log_iv_inc_t = log_iv_t - log_iv_tm1
    # same here the time series dates are the rows, the COLUMN is a specific underlying asset with specific tenor and maturity 
    
    # SET UP NORMALIZATION PARAMETERS (I think this is used later in model training, it might be useful to keep)
    
    #calculate normalisation parameters in case it is needed
    #log-returns of the underlying
    m_ret = np.mean(log_rtn_t[0:100], axis=0)
    sigma_ret = np.std(log_rtn.iloc[0:100].values, axis=0)
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
    condition = np.concatenate((np.expand_dims(log_rtn_tm1,axis=1),np.expand_dims(log_rtn_tm2,axis=1),np.expand_dims(realised_vol_tm1,axis=1),np.expand_dims(log_iv_tm1, axis=1)),axis=1)
    # shape (434, 4, 144)
    # each asset at each time has the condition vector

    #true: what we are trying to predict, increments at time t
    log_rtn_t_ann = np.sqrt(252) * log_rtn_t
    true = np.concatenate((np.expand_dims(log_rtn_t_ann,axis=1),np.expand_dims(log_iv_inc_t, axis=1)),axis=1)
    # shape (434, 2, 144)
    # each asset at each time has the predicted annualized log return and log implied vol increment

    return true, condition, m_in,sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus