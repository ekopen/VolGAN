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
    surfaces_transform = np.empty((returns.shape[0], returns.shape[1]))  # Preallocate memory

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

def DataPreprocesssing(datapath, surfacepath, vol_model='normal'):
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
        realised_vol_tm1[i] = np.sqrt(252/21) * np.sqrt((returns.iloc[i:(i+21)]**2).sum())

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

    # GET IMPLIED VOLS
    #implied vol at t and t-1

    # Note about volatility: if we are using normal model, then we can use the implied vol directly
    # If we are using log-normal model, then we need to convert the implied vol to log implied vol
    if vol_model == 'normal':
        #implied vol at t
        iv_t = surfaces_transform[22:]
        #implied vol at t-1
        iv_tm1 = surfaces_transform[21:-1]
        iv_inc_t = iv_t - iv_tm1
    elif vol_model == 'log':
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

    if vol_model == 'normal':

        # normal implied vol
        m_iv = np.mean(iv_t[0:100], axis=0)
        sigma_iv = np.std(iv_t[0:100], axis=0)

        # normal implied vol increment
        m_iv_inc = np.mean(iv_inc_t[0:100], axis=0)
        sigma_iv_inc = np.std(iv_inc_t[0:100], axis=0)
    
        m_in = np.concatenate((m_ret,m_ret,m_rv,m_iv))
        sigma_in = np.concatenate((sigma_ret,sigma_ret,sigma_rv,sigma_iv))

        m_out = np.concatenate((m_ret,m_iv_inc))
        sigma_out = np.concatenate((sigma_ret,sigma_iv_inc))

        condition = np.concatenate((np.expand_dims(return_tm1,axis=2),np.expand_dims(return_tm2,axis=2),np.expand_dims(realised_vol_tm1,axis=2),np.expand_dims(iv_t, axis=2)),axis=2)
        # shape (434, 144, 4)
        # each asset at each time has the condition vector

    elif vol_model == 'log':

        # log implied vol 
        m_liv = np.mean(log_iv_t[0:100], axis=0)
        sigma_liv = np.std(log_iv_t[0:100], axis=0)
        # shape (144,)

        #log implied vol increment
        m_liv_inc = np.mean(log_iv_inc_t[0:100], axis=0)
        sigma_liv_inc = np.std(log_iv_inc_t[0:100], axis=0)
        # shape (144,)

        m_in = np.concatenate((m_ret,m_ret,m_rv,m_liv))
        sigma_in = np.concatenate((sigma_ret,sigma_ret,sigma_rv,sigma_liv))

        m_out = np.concatenate((m_ret,m_liv_inc))
        sigma_out = np.concatenate((sigma_ret,sigma_liv_inc))
        
        condition = np.concatenate((np.expand_dims(return_tm1,axis=2),np.expand_dims(return_tm2,axis=2),np.expand_dims(realised_vol_tm1,axis=2),np.expand_dims(log_iv_t, axis=2)),axis=2)    
        # shape (434, 144, 4)
        # each asset at each time has the condition vector

    #true: what we are trying to predict, increments at time t
    return_t_annualized = 252 * return_t

    if vol_model == 'normal':
        true = np.concatenate((np.expand_dims(return_t_annualized,axis=2),np.expand_dims(iv_inc_t, axis=2)),axis=2)
    elif vol_model == 'log':
        true = np.concatenate((np.expand_dims(return_t_annualized,axis=2),np.expand_dims(log_iv_inc_t, axis=2)),axis=2)
    
    # shape (434, 144, 2)
    # each asset at each time has the predicted annualized return and (normal or log) implied vol increment

    return true, condition, m_in, sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus

def DataTrainTest(datapath,surfacepath, tr, vol_model = 'normal', device = 'cpu'):
    """
    function to split the data into train, test
    tr are the proportions to use for testing
    tr is specifically the percentage of data to use for training
    """

    true, condition, m_in,sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus = DataPreprocesssing(datapath, surfacepath, vol_model=vol_model)

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

##############################
##### MODEL ARCHITECTURE #####
##############################

class Generator(nn.Module):
    '''
    VolGAN generator
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
        output_dim: output dimension, a scalar
    '''
    def __init__(self, noise_dim,cond_dim, hidden_dim,output_dim, 
                 mean_in = False, std_in = False, mean_out = False, std_out = False):
        
        super(Generator, self).__init__()

        self.input_dim = noise_dim+cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        self.mu_i = mean_in
        self.std_i = std_in
        self.mu_o = mean_out
        self.std_o = std_out

        #Add the modules
   
        self.linear1 = nn.Linear(in_features = self.input_dim, out_features = self.hidden_dim)
        self.linear2 = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim * 2)
        self.linear3 = nn.Linear(in_features = self.hidden_dim * 2, out_features = self.output_dim)
        self.activation1 = nn.Softplus()
        self.activation2 = nn.Softplus()
        self.activation3 = nn.Sigmoid()
       

    def forward(self, noise, condition):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        #x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        #condition: S_t-1, sigma_t-1, r_t-1, implied vol_t-1
        #out: increment in r_t, increment in implied vol _t
        
        # condition = (condition - self.mu_i) / self.std_i

        out = torch.cat([noise,condition],dim=-1).to(torch.float)
        # out = self.linear1(out)
        # out = self.activation1(out)
        # out = self.linear2(out)
        # out = self.activation2(out)
        # out = self.linear3(out)

        out = self.linear1(out)
        out = self.activation1(out)
        out = self.linear2(out)
        out = self.activation2(out)
        out = self.linear3(out)

        #uncomment to normalise
        # out = self.mu_o + self.std_o * out
        # out = torch.max(out,torch.tensor(10**(-5)))
        
        return out

class Discriminator(nn.Module):
    '''
    VolGAN discriminator
      in_dim: the input dimension (concatenated with the condition), a scalar
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, in_dim, hidden_dim, mean = False, std = False):
        super(Discriminator, self).__init__()
        self.input_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_features=self.input_dim, out_features= self.hidden_dim)
        self.Softplus = nn.Softplus()
        self.linear2 = nn.Linear(in_features = self.hidden_dim, out_features = 1)
        self.sigmoid = nn.Sigmoid()

        self.mu_i = mean
        self.std_i = std


    def forward(self, in_chan):
        '''
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        '''
        x = in_chan
        #uncomment to normalise
        # x = (x - self.mu_i) / self.std_i
        out = self.linear1(x)
        out = self.Softplus(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out

##############################
##### RUNNING THE MODEL #####
##############################

def VolGAN(datapath, surfacepath, tr, vol_model = 'normal',
           noise_dim = 16, hidden_dim = 8, 
           n_epochs = 1000,n_grad = 100, 
           lrg = 0.0001, lrd = 0.0001, 
           batch_size = 100, device = 'cpu'):
   
    true_train, true_test, condition_train, condition_test,  m_in,sigma_in, m_out, sigma_out, dates_t,  tenor, tau, tenors, taus = DataTrainTest(datapath,surfacepath, tr, vol_model, device)
    gen = Generator(noise_dim=noise_dim,cond_dim=condition_train.shape[2], hidden_dim=hidden_dim,output_dim=true_train.shape[2],mean_in = m_in, std_in = sigma_in, mean_out = m_out, std_out = sigma_out)
    gen.to(device)
    
    # m_disc and sigma_disc are not used in the original VolGAN, you can see in the Discriminator forward function 
    # we'll preserve the forward pass for now but we can maybe incorporate these later if needed    
    m_disc = torch.cat((m_in,m_out),dim=-1)
    sigma_disc = torch.cat((sigma_in,sigma_out),dim=-1)

    disc = Discriminator(in_dim = condition_train.shape[2] + true_train.shape[2], hidden_dim = hidden_dim, mean = m_disc, std = sigma_disc)
    disc.to(device)
    
    true_val = False
    condition_val = False
    
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)
    
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    
    gen,gen_opt,disc,disc_opt,criterion, alpha, beta = GradientMatching(gen,gen_opt,disc,disc_opt,criterion,condition_train,true_train,tenor,tau,tenors,taus,n_grad,lrg,lrd,batch_size,noise_dim,device, vol_model=vol_model)
    gen,gen_opt,disc,disc_opt,criterion = TrainLoopNoVal(alpha,beta,gen,gen_opt,disc,disc_opt,criterion,condition_train,true_train,tenor,tau,tenors,taus,n_epochs,lrg,lrd,batch_size,noise_dim,device)
    
    return gen, gen_opt, disc, disc_opt, true_train, true_val, true_test, condition_train, condition_val, condition_test, dates_t,  tenor, tau, tenors, taus

def GradientMatching(gen,gen_opt,disc,disc_opt,criterion,
                     condition_train,true_train,
                     tenor,tau,tenors,taus,
                     n_grad,lrg,lrd,batch_size,noise_dim,
                     device, lk = 10, lt = 8, vol_model = 'normal'):
    """
    perform gradient matching
    """

    n_train = condition_train.shape[0]
    underlying_dim = condition_train.shape[1]
    n_batches =  n_train // batch_size + 1
    dtm = tau * 365

    #mP_t,mP_k,mPb_K = penalty_mutau_tensor(m,dtm,device)
    
    tenor_t = torch.tensor(tenor,dtype=torch.float,device=device)
    
    #smoothness penalties
    Ngrid = lk * lt
    tau_t = torch.tensor(tau,dtype=torch.float,device=device)
    t_seq = torch.zeros((tau_t.shape[0]),dtype=torch.float,device=device)
    for i in range(tau_t.shape[0]-1):
        t_seq[i] = 1/((tau_t[i+1]-tau_t[i])**2)
    matrix_t = torch.zeros((Ngrid,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-1):
        matrix_t[i,i] = -1
        matrix_t[i,i+1] = 1
    tsq = t_seq.repeat(lk).unsqueeze(0)
    matrix_tenor = torch.zeros((Ngrid-lk,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-lk):
        matrix_tenor[i,i] = -1
        matrix_tenor[i,i+lk] = 1
        
    m_seq = torch.zeros((lk*(lt-1)),dtype=torch.float,device=device)
    for i in range(tenor_t.shape[0]-1):
        m_seq[i*lk:(i+1)*lk] = 1/((tenor_t[i+1]-tenor_t[i])**2)
    
    n_epochs = n_grad
    discloss = [False] * (n_batches*n_epochs)
    genloss = [False] * (n_batches*n_epochs)
    dscpred_real = [False] * (n_batches*n_epochs)
    dscpred_fake = [False] * (n_batches*n_epochs)
    gen_fake = [False] * (n_batches*n_epochs)
    genprices_fk = [False] * (n_batches*n_epochs)
    BCE_grad = []
    m_smooth_grad = []
    t_smooth_grad = []
    gen.train()

    for epoch in tqdm(range(n_epochs)):

        perm = torch.randperm(n_train)
        condition_train = condition_train[perm,:]
        true_train = true_train[perm,:]

        for i in range(n_batches):
            
            print("Epoch: ", epoch, "Batch: ", i)

            curr_batch_size = batch_size
            
            if i==(n_batches-1):
                curr_batch_size = n_train-i*batch_size
            
            condition = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),:, :]
            
            surface_past = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),:,3:]

            real = true_train[(i*batch_size):(i*batch_size+curr_batch_size),:,:]

            real_and_cond = torch.cat((condition,real),dim=-1)
            #update the discriminator
            disc_opt.zero_grad()

            noise = torch.randn((curr_batch_size, underlying_dim, noise_dim), device=device,dtype=torch.float)

            fake = gen(noise,condition) # last layer is sigmoid so fake values are all between 0 and 1
            fake_and_cond = torch.cat((condition,fake),dim=-1) 

            disc_fake_pred = disc(fake_and_cond.detach()) 
            # last layer is sigmoid so fake values should all be between 0 and 1 
            # #oh there are nan values for some reason maybe propogating from generator
            disc_real_pred = disc(real_and_cond)

            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()
            
            # in the original VolGAN the prediction had shape (batch size, 1) 
            # now with tensors it doesn't really like this dimensionality 

            # dscpred_real[epoch*n_batches+i] = disc_real_pred[0].detach().item()
            # dscpred_fake[epoch*n_batches+i] = disc_fake_pred[0].detach().item()
            
            # discloss[epoch*n_batches+i] = disc_loss.detach().item()
            
            # these don't seem to be working right now but might be a useful metric for evaluation later
            # will comment these out for now

            #update the generator
            gen_opt.zero_grad()
            noise = torch.randn((curr_batch_size, underlying_dim, noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)

            fake_and_cond = torch.cat((condition,fake),dim=-1) 
            disc_fake_pred = disc(fake_and_cond)
            
            if vol_model == 'normal':
                fake_surface = fake[:,:,1:]+ surface_past
            elif vol_model == 'log':
                fake_surface = torch.exp(fake[:,:,1:]+ surface_past)

            # TRAINING LOOP IS WORKING UP UNTIL HERE
            # BEFORE THERE WERE NAN PROBLEMS SO THAT THE LOSS COULD NOT BE CALCULATED

            penalties_tenor = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_tenor[iii] = torch.matmul(m_seq,(torch.matmul(matrix_tenor,fake_surface[iii])**2))
                penalties_t[iii] = torch.matmul(tsq,(torch.matmul(matrix_t,fake_surface[iii])**2))
            m_penalty = sum(penalties_tenor) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size
            
            m_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            m_smooth_grad.append(total_norm)
            gen_opt.zero_grad()
            
            t_penalty.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            t_smooth_grad.append(total_norm)
            
            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            BCE_grad.append(total_norm)
            gen_opt.step()
            genloss[epoch*n_batches+i] = gen_loss.detach().item()
            gen_fake[epoch*n_batches+i] = fake[0].detach()
            genprices_fk[epoch*n_batches+i]= condition[0].detach()
        
            
    alpha = np.mean(np.array(BCE_grad) / np.array(m_smooth_grad))
    beta = np.mean(np.array(BCE_grad) / np.array(t_smooth_grad))
    print("alpha :", alpha, "beta :", beta)
    return gen,gen_opt,disc,disc_opt,criterion, alpha, beta

def TrainLoopNoVal(alpha,beta,gen,gen_opt,disc,disc_opt,criterion,condition_train,true_train,tenor,tau,tenors,taus,n_epochs,lrg,lrd,batch_size,noise_dim,device, lk = 10, lt = 8):
    """
    train loop for VolGAN
    """
    n_train = condition_train.shape[0]
    n_batches =  n_train // batch_size + 1
    dtm = tau * 365
    #mP_t,mP_k,mPb_K = penalty_mutau_tensor(m,dtm,device)
    
    tenor_t = torch.tensor(tenor,dtype=torch.float,device=device)
    #smoothness penalties
    Ngrid = lk * lt
    tau_t = torch.tensor(tau,dtype=torch.float,device=device)
    t_seq = torch.zeros((tau_t.shape[0]),dtype=torch.float,device=device)
    for i in range(tau_t.shape[0]-1):
        t_seq[i] = 1/((tau_t[i+1]-tau_t[i])**2)
    matrix_t = torch.zeros((Ngrid,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-1):
        matrix_t[i,i] = -1
        matrix_t[i,i+1] = 1
    tsq = t_seq.repeat(lk).unsqueeze(0)
    matrix_tenor = torch.zeros((Ngrid-lk,Ngrid), device = device, dtype = torch.float)
    for i in range(Ngrid-lk):
        matrix_tenor[i,i] = -1
        matrix_tenor[i,i+lk] = 1
        
    m_seq = torch.zeros((lk*(lt-1)),dtype=torch.float,device=device)
    for i in range(tenor_t.shape[0]-1):
        m_seq[i*lk:(i+1)*lk] = 1/((tenor_t[i+1]-tenor_t[i])**2)
    
    discloss = [False] * (n_batches*n_epochs)
    genloss = [False] * (n_batches*n_epochs)
    dscpred_real = [False] * (n_batches*n_epochs)
    dscpred_fake = [False] * (n_batches*n_epochs)
    gen_fake = [False] * (n_batches*n_epochs)
    genprices_fk = [False] * (n_batches*n_epochs)

    gen.train()
    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(n_train)
        condition_train = condition_train[perm,:]
        true_train = true_train[perm,:]
        for i in range(n_batches):
            curr_batch_size = batch_size
            if i==(n_batches-1):
                curr_batch_size = n_train-i*batch_size
            condition = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),:]
            surface_past = condition_train[(i*batch_size):(i*batch_size+curr_batch_size),3:]
            real = true_train[(i*batch_size):(i*batch_size+curr_batch_size),:]

            real_and_cond = torch.cat((condition,real),dim=-1)
            #update the discriminator
            disc_opt.zero_grad()
            noise = torch.randn((curr_batch_size,noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)
            fake_and_cond = torch.cat((condition,fake),dim=-1)

            disc_fake_pred = disc(fake_and_cond.detach())
            disc_real_pred = disc(real_and_cond)
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()
            
            dscpred_real[epoch*n_batches+i] = disc_real_pred[0].detach().item()
            dscpred_fake[epoch*n_batches+i] = disc_fake_pred[0].detach().item()
            
            discloss[epoch*n_batches+i] = disc_loss.detach().item()
            
            #update the generator
            gen_opt.zero_grad()
            noise = torch.randn((curr_batch_size,noise_dim), device=device,dtype=torch.float)
            fake = gen(noise,condition)

            fake_and_cond = torch.cat((condition,fake),dim=-1)
            
            disc_fake_pred = disc(fake_and_cond)
            
            # fake_surface = torch.exp(fake[:,1:]+ surface_past)
            fake_surface = fake[:,1:]+ surface_past

            penalties_m = [None] * curr_batch_size
            penalties_t = [None] * curr_batch_size
            for iii in range(curr_batch_size):
                penalties_m[iii] = torch.matmul(m_seq,(torch.matmul(matrix_tenor,fake_surface[iii])**2))
                penalties_t[iii] = torch.matmul(tsq,(torch.matmul(matrix_t,fake_surface[iii])**2))
            m_penalty = sum(penalties_m) / curr_batch_size
            t_penalty = sum(penalties_t) / curr_batch_size
            
            gen_opt.zero_grad()
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) + alpha * m_penalty + beta * t_penalty
            gen_loss.backward()
            gen_opt.step()
            genloss[epoch*n_batches+i] = gen_loss.detach().item()
            gen_fake[epoch*n_batches+i] = fake[0].detach()
            genprices_fk[epoch*n_batches+i]= condition[0].detach()
            
        
    return gen,gen_opt,disc,disc_opt,criterion

##############################
##### PENALTY FUNCTIONS #####
##############################

def penalty_mutau_tensor(mu,T,device):
    return penalty_tensor(mu,T,device)

def penalty_tensor(K,T,device):
    """
    matrices for calculating the arbitrage penalty (tensors)
    """
    P_T = torch.zeros(size=(len(T),len(T)),dtype=torch.float,device = device)
    P_K = torch.zeros(size=(len(K),len(K)),dtype=torch.float,device = device)
    PB_K = torch.zeros(size=(len(K),len(K)),dtype=torch.float,device = device)
    #P_T first, the last one is zero
    for j in tqdm(np.arange(0,len(T)-1,1)):
        P_T[j,j] = T[j]/(T[j+1]-T[j])
        P_T[j+1,j] = -T[j]/(T[j+1]-T[j])
    #now P_K and then PB_K
    for i in tqdm(np.arange(0,len(K)-1,1)):
        P_K[i,i] = -1/(K[i+1]-K[i])
        P_K[i,i+1] = 1/(K[i+1]-K[i])
    #PB_K: note that it is a scaled finite difference, but let's compute it on its own just in case
    #once we fix the grid we have to run this function only once so it doesn't matter much
    for i in tqdm(np.arange(1,len(K)-1,1)):
        PB_K[i,i-1] =  -(K[i+1]-K[i]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
        PB_K[i,i] = (K[i+1] - K[i-1]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
        PB_K[i,i+1] = -(K[i]-K[i-1]) / ((K[i]-K[i-1]) * (K[i+1]-K[i]))
    return P_T,P_K,PB_K