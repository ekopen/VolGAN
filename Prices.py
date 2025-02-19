import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import fsolve
warnings.filterwarnings("ignore")
import Inputs

import sys
sys.path.insert(0, '../cmds')
from treasury_cmds import *


def data_prep(filename):
    usd_spot = pd.read_excel(filename).iloc[1:, :].set_index("TENOR")

    lst = []

    for i in usd_spot.columns.values:
        val = i[:-1]
        term = i[-1:]

        if term == "M":
            lst.append(int(val) * 1/12)
        elif term == "Y":
            lst.append(int(val) * 1)

    usd_spot.columns = lst
    usd_spot = usd_spot/100
    return usd_spot

class Bachelier_Model:
    def __init__(self, r, date, T0, Ts, sig, K, F):
        self.df = r
        self.date = date
        self.sig = sig
        self.K = K
        self.F = F
        self.T0 = T0
        self.Ts = Ts
        
    def nelson_siegel(self, maturity, theta0, theta1, theta2, lambda1):
        rate = theta0 + (theta1 + theta2) * (1 - np.exp(-maturity/lambda1))/(maturity/lambda1) - theta2 * np.exp(-maturity/lambda1)

        return rate

    def price(self):
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        p = 0
        norm_dist = stats.norm()
        
        r = self.df.loc[self.date]

        if isinstance(r, pd.DataFrame):
            r = r.squeeze()

        time_points = r.index.to_numpy(dtype=float)
        rate_values = r.to_numpy(dtype=float)
        
        initial_guess = [np.mean(rate_values), -1, 1, 2]
        params, _ = curve_fit(self.nelson_siegel, time_points, rate_values, p0=initial_guess)

        for i in np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25):
            r_interp = self.nelson_siegel(i, *params)

            term1 = (self.F - self.K) * norm_dist.cdf(d)
            term2 = (self.sig * np.sqrt(self.T0)) * norm_dist.pdf(d)

            Z = np.exp(-r_interp * i)
            p += Z * (term1 + term2)

        return p


datapath = "swaption_atm_vol_full.xlsx"

mat_n_ten1 = Inputs.maturity_tenor("data/forward_sofr_swap_full_NEW.xlsx").T

#gen_s = pd.read_csv("generated_surfaces.csv", skiprows = 2).iloc[1:, :].set_index("Ticker")

forward_swap = pd.read_excel("data/forward_sofr_swap_full_NEW.xlsx", skiprows = 2).set_index("Ticker")
    
def all_prices(date, gen_s):
    d1 = pd.DataFrame(forward_swap.loc[date])
    d2 = pd.DataFrame(gen_s.loc[date])
    
    df = d1.join(mat_n_ten1)
    df.columns = ["Forward", "Tenor", "Maturity"]
    df = df.loc[~(df["Tenor"] == 30)]
    
    df["Vol"] = d2.values
    
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    BM = Bachelier_Model(Z, date, 0, 0, 0, 0, 0)
    lst = []
    
    for i in range(len(df)):
        BM.sig = df["Vol"].iloc[i]/100
        BM.F = df["Forward"].iloc[i]
        BM.K = df["Forward"].iloc[i]
        BM.T0 = df["Maturity"].iloc[i]
        BM.Ts = df["Tenor"].iloc[i]
        
        lst.append(BM.price())
    
    df["Price"] = lst
    df = df[["Tenor", "Maturity", "Price"]]
    
    return df

def grid_prices(date, gen_s):
    df = all_prices(date, gen_s)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Price')
    return grid

def arbitrage(date, gen_s):
    df = grid_prices(date, gen_s)
    violations_down = (df.diff(axis=0) < 0)
    violations_right = (df.diff(axis=1) < 0)

    violations_combined = violations_down | violations_right
    
    s1 = violations_combined.sum().sum()
    return s1/violations_combined.size

def total_penalty(gen_s):
    s1 = 0
    
    for i in gen_s.index:
        s1 += arbitrage(i)
    
    return s1/len(gen_s.index)
