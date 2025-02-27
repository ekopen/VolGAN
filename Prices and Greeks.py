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

def maturity_tenor(filename):
    mat_n_ten = pd.read_csv(filename, nrows=2)
    new_header = mat_n_ten.iloc[1]
    
    df = pd.read_csv(filename, nrows=2, header=None)
    df.columns = new_header
    df.index = df["Ticker"]
    df = df.iloc[:, 1:]
    df = df.T
    
    df["Tenor"] = 0
    df["Mat"] = 0

    for i in range(len(df)):
        xT,yT = (int(df["TERM (TENOR)"][i][:-1]), df["TERM (TENOR)"][i][-1:])
        xM,yM = (int(df["MATURITY (EXPIRY)"][i][:-1]), df["MATURITY (EXPIRY)"][i][-1:])

        if yT == "M":
            df["Tenor"][i] = xT * 1/12
        elif yT == "Y":
            df["Tenor"][i] = xT

        if yM == "M":
            df["Mat"][i] = xM * 1/12
        elif yM == "Y":
            df["Mat"][i] = xM
    
    df = df[["Tenor", "Mat"]]
    df = df.T
    return df

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
    
    def ns_params(self):
        r = self.df.loc[self.date].dropna()

        if isinstance(r, pd.DataFrame):
            r = r.squeeze()

        time_points = r.index.to_numpy(dtype=float)
        rate_values = r.to_numpy(dtype=float)
        
        initial_guess = [np.mean(rate_values), -1, 1, 2]
        params, _ = curve_fit(self.nelson_siegel, time_points, rate_values, p0=initial_guess)
        
        return params

    def price(self):
        p = 0
        norm_dist = stats.norm()
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        
        params = self.ns_params()
        
        for i in np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25):
            r_interp = self.nelson_siegel(i, *params)

            term1 = (self.F - self.K) * norm_dist.cdf(d)
            term2 = (self.sig * np.sqrt(self.T0)) * norm_dist.pdf(d)

            Z = np.exp(-r_interp * i)
            p += Z * (term1 + term2)
        

        return p
    
    def delta(self):
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        norm_dist = stats.norm()
        params = self.ns_params()
        Zs = 0
        
        for i in np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25):
            r_interp = self.nelson_siegel(i, *params)

            Z = np.exp(-r_interp * i)
            Zs += Z
        
        return Zs * norm_dist.cdf(d)
    
    def gamma(self):
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        norm_dist = stats.norm()
        fd = 1/(self.sig * np.sqrt(self.T0))
        Zs = 0
        
        for i in np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25):
            r_interp = self.nelson_siegel(i, *params)

            Z = np.exp(-r_interp * i)
            Zs += Z
        
        return Zs * fd * norm_dist.pdf(d)
    
    def vega(self):
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        norm_dist = stats.norm()
        fd = np.sqrt(self.T0)
        Zs = 0
        
        for i in np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25):
            r_interp = self.nelson_siegel(i, *params)

            Z = np.exp(-r_interp * i)
            Zs += Z
        
        return Zs * fd * norm_dist.pdf(d)
    
#Assuming n = 4 and that we are paying fixed
class Forward_Swap:
    def __init__(self, df, date, cpn, ytm, T0, Ts):
        self.df = df
        self.date = date
        self.cpn = cpn
        self.ytm = ytm
        self.T0 = T0
        self.Ts = Ts
    
    def nelson_siegel(self, maturity, theta0, theta1, theta2, lambda1):
        rate = theta0 + (theta1 + theta2) * (1 - np.exp(-maturity/lambda1))/(maturity/lambda1) - theta2 * np.exp(-maturity/lambda1)

        return rate
    
    def ns_params(self):
        r = self.df.loc[self.date].dropna()

        if isinstance(r, pd.DataFrame):
            r = r.squeeze()

        time_points = r.index.to_numpy(dtype=float)
        rate_values = r.to_numpy(dtype=float)
        
        initial_guess = [np.mean(rate_values), -1, 1, 2]
        params, _ = curve_fit(self.nelson_siegel, time_points, rate_values, p0=initial_guess)
        
        return params
    
    def forward_floating_bond(self):
        params = self.ns_params()
        r0 = self.nelson_siegel(self.T0, *params)
        Z0 = 1/ (1 + r0/4) ** (self.T0 * 4)
        
        price = 0

        for i in np.arange(1/4, self.T0 + self.Ts + 1/4, 1/4):
            if i > self.T0:
                z = 1 / (1 + self.cpn/4) ** (4 * (i))
                val = self.cpn * 100 * z/4
                price += val

        price += (100) / ((1 + self.cpn/4) ** (4 * (self.T0 + self.Ts)))

        return price/Z0
    
    def forward_bond(self):
        params = self.ns_params()
        r0 = self.nelson_siegel(self.T0, *params)
        Z0 = 1/ (1 + r0/4) ** (self.T0 * 4)
        
        price = 0

        for i in np.arange(1/4, self.T0 + self.Ts + 1/4, 1/4):
            if i > self.T0:
                z = 1 / (1 + self.ytm/4) ** (4 * (i))
                val = self.cpn * 100 * z/4
                price += val

        price += (100) / ((1 + self.ytm/4) ** (4 * (self.T0 + self.Ts)))

        return price/Z0
    
    def price(self):
        fl = self.forward_floating_bond()
        fi = self.forward_bond()
        
        return fl - fi
                
        
        
        

    

filename = "generated_surfaces_test_new.csv"   
    
gen_s = pd.read_csv(filename, skiprows = 2).set_index("Ticker")
forward_swap = pd.read_excel(filepath, skiprows = 2).set_index("Ticker")

mat_n_ten1 = maturity_tenor(filename).T

    
def all_prices(date):
    d1 = pd.DataFrame(forward_swap.loc[date])
    d2 = pd.DataFrame(gen_s.loc[date]).iloc[:-1]

    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]

    df["Vol"] = d2.values
    
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    BM = Bachelier_Model(Z, date, 0, 0, 0, 0, 0)
    lst = []
    
    for i in range(len(df)):
        BM.sig = df["Vol"].iloc[i]/100
        BM.F = df["Forward"].iloc[i]/100
        BM.K = df["Forward"].iloc[i]/100
        BM.T0 = df["Maturity"].iloc[i]
        BM.Ts = df["Tenor"].iloc[i]
        
        lst.append(BM.price())
    
    df["Price"] = lst
    df = df[["Tenor", "Maturity", "Price"]]
    
    return df

def grid_prices(date):
    df = all_prices(date)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Price')
    return grid

#Delta using Bachelier
filename = "swaption_atm_vol_full.xlsx"   
    
atm_vol = pd.read_excel(filename, skiprows = 2).set_index("Ticker")
forward_swap = pd.read_excel(filepath, skiprows = 2).set_index("Ticker")
forward_swap.index = pd.to_datetime(forward_swap.index)

def all_deltas(date):
    d1 = pd.DataFrame(forward_swap.loc[date])
    d2 = pd.DataFrame(atm_vol.loc[date])

    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]

    df["Vol"] = d2.values
    
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    BM = Bachelier_Model(Z, date, 0, 0, 0, 0, 0)
    lst = []
    
    for i in range(len(df)):
        BM.sig = df["Vol"].iloc[i]/100
        BM.F = df["Forward"].iloc[i]/100
        BM.K = df["Forward"].iloc[i]/100
        BM.T0 = df["Maturity"].iloc[i]
        BM.Ts = df["Tenor"].iloc[i]
        
        lst.append(BM.delta())
    
    df["Delta"] = lst
    
    return df

def realized_prices(date):
    d1 = pd.DataFrame(forward_swap.loc[date])
    d2 = pd.DataFrame(atm_vol.loc[date])

    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]

    df["Vol"] = d2.values
    
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    BM = Bachelier_Model(Z, date, 0, 0, 0, 0, 0)
    lst = []
    
    for i in range(len(df)):
        BM.sig = df["Vol"].iloc[i]/100
        BM.F = df["Forward"].iloc[i]/100
        BM.K = df["Forward"].iloc[i]/100
        BM.T0 = df["Maturity"].iloc[i]
        BM.Ts = df["Tenor"].iloc[i]
        
        lst.append(BM.price())
    
    df["Price"] = lst
    
    return df

def underlying_PnL(date):
    i = forward_swap.index.get_loc(date)

    
    d2 = pd.DataFrame(forward_swap.iloc[i, :])
    d1 = pd.DataFrame(forward_swap.iloc[i + 1, :])
    
    df = d1.join(d2)/100
    df.columns = ["CPN", "YTM"]
    d3 = all_deltas(date).iloc[:-1]
    d3.index = df.index
    
    df = d3[["Maturity", "Tenor"]].join(df)
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    FS = Forward_Swap(Z, date, 0, 0, 0, 0)
    ps = []
    
    for i in range(len(df)):
        FS.cpn = df["CPN"].iloc[i]
        FS.ytm = df["YTM"].iloc[i]
        FS.T0 = df["Maturity"].iloc[i]
        FS.Ts = df["Maturity"].iloc[i]
        
        ps.append(FS.price())
    
    df["Swap PnL"] = ps
    
    return df[["Maturity", "Tenor", "Swap PnL"]]

def swaption_change(date):
    i = forward_swap.index.get_loc(date)
    
    d1 = forward_swap.index.values[i]
    d2 = forward_swap.index.values[i+1]
    
    p1 = realized_prices(d1)[["Tenor", "Maturity", "Price"]]
    p1 = p1.rename(columns = {"Price": d1})
    p2 = realized_prices(d2)[["Tenor", "Maturity", "Price"]]
    p2 = p2.rename(columns = {"Price": d2})
    
    df = p1.merge(p2, on=["Tenor", "Maturity"], how="inner")
    df["Swaption PnL"] = df[d1] - df[d2]
    
    return df[["Tenor", "Maturity", "Swaption PnL"]]

def hedge_strat(date):
    d = all_deltas(date)[["Tenor", "Maturity", "Delta"]]
    sc = swaption_change(date)
    up = underlying_PnL(date)
    
    df = d.merge(sc, on=["Tenor", "Maturity"], how="inner")
    df2 = df.merge(up, on=["Tenor", "Maturity"], how="inner")
    
    df2["Swap Pos"] = df2["Delta"] * df2["Swap PnL"]
    df2["Total Pos"] = df2["Swaption PnL"] - df2["Swap Pos"]
    
    return df2[["Tenor", "Maturity", "Total Pos"]]
