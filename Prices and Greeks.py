import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings
import scipy.stats as stats
from scipy.optimize import curve_fit, fsolve
warnings.filterwarnings("ignore")
import Inputs

import sys
sys.path.insert(0, '../cmds')
from treasury_cmds import *


def data_prep(filename):
    df = pd.read_excel(filename).iloc[1:, :].set_index("TENOR")
    df.columns = [int(col[:-1]) * (1/12 if col[-1] == "M" else 1)
                  for col in df.columns]
    return df/100

def maturity_tenor(filename):
    header_df = pd.read_csv(filename, nrows=2)
    new_header = header_df.iloc[1]
    
    df = pd.read_csv(filename, nrows=2, header=None)
    df.columns = new_header
    df.index = df["Ticker"]
    df = df.iloc[:, 1:].T
    
    tenor_extract = df["TERM (TENOR)"].str.extract(r'(?P<val>\d+)(?P<unit>[MY])')
    mat_extract = df["MATURITY (EXPIRY)"].str.extract(r'(?P<val>\d+)(?P<unit>[MY])')
    
    df["Tenor"] = tenor_extract["val"].astype(float) * tenor_extract["unit"].map({"M": 1/12, "Y": 1})
    df["Mat"] = mat_extract["val"].astype(float) * mat_extract["unit"].map({"M": 1/12, "Y": 1})
    
    return df[["Tenor", "Mat"]].T


class Bachelier_Model:
    def __init__(self, r, date, T0, Ts, sig, K, F):
        self.df = r
        self.date = date
        self.sig = sig
        self.K = K
        self.F = F
        self.T0 = T0
        self.Ts = Ts
        self.params = self.ns_params()
    
    @staticmethod
    def nelson_siegel(maturity, theta0, theta1, theta2, lambda1):
        return theta0 + (theta1 + theta2) * (1 - np.exp(-maturity/lambda1))/(maturity/lambda1) - theta2 * np.exp(-maturity/lambda1)
    
    def ns_params(self):
        r_series = self.df.loc[self.date].dropna()
        if isinstance(r_series, pd.DataFrame):
            r_series = r_series.squeeze()
        time_points = r_series.index.to_numpy(dtype=float)
        rate_values = r_series.to_numpy(dtype=float)
        initial_guess = [np.mean(rate_values), -1, 1, 2]
        params, _ = curve_fit(self.nelson_siegel, time_points, rate_values, p0=initial_guess)
        return params

    def price(self):
        norm_dist = stats.norm()
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        times = np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25)
        r_interp = self.nelson_siegel(times, *self.params)
        Z = np.exp(-r_interp * times)
        term1 = (self.F - self.K) * norm_dist.cdf(d)
        term2 = (self.sig * np.sqrt(self.T0)) * norm_dist.pdf(d)
        return np.sum(Z * (term1 + term2))
    
    def delta(self):
        norm_dist = stats.norm()
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        times = np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25)
        r_interp = self.nelson_siegel(times, *self.params)
        Z = np.exp(-r_interp * times)
        Zs = np.sum(Z)
        return norm_dist.cdf(d) * Zs
    
    def gamma(self):
        norm_dist = stats.norm()
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        fd = 1/(self.sig * np.sqrt(self.T0))
        times = np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25)
        r_interp = self.nelson_siegel(times, *self.params)
        Z = np.exp(-r_interp * times)
        Zs = np.sum(Z)
        return fd * norm_dist.pdf(d) * Zs
    
    def vega(self):
        norm_dist = stats.norm()
        d = (self.F - self.K) / (self.sig * np.sqrt(self.T0))
        fd = np.sqrt(self.T0)
        times = np.arange(self.T0 + 0.25, self.Ts + self.T0 + 0.25, 0.25)
        r_interp = self.nelson_siegel(times, *self.params)
        Z = np.exp(-r_interp * times)
        Zs = np.sum(Z)
        return fd * norm_dist.pdf(d)*Zs

class Forward_Swap:
    def __init__(self, df, date, cpn, ytm, T0, Ts):
        self.df = df
        self.date = date
        self.cpn = cpn
        self.ytm = ytm
        self.T0 = T0
        self.Ts = Ts
        self.params = self.ns_params()
    
    @staticmethod
    def nelson_siegel(maturity, theta0, theta1, theta2, lambda1):
        return theta0 + (theta1 + theta2) * (1 - np.exp(-maturity/lambda1))/(maturity/lambda1) - theta2 * np.exp(-maturity/lambda1)
    
    def ns_params(self):
        r_series = self.df.loc[self.date].dropna()
        if isinstance(r_series, pd.DataFrame):
            r_series = r_series.squeeze()
        time_points = r_series.index.to_numpy(dtype=float)
        rate_values = r_series.to_numpy(dtype=float)
        initial_guess = [np.mean(rate_values), -1, 1, 2]
        params, _ = curve_fit(self.nelson_siegel, time_points, rate_values, p0=initial_guess)
        return params
    
    def forward_floating_bond(self):
        r0 = self.nelson_siegel(self.T0, *self.params)
        Z0 = 1 / (1 + r0/4) ** (self.T0 * 4)
        times = np.arange(1/4, self.T0 + self.Ts + 1/4, 1/4)
        mask = times > self.T0
        z = 1 / (1 + self.cpn/4) ** (4 * times[mask])
        price = np.sum(self.cpn * 100 * z / 4)
        price += 100 / ((1 + self.cpn/4) ** (4 * (self.T0 + self.Ts)))
        return price / Z0
    
    def forward_bond(self):
        r0 = self.nelson_siegel(self.T0, *self.params)
        Z0 = 1 / (1 + r0/4) ** (self.T0 * 4)
        times = np.arange(1/4, self.T0 + self.Ts + 1/4, 1/4)
        mask = times > self.T0
        z = 1 / (1 + self.ytm/4) ** (4 * times[mask])
        price = np.sum(self.cpn * 100 * z / 4)
        price += 100 / ((1 + self.ytm/4) ** (4 * (self.T0 + self.Ts)))
        return price / Z0
    
    def price(self):
        return self.forward_floating_bond() - self.forward_bond()


filename = "generated_surfaces_test_new.csv"
filepath = "forward_sofr_swap_full.xlsx"
gen_s = pd.read_csv(filename, skiprows=2).set_index("Ticker")
forward_swap = pd.read_excel(filepath, skiprows=2).set_index("Ticker")
returns = pd.read_excel("generated_returns.xlsx", skiprows=2).set_index("Ticker").iloc[1:, :]/100
mat_n_ten1 = maturity_tenor(filename).T

def new_prices(date):
    d2 = pd.DataFrame(gen_s.loc[date]).iloc[:-1]
    date1 = pd.to_datetime(date)
    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]
    df["Vol"] = d2.values
    
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    def calc_price(row):
        bm = Bachelier_Model(Z, date,
                             T0=row["Maturity"],
                             Ts=row["Tenor"],
                             sig=row["Vol"]/100,
                             K=row["Forward"]/100,
                             F=row["Forward"]/100)
        return bm.price()
    
    df["New Price"] = df.apply(calc_price, axis=1)
    df = df[["Tenor", "Maturity", "New Price"]]
    df2 = realized_prices(date)
    mdf = df2.join(df[["New Price"]]).dropna()
    mdf["Pred PnL"] = mdf["New Price"] - mdf["Price"]
    return mdf[["Tenor", "Maturity", "Pred PnL"]]

def all_prices(date):
    r = pd.DataFrame(returns.loc[pd.to_datetime(date)]).iloc[:-1]
    d2 = pd.DataFrame(gen_s.loc[date]).iloc[:-1]
    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]
    df["Vol"] = d2.values
    df = df.join(r).dropna()
    
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    def calc_price(row):
        bm = Bachelier_Model(Z, date,
                             T0=row["Maturity"],
                             Ts=row["Tenor"],
                             sig=row["Vol"]/100,
                             K=row["Forward"]/100,
                             F=row["Forward"]/100 + row[pd.to_datetime(date)])
        return bm.price()
    
    df["Price"] = df.apply(calc_price, axis=1)
    return df[["Tenor", "Maturity", "Price"]]

def grid_prices(date):
    df = all_prices(date)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Price')
    return grid

def all_deltas(date):
    d2 = pd.DataFrame(atm_vol.loc[date])
    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]
    df["Vol"] = d2.values
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    def calc_delta(row):
        bm = Bachelier_Model(Z, date,
                             T0=row["Maturity"],
                             Ts=row["Tenor"],
                             sig=row["Vol"]/100,
                             K=row["Forward"]/100,
                             F=row["Forward"]/100)
        return bm.delta()
    
    df["Delta"] = df.apply(calc_delta, axis=1)
    return df[["Tenor", "Maturity", "Delta"]]

def underlying_PnL(date):
    idx = forward_swap.index.get_loc(date)
    d2 = pd.DataFrame(forward_swap.iloc[idx, :])
    d1 = pd.DataFrame(forward_swap.iloc[idx + 1, :])
    df = d1.join(d2)/100
    df.columns = ["CPN", "YTM"]
    d3 = all_deltas(date).iloc[:-1]
    d3.index = df.index
    df = d3[["Maturity", "Tenor"]].join(df)
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    FS = Forward_Swap(Z, date, 0, 0, 0, 0)
    def calc_swap(row):
        fs = Forward_Swap(Z, date,
                          cpn=row["CPN"],
                          ytm=row["YTM"],
                          T0=row["Maturity"],
                          Ts=row["Maturity"])
        return fs.price()
    
    df["Swap PnL"] = df.apply(calc_swap, axis=1)
    return df[["Maturity", "Tenor", "Swap PnL"]]

def realized_prices(date):
    d2 = pd.DataFrame(atm_vol.loc[date])
    df = d2.copy()
    df[["Maturity", "Tenor"]] = mat_n_ten1[["Mat", "Tenor"]]
    df.columns = ["Forward", "Tenor", "Maturity"]
    df["Vol"] = d2.values
    Z = data_prep("usd_sofr_curve_full.xlsx")
    
    def calc_price(row):
        bm = Bachelier_Model(Z, date,
                             T0=row["Maturity"],
                             Ts=row["Tenor"],
                             sig=row["Vol"]/100,
                             K=row["Forward"]/100,
                             F=row["Forward"]/100)
        return bm.price()
    
    df["Price"] = df.apply(calc_price, axis=1)
    return df

def swaption_change(date):
    i = forward_swap.index.get_loc(date)
    d1 = forward_swap.index.values[i]
    d2 = forward_swap.index.values[i+1]
    
    p1 = realized_prices(d1)[["Tenor", "Maturity", "Price"]].rename(columns={"Price": d1})
    p2 = realized_prices(d2)[["Tenor", "Maturity", "Price"]].rename(columns={"Price": d2})
    df = p1.merge(p2, on=["Tenor", "Maturity"], how="inner")
    df["Swaption PnL"] = df[d1] - df[d2]
    return df[["Tenor", "Maturity", "Swaption PnL"]]

def hedge_strat(date):
    d = all_deltas(date)[["Tenor", "Maturity", "Delta"]]
    sc = swaption_change(date)
    up = underlying_PnL(date)
    df = d.merge(sc, on=["Tenor", "Maturity"], how="inner")
    df2 = df.merge(up, on=["Tenor", "Maturity"], how="inner")
    df2["Swap Pos"] = df2["Delta"] * df2["Swaption PnL"]
    df2["Total Pos"] = df2["Swaption PnL"] - df2["Swap Pos"]
    return df2[["Tenor", "Maturity", "Total Pos"]]

def pred_strat(date):
    sc = swaption_change(date)
    pc = new_prices(date)
    df = pc.merge(sc, on=["Tenor", "Maturity"], how="inner")
    df["Total Pos"] = df["Swaption PnL"] - df["Pred PnL"]
    return df[["Tenor", "Maturity", "Total Pos"]]
