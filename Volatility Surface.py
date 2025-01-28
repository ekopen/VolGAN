import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def maturity_tenor(filename):
    mat_n_ten = pd.read_excel(filename, nrows=2)
    new_header = mat_n_ten.iloc[1]
    
    df = pd.read_excel(filename, nrows=2, header=None)
    df.columns = new_header
    df.index = df["Ticker"]
    df = df.iloc[:, 1:]
    df = df.T
    
    df["Tenor"] = 0
    df["Mat"] = 0

    for i in range(len(df)):
        xT,yT = (int(df["TERM (TENOR)"][i][:-1]), df["TERM (TENOR)"][i][-1:])
        xM,yM = (int(df["MATRUITY (EXPIRY)"][i][:-1]), df["MATRUITY (EXPIRY)"][i][-1:])

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

def volatility_data(date, filename = "swaption_atm_vol_full.xlsx"):
    volatilites = pd.read_excel(filename, skiprows = 2).set_index("Ticker")
    mat_n_ten = maturity_tenor(filename).T
    d1 = pd.DataFrame(volatilites.loc[date])
    
    df = d1.join(mat_n_ten)
    df.columns = ["Values", "Tenor", "Maturity"]
    
    return df
    
def tabular_volatility_form(date, filename = "swaption_atm_vol_full.xlsx"):
    df = volatility_data(filename, date)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Values')
    return grid

def scatter_swaption_surface(date, filename = "swaption_atm_vol_full.xlsx"):
    df = volatility_data(filename, date)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df['Tenor']
    y = df['Maturity']
    z = df['Values']
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_xlabel('Tenor')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')

    ax.set_title(f'Swaption Volatility Surface on {date}')
    
    plt.show

def plot_swaption_surface(date, filename = "swaption_atm_vol_full.xlsx"):
    grid = tabular_form(filename, date)

    X, Y = np.meshgrid(grid.columns, grid.index)
    Z = grid.values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    ax.set_xlabel('Maturity')
    ax.set_ylabel('Tenor')
    ax.set_zlabel('Implied Volatility')

    ax.set_title(f'Swaption Volatility Surface on {date}')

    plt.show()
    
