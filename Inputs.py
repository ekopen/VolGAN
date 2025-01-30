import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Helper Functions

def read_excel_default(excel_name: str, index_col : int = 0, parse_dates: bool =True, print_sheets: bool = False, sheet_name: str = None, **kwargs):
    """
    Reads an Excel file and returns a DataFrame with specified options.

    Parameters:
    excel_name (str): The path to the Excel file.
    index_col (int, default=0): Column to use as the row labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_sheets (bool, default=False): If True, prints the names and first few rows of all sheets.
    sheet_name (str or int, default=None): Name or index of the sheet to read. If None, reads the first sheet.
    **kwargs: Additional arguments passed to `pd.read_excel`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the specified Excel sheet.

    Notes:
    - If `print_sheets` is True, the function will print the names and first few rows of all sheets and return None.
    - The function ensures that the index name is set to 'date' if the index column name is 'date' or 'dates', or if the index contains date-like values.
    """
    if print_sheets:
        n = 0
        while True:
            try:
                sheet = pd.read_excel(excel_name, sheet_name=n)
                print(f'Sheet {n}:')
                print(", ".join(list(sheet.columns)))
                print(sheet.head(3))
                n += 1
                print('\n' * 2)
            except:
                return
    sheet_name = 0 if sheet_name is None else sheet_name
    returns = pd.read_excel(excel_name, index_col=index_col, parse_dates=parse_dates,  sheet_name=sheet_name, **kwargs)
    if returns.index.name is not None:
        if returns.index.name.lower() in ['date', 'dates']:
            returns.index.name = 'date'
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = 'date'
    return returns


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
    
"""
Section 1: Volatility Surface
"""

def volatility_data(date, filename = "swaption_atm_vol_full.xlsx"):
    volatilites = pd.read_excel(filename, skiprows = 2).set_index("Ticker")
    mat_n_ten = maturity_tenor(filename).T
    d1 = pd.DataFrame(volatilites.loc[date])
    
    df = d1.join(mat_n_ten)
    df.columns = ["Values", "Tenor", "Maturity"]
    
    return df
    
def tabular_volatility_form(date, filename = "swaption_atm_vol_full.xlsx"):
    df = volatility_data(date, filename)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Values')
    return grid

def scatter_swaption_surface(date, filename = "swaption_atm_vol_full.xlsx"):
    df = volatility_data(date, filename)
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
    
    plt.show()

def plot_swaption_surface(date, filename = "swaption_atm_vol_full.xlsx"):
    grid = tabular_volatility_form(date, filename)

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

"""
Section 2: Returns
Just using today's rate minus today's rate for returns
Not using Log Returns - using Normal Model
"""

#Changed to difference
def returns_data(date, filename = "forward_sofr_swap_full.xlsx"):
    volatilites = pd.read_excel(filename, skiprows = 2).set_index("Ticker").sort_index().diff()
    mat_n_ten = maturity_tenor(filename).T
    d1 = pd.DataFrame(volatilites.loc[date])
    
    df = d1.join(mat_n_ten)
    df.columns = ["Values", "Tenor", "Maturity"]
    
    return df

def tabular_returns_form(date, filename = "forward_sofr_swap_full.xlsx"):
    df = returns_data(date, filename)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Values')
    return grid

"""
Section 3: Realized Volatility
Just using today's rate minus today's rate for returns
Not using Log Returns - using Normal Model
"""

#In the paper, they use nperiods = 20 and ann = 252
def realized_volatility_data(date, nperiods, ann, filename = "forward_sofr_swap_full.xlsx"):
    volatilites = pd.read_excel(filename, skiprows = 2).set_index("Ticker").sort_index().diff().rolling(nperiods).std()
    mat_n_ten = maturity_tenor(filename).T
    d1 = pd.DataFrame(volatilites.loc[date] * np.sqrt(ann))
    
    df = d1.join(mat_n_ten)
    df.columns = ["Values", "Tenor", "Maturity"]
    
    return df

def tabular_rvol_form(date, nperiods, ann, filename = "forward_sofr_swap_full.xlsx"):
    df = realized_volatility_data(date, nperiods, ann)
    grid = df.pivot(index='Tenor', columns="Maturity", values='Values')
    return grid

