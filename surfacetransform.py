import pandas as pd
import numpy as np

# Load the swaption implied volatility data
swaption_vol_path = "data\swaption_atm_vol_full.xlsx"
swaption_vol_df = pd.read_excel(swaption_vol_path)

# Display the structure of the data
print(swaption_vol_df.head())

# Extract necessary columns
maturities = ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]
tenors = ["1M", "3M", "6M", "1Y", "2Y", "5Y"]

# Create an empty dictionary to store surfaces
surfaces_dict = {}

for date in swaption_vol_df["Ticker"].unique():
    date_filter = swaption_vol_df[swaption_vol_df["Ticker"] == date]
    
    # Initialize an empty grid
    surface_matrix = np.zeros((len(maturities), len(tenors)))

    for i, mat in enumerate(maturities):
        for j, tenor in enumerate(tenors):
            column_name = f"USSNA{mat[0]} ICPL C"  # Adjust column naming if needed
            if column_name in date_filter.columns:
                surface_matrix[i, j] = date_filter[column_name].values[0]

    surfaces_dict[date] = surface_matrix

# Convert to a structured NumPy array
surfaces_transform = np.array([surfaces_dict[date] for date in surfaces_dict])

# np.save("\data\surfacestransform.npy", surfaces_transform)
