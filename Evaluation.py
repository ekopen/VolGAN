# Evaluation.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def plot_gradient_norm_and_arb_penalties(iterations, grad_norms, arb_penalties):
    """
    Plots the gradient norms and arbitrage penalty values over iterations.
    
    Parameters:
        iterations (array-like): Iteration numbers.
        grad_norms (array-like): Recorded L2 norms (gradient norms) over iterations.
        arb_penalties (array-like): Recorded arbitrage penalty values over iterations.
    """
    # Plot Gradient Norms
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, grad_norms, marker='o', label='Gradient Norm')
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm")
    plt.title("Gradient Norm Over Training Iterations")
    plt.legend()
    plt.show()

    # Plot Arbitrage Penalties
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, arb_penalties, marker='s', color='r', label='Arbitrage Penalty')
    plt.xlabel("Iteration")
    plt.ylabel("Arbitrage Penalty")
    plt.title("Arbitrage Penalty Over Training Iterations")
    plt.legend()
    plt.show()


def plot_forecast_vs_actual_returns(gen, condition_test, true_test, noise_dim, device, sample_idx=0):
    """
    Plots forecasted vs actual returns for one test sample.
    
    Assumes that:
      - The generator's output vector's column 0 contains the annualized return.
      - condition_test and true_test are PyTorch tensors of shape 
        (num_samples, num_assets, feature_dim).
    
    Parameters:
        gen (nn.Module): The trained generator model.
        condition_test (torch.Tensor): Test condition tensor.
        true_test (torch.Tensor): Test true label tensor.
        noise_dim (int): The noise dimension used by the generator.
        device (str): Device identifier (e.g., 'cpu' or 'cuda').
        sample_idx (int): Index of the sample to plot.
    """
    # Select a single sample from test set
    condition_sample = condition_test[sample_idx:sample_idx+1]  # shape: (1, num_assets, cond_dim)
    noise_sample = torch.randn((1, condition_sample.shape[1], noise_dim), device=device, dtype=torch.float)
    
    # Get generator forecast
    forecast = gen(noise_sample, condition_sample)  # shape: (1, num_assets, output_dim)
    forecast = forecast.squeeze(0).detach().cpu().numpy()  # shape: (num_assets, output_dim)
    true_sample = true_test[sample_idx].detach().cpu().numpy()  # shape: (num_assets, output_dim)

    # Plot returns (assume returns are in column 0)
    plt.figure(figsize=(8, 4))
    plt.plot(forecast[:, 0], marker='o', label='Forecast Returns')
    plt.plot(true_sample[:, 0], marker='x', linestyle='--', label='Actual Returns')
    plt.xlabel("Asset Index")
    plt.ylabel("Annualized Return")
    plt.title("Forecast vs Actual Returns")
    plt.legend()
    plt.show()


def plot_forecast_vs_actual_vol_surface(gen, condition_test, true_test, noise_dim, device,
                                        grid_rows=15, grid_cols=9, vol_col=3,
                                        forecast_vol_idx=1, true_vol_idx=1, sample_idx=0):
    """
    Plots the forecasted and actual volatility surfaces.
    
    Assumes that:
      - Each asset corresponds to a point on the vol surface grid.
      - The past vol (to be added to the forecast vol increment) is stored in column `vol_col`
        of condition_test.
      - The generator outputs a vol increment at column `forecast_vol_idx` and the true
        vol increment is in true_test at column `true_vol_idx`.
    
    Parameters:
        gen (nn.Module): The trained generator model.
        condition_test (torch.Tensor): Test condition tensor.
        true_test (torch.Tensor): Test true tensor.
        noise_dim (int): The noise dimension used by the generator.
        device (str): Device identifier.
        grid_rows (int): Number of rows in the vol surface grid.
        grid_cols (int): Number of columns in the vol surface grid.
        vol_col (int): Column index in condition_test containing past vol.
        forecast_vol_idx (int): Column index in forecast corresponding to vol increment.
        true_vol_idx (int): Column index in true_test corresponding to vol increment.
        sample_idx (int): Index of the sample to plot.
    """
    condition_sample = condition_test[sample_idx:sample_idx+1]  # shape: (1, num_assets, cond_dim)
    noise_sample = torch.randn((1, condition_sample.shape[1], noise_dim), device=device, dtype=torch.float)
    
    # Get generator forecast and true values
    forecast = gen(noise_sample, condition_sample)  # shape: (1, num_assets, output_dim)
    forecast = forecast.squeeze(0).detach().cpu().numpy()  # shape: (num_assets, output_dim)
    true_sample = true_test[sample_idx].detach().cpu().numpy()  # shape: (num_assets, output_dim)
    
    num_assets = condition_sample.shape[1]
    required_assets = grid_rows * grid_cols
    if num_assets < required_assets:
        print("Not enough assets to reshape into a vol surface grid.")
        return

    # Extract past vol from condition_sample (assumes it is stored in column vol_col)
    past_vol = condition_sample[0, :required_assets, vol_col].detach().cpu().numpy()

    # Reconstruct forecast and actual vol surfaces by adding past vol to vol increments
    forecast_vol = forecast[:required_assets, forecast_vol_idx] + past_vol
    actual_vol = true_sample[:required_assets, true_vol_idx] + past_vol

    # Reshape into grid
    forecast_vol_grid = forecast_vol.reshape(grid_rows, grid_cols)
    actual_vol_grid = actual_vol.reshape(grid_rows, grid_cols)

    # Plot the forecast and actual vol surfaces side-by-side
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(forecast_vol_grid, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Forecast Vol Surface")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(actual_vol_grid, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Actual Vol Surface")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()



def evaluate_prediction_accuracy(forecast, true_sample):
    """
    Evaluates the accuracy of the generator's swaption return forecast.

    Parameters:
        forecast (np.array): Predicted returns, shape (num_assets, output_dim)
        true_sample (np.array): True returns, shape (num_assets, output_dim)

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Extract the annualized return column (column 0)
    forecast_returns = forecast[:, 0]
    true_returns = true_sample[:, 0]

    # Compute metrics
    mae = mean_absolute_error(true_returns, forecast_returns)
    mse = mean_squared_error(true_returns, forecast_returns)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_returns, forecast_returns)
    mape = np.mean(np.abs((true_returns - forecast_returns) / true_returns)) * 100  # Convert to %

    # Return results
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r2,
        "MAPE (%)": mape
    }


def evaluate_volatility_prediction_accuracy(forecast_vol, actual_vol):
    """
    Evaluates the accuracy of the generator's volatility surface forecast.

    Parameters:
        forecast_vol (np.array): Predicted volatilities, shape (grid_rows, grid_cols)
        actual_vol (np.array): True volatilities, shape (grid_rows, grid_cols)

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Flatten volatility surfaces for metric calculations
    forecast_vol = forecast_vol.flatten()
    actual_vol = actual_vol.flatten()

    # Compute metrics
    mae = mean_absolute_error(actual_vol, forecast_vol)
    mse = mean_squared_error(actual_vol, forecast_vol)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_vol, forecast_vol)
    mape = np.mean(np.abs((actual_vol - forecast_vol) / actual_vol)) * 100  # Convert to %

    # Forecast bias (mean signed error)
    bias = np.mean(forecast_vol - actual_vol)

    # RMSE ratio: compare RMSE to a simple historical volatility benchmark (naïve)
    naive_rmse = np.sqrt(np.mean(np.square(actual_vol - np.mean(actual_vol))))
    rmse_ratio = rmse / naive_rmse if naive_rmse != 0 else np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r2,
        "MAPE (%)": mape,
        "Forecast Bias": bias,
        "RMSE Ratio": rmse_ratio
    }



# Uncomment the lines below to perform a simple test if running this file directly.
# if __name__ == "__main__":
#     # Example dummy data (for testing purposes only)
#     iterations = np.arange(50)
#     grad_norms = np.random.uniform(0.4, 0.6, size=50)
#     arb_penalties = np.random.uniform(8.0, 12.0, size=50)
#     plot_gradient_norm_and_arb_penalties(iterations, grad_norms, arb_penalties)
