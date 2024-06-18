"""
This script simulates the prices of two correlated assets using the Geometric Brownian Motion model and
estimates the value of American put options with different strike prices using the Least Squares Monte Carlo (LSMC) method.
The script performs the following steps:

1. Initialize parameters for the simulation and option pricing.
2. Generate correlated random variables for the simulation.
3. Define the least_monte_carlo function to estimate the option value using the LSMC method.
4. Construct the correlation matrix and perform Cholesky decomposition.
5. Initialize and simulate the asset prices using the correlated random variables.
6. Calculate and print the estimated option values for different strike prices.

Parameters:
- time: Number of time steps in the simulation.
- S0: Initial stock price.
- sigma: Volatility of the stock.
- T: Time to maturity in years.
- r: Risk-free interest rate.
- n_paths: Number of simulated paths.
- K: List of strike prices for the options.
- rho: Correlation coefficient between the two assets.
"""

import numpy as np
from scipy.optimize import bisect

# Parameters
time = 6
S0 = 100
sigma = 0.3
T = 1
r = 0.03
n_paths = 40000
K = [80, 100, 120]
rho = 0.5

# Generate correlated random variables
Z = np.random.normal(size=(2, n_paths, time + 1))

# Define the least_monte_carlo function
def least_monte_carlo(S, T, r, time, n_paths, K):
    dt = T / time
    S1, S2 = S
    V = np.maximum(K - 0.5 * (S1[:, -1] + S2[:, -1]), 0)
    for period in range(time - 1, 0, -1):
        A_value = np.ones((n_paths, 4))
        S_avg = 0.5 * (S1[:, period] + S2[:, period])
        A_value[:, 1] = S_avg
        A_value[:, 2] = S_avg ** 2
        A_value[:, 3] = np.maximum(K - S_avg, 0)
        Y = np.exp(-r * dt) * V
        B = np.linalg.lstsq(A_value, Y, rcond=None)[0]
        C = A_value @ B
        V = np.maximum(A_value[:, 3], C)

    V = np.mean(np.exp(-r * dt) * V)
    return V

# Construct the correlation matrix
sigma_matrix = sigma * np.ones(2)
rho_matrix = np.array([[1, rho], [rho, 1]])
matrix = np.diag(sigma_matrix) @ rho_matrix @ np.diag(sigma_matrix)
A = np.linalg.cholesky(matrix)
dt = T / time

# Initialize asset prices
S = np.zeros((2, n_paths, time + 1))
S[:, :, 0] = S0

# Simulate asset prices
for i in range(1, time + 1):
    X = A @ Z[:, :, i]
    S[:, :, i] = S[:, :, i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * X)

# Calculate and print results for each strike price
for k in K:
    V_r = least_monte_carlo(S, T, r, time, n_paths, k)
    print(f"The estimated V for rho {rho:.2f} and K {k:.2f} is: {V_r:.4f}")
