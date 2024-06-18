"""
This script simulates the prices of two potentially correlated assets using the Geometric Brownian Motion model and estimates the value of American put options with different strike prices using the Least Squares Monte Carlo (LSMC) method. The script also calculates the premium of American options over their corresponding European options. The script performs the following steps:

1. Initialize parameters for the simulation and option pricing.
2. Generate random variables for the simulation.
3. Define the least_monte_carlo function to estimate the option value using the LSMC method.
4. Construct the correlation matrix and perform Cholesky decomposition.
5. Initialize and simulate the asset prices using the random variables.
6. Calculate and print the estimated option values and premiums for different strike prices.

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
from scipy.stats import norm  # gives us the normal pdf, cdf, and inverse cdf (ppf)
from scipy.optimize import bisect
import math

# Parameters
time = 6
S0 = 100
sigma = 0.3
T = 1
r = 0.03
n_paths = 40000
K = [80, 100, 120]
rho = 0

# Random variables
Z = np.random.normal(size=(2, n_paths, time + 1))

# Function to calculate least Monte Carlo
def least_monte_carlo(S, T, r, time, n_paths, K):
    dt = T / time
    S1, S2 = S
    V = np.maximum(K - 0.5 * (S1[:, -1] + S2[:, -1]), 0)
    for period in range(time - 1, 0, -1):
        A_value = np.column_stack([
            np.ones(n_paths),
            0.5 * (S1[:, period] + S2[:, period]),
            (0.5 * (S1[:, period] + S2[:, period])) ** 2,
            np.maximum(K - 0.5 * (S1[:, period] + S2[:, period]), 0)
        ])
        Y = np.exp(-r * dt) * V
        B = np.linalg.lstsq(A_value, Y, rcond=None)[0]
        C = A_value @ B
        V = np.maximum(A_value[:, -1], C)
    V = np.mean(np.exp(-r * dt) * V)
    return V

# Construct the correlation matrix
rho_matrix = np.full((2, 2), rho) + np.eye(2) * (1 - rho)
A = np.linalg.cholesky((sigma * np.eye(2)) @ rho_matrix @ (sigma * np.eye(2)))
dt = T / time
time_steps = np.cumsum(np.full((2, n_paths, time), dt), axis=2)
time_steps = np.insert(time_steps, 0, 0, axis=2)
S = np.zeros((2, n_paths, time + 1))
S[:, :, 0] = S0
for i in range(1, time + 1):
    X = A @ Z[:, :, i]
    S[:, :, i] = S[:, :, i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * X)

# Calculate and print results for each strike price
for k in K:
    V = least_monte_carlo(S, T, r, time, n_paths, k)
    SMC = np.mean(np.exp(-r * T) * np.maximum(k - 0.5 * (S[0, :, -1] + S[1, :, -1]), 0))
    Premium = V - SMC
    print(f"Premium of American call with strike {k} is: {Premium:.4f}")
