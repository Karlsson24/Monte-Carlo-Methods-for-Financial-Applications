"""
This code performs Monte Carlo simulations to estimate the prices of European call options using 
Geometric Brownian Motion. It calculates the option prices using both the arithmetic and geometric 
mean of simulated stock prices. Additionally, it applies a correction using the control variate technique 
to improve the accuracy of the option price estimate.

Steps:
1. Initialize the parameters for the simulation, including the number of paths, number of time steps, 
   risk-free rate, volatility, and initial stock price.
2. Generate random standard normal numbers for the simulation.
3. Simulate the stock price paths using Geometric Brownian Motion.
4. Calculate the average stock price for each path using both arithmetic and geometric means.
5. Estimate the call option prices for the arithmetic and geometric mean paths.
6. Apply the control variate technique to correct the option price estimate.
7. Print the corrected call option price estimate.

Parameters:
- r: Risk-free interest rate.
- K: Strike price.
- sigma: Volatility of the stock.
- S0: Initial stock price.
- m: Number of time steps (weeks).
- T: Time to maturity of the option in years.
- n_paths: Number of simulation paths.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import gmean  # geometric mean, used in assignment one

# Parameters
r = 0.03  # Risk-free interest rate
K = 130  # Strike price
sigma = 0.2  # Volatility
S0 = 100  # Initial stock price
m = 52  # Number of time steps (weeks)
T = 1  # Time to maturity (in years)
n_paths = 20000  # Number of simulation paths
n_month = 52  # Number of time steps (weeks)
dt = 1 / n_month  # Time step

# Initialize stock price matrix
S = np.zeros((n_paths, n_month + 1))
S[:, 0] = S0

# Generate random standard normal numbers for simulation
Z = np.random.standard_normal((n_paths, n_month + 1))

# Simulate stock price paths
for n in range(n_paths):
    for m in range(1, n_month + 1):
        S[n, m] = S[n, m - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[n, m])

# Calculate average stock price using arithmetic mean
avg_price_smc = np.mean(S[:, 1:], axis=1)
C_smc = np.exp(-r) * np.maximum(avg_price_smc - K, 0)
C_smc_mean = np.mean(C_smc)

# Calculate average stock price using geometric mean
avg_price_gmean = gmean(S[:, 1:], axis=1)
C_gmean = np.exp(-r) * np.maximum(avg_price_gmean - K, 0)
C_gmean_mean = np.mean(C_gmean)

# Calculate covariance and variance for control variate technique
covariance = np.cov(C_smc, C_gmean)[0, 1]
variance_gmean = np.var(C_gmean)

# Calculate control variate coefficient b
b = covariance / variance_gmean

# Apply control variate correction
Yi = C_smc_mean - b * (C_gmean_mean - 0.075829)

# Print the corrected call option price estimate
print(f"Call option CORRECTED with strike: {K} and price: {Yi:.5f}")
