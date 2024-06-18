"""
This code performs Monte Carlo simulations to estimate the prices of European call options using 
Geometric Brownian Motion. It calculates the option prices using both standard Monte Carlo and 
geometric mean methods. Additionally, it applies a correction using the control variate technique 
to improve the accuracy of the option price estimate. The results are compared with the exact price 
calculated using the Black-Scholes formula.

Steps:
1. Initialize the parameters for the simulation, including the number of paths, number of time steps, 
   risk-free rate, volatility, and initial stock price.
2. Calculate relevant times and volatilities for the Black-Scholes formula.
3. Loop over different numbers of simulation paths to perform the following:
   - Generate random standard normal numbers for the simulation.
   - Simulate stock price paths using Geometric Brownian Motion.
   - Calculate the exact call option price using the Black-Scholes formula.
   - Calculate the average stock price for each path using both arithmetic and geometric means.
   - Estimate the call option prices using standard Monte Carlo and geometric mean methods.
   - Apply the control variate technique to correct the option price estimate.
   - Store the results for plotting.
4. Plot the estimated prices from different methods against the number of simulation paths.

Parameters:
- r: Risk-free interest rate.
- K: Strike price.
- sigma: Volatility of the stock.
- S0: Initial stock price.
- m: Number of time steps (weeks).
- T: Time to maturity of the option in years.
- n_month: Number of time steps (weeks).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gmean  # geometric mean

# Parameters
r = 0.03  # Risk-free interest rate
K = 100  # Strike price
sigma = 0.2  # Volatility
S0 = 100  # Initial stock price
m = 52  # Number of time steps (weeks)
T = 1  # Time to maturity (in years)
n_month = 52  # Number of time steps (weeks)

# Calculate relevant times and volatilities
dt = 1 / m
times = np.arange(1, m + 1) * dt
T_tak = dt * sum(times)
tau = times
sigma_tak_squared = sigma ** 2 / (m ** 2 * T_tak) * sum((2 * i - 1) * tau[m - i] for i in range(1, m + 1))
delta = 0.5 * (sigma ** 2) - 0.5 * sigma_tak_squared

# Prepare lists to store results
n_paths_list = []
exact_prices = []
mc_prices = []
gmean_prices = []
yi_prices = []

# Calculate Black-Scholes price (outside the loop for efficiency)
d1 = (np.log(S0 / K) + ((r - delta + 0.5 * sigma_tak_squared) * T_tak)) / (np.sqrt(sigma_tak_squared * T_tak))
d2 = d1 - np.sqrt(sigma_tak_squared * T_tak)
BS_formula = np.exp(-delta * T_tak) * S0 * norm.cdf(d1) - np.exp(-r * T_tak) * K * norm.cdf(d2)
correction_factor = np.exp(-r * (T - T_tak))
exact_price = correction_factor * BS_formula

# Simulations and data collection
for n_paths in range(10000, 110000, 10000):
    Z = np.random.standard_normal((n_paths, n_month))
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))

    # Calculate Monte Carlo prices using standard mean
    avg_price_smc = np.mean(S, axis=1)
    C_smc = np.exp(-r * T) * np.maximum(avg_price_smc - K, 0)
    C_smc_mean = np.mean(C_smc)

    # Calculate Monte Carlo prices using geometric mean
    avg_price_gmean = gmean(S, axis=1)
    C_gmean = np.exp(-r * T) * np.maximum(avg_price_gmean - K, 0)
    C_gmean_mean = np.mean(C_gmean)

    # Correction and covariance analysis
    covariance = np.cov(C_smc, C_gmean)[0, 1]
    variance_gmean = np.var(C_gmean)
    b = covariance / variance_gmean

    Yi = C_smc_mean - b * (C_gmean_mean - exact_price)

    # Store the results
    n_paths_list.append(n_paths)
    exact_prices.append(exact_price)
    mc_prices.append(C_smc_mean)
    gmean_prices.append(C_gmean_mean)
    yi_prices.append(Yi)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(n_paths_list, exact_prices, label='Exact price', marker='o')
plt.plot(n_paths_list, mc_prices, label='Standard Monte Carlo', marker='o')
plt.plot(n_paths_list, gmean_prices, label='Geometric mean', marker='o')
plt.plot(n_paths_list, yi_prices, label='Corrected Yi', marker='o')
plt.xlabel('Number of simulations')
plt.ylabel('Price')
plt.title('Price estimation using different methods over simulations')
plt.legend()
plt.grid(True)
plt.show()
