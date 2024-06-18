"""
This code performs Monte Carlo simulations to estimate the prices of European call options using
Geometric Brownian Motion. It calculates the option prices for a barrier option where the option
payout depends on whether the stock price hits a certain barrier level during the option's lifetime.

Steps:
1. Initialize the parameters for the simulation, including the number of paths, number of time steps,
   risk-free rate, volatility, and initial stock price.
2. Generate random standard normal numbers for the simulation.
3. Simulate the stock price paths using Geometric Brownian Motion.
4. Check if the stock price hits the barrier level at any time step.
5. Calculate the payoff based on whether the barrier condition is met.
6. Calculate the average option price from the simulated paths.

Parameters:
- T: Time to maturity of the option in years.
- n_paths: Number of simulation paths.
- n_month: Number of time steps (months).
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- S0: Initial stock price.
- K: Strike price.
- b: Barrier level.
"""

import numpy as np
from scipy.stats import norm
from scipy.stats.mstats import gmean  # geometric mean

# Parameters
T = 1  # Time to maturity (in years)
n_paths = 20000  # Number of simulation paths
n_month = 12  # Number of time steps (months)
r = 0.03  # Risk-free interest rate
sigma = 0.4  # Volatility
S0 = 100  # Initial stock price
K = 100  # Strike price
b = 80  # Barrier level
dt = T / n_month  # Time step

# Simulate stock price paths
Z = np.random.standard_normal((n_paths, n_month + 1))
S = np.ones((n_paths, n_month + 1)) * S0
S[:, 1:] = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, 1:], axis=1))

# Check if the stock price hits the barrier level at any time step
hits_barrier = np.any(S[:, 1:] <= b, axis=1)
stock_value_end = np.where(hits_barrier, S[:, -1], 0)

# Calculate the payoff
payment = np.exp(-r * T) * np.maximum(stock_value_end - K, 0)

# Calculate the average option price
avg_price_option = np.mean(payment)
print(avg_price_option)

# Repeat the simulation
Z = np.random.standard_normal((n_paths, n_month + 1))
S = np.ones((n_paths, n_month + 1)) * S0
S[:, 1:] = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, 1:], axis=1))

hits_barrier = np.any(S[:, 1:] <= b, axis=1)
stock_value_end = np.where(hits_barrier, S[:, -1], 0)

payment = np.exp(-r * T) * np.maximum(stock_value_end - K, 0)

avg_price_option = np.mean(payment)
print(avg_price_option)
