"""
This code performs Monte Carlo simulations to calculate the prices of European call options
using Geometric Brownian Motion. The code uses both standard Monte Carlo and antithetic variates
to improve accuracy. The results are compared with the exact price calculated using the Black-Scholes formula.

Functions:
1. blackscholes(S0, T, r, sigma, K): Calculates the exact call option price using the Black-Scholes formula.
2. monte_carlo_simulation(S0, r, sigma, T, n_paths, n_years, Z): Performs the standard Monte Carlo simulation.
3. antithetic_variates(S0, r, sigma, T, n_paths_ant, Zi): Performs the antithetic variates simulation.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm  # Normal and lognormal pdf, cdf, and inverse cdf/pdf

# Parameters
K = 100  # Strike price
S0 = 100  # Initial stock price
r = 0.03  # Risk-free interest rate
sigma = 0.2  # Volatility
T = 1  # Time to maturity (years)
n_years = 1  # Number of years for simulation
dt = 1  # Time step
n_paths_list = range(10000, 100001, 10000)  # Number of simulation paths: 10k, 20k, ..., 100k

# Function to calculate the exact Black-Scholes price
def blackscholes(S0, T, r, sigma, K):
    """
    Calculate the Black-Scholes price for a European call option.

    Parameters:
    S0 (float): Initial stock price
    T (float): Time to maturity (years)
    r (float): Risk-free interest rate
    sigma (float): Volatility
    K (float): Strike price

    Returns:
    float: Call option price
    """
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

# Function to perform standard Monte Carlo simulation
def monte_carlo_simulation(S0, r, sigma, T, n_paths, Z):
    """
    Perform Monte Carlo simulation for call option pricing.

    Parameters:
    S0 (float): Initial stock price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    T (float): Time to maturity (years)
    n_paths (int): Number of simulation paths
    Z (np.ndarray): Random variables for simulation

    Returns:
    float: Monte Carlo estimated call option price
    """
    S = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * Z[:n_paths])
    C = np.exp(-r * T) * np.maximum(S - K, 0)
    return np.mean(C)

# Function to perform antithetic variates simulation
def antithetic_variates(S0, r, sigma, T, n_paths_ant, Zi):
    """
    Perform antithetic variates simulation for call option pricing.

    Parameters:
    S0 (float): Initial stock price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    T (float): Time to maturity (years)
    n_paths_ant (int): Number of antithetic variate simulation paths
    Zi (np.ndarray): Random variables for antithetic simulation

    Returns:
    float: Antithetic variates estimated call option price
    """
    y1_call = np.exp(-r) * np.maximum(S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * Zi[:n_paths_ant]) - K, 0)
    y2_call = np.exp(-r) * np.maximum(S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * (-Zi[:n_paths_ant])) - K, 0)
    return np.mean(0.5 * (y1_call + y2_call))

# Calculate the exact Black-Scholes price
price_call_bs = blackscholes(S0, T, r, sigma, K)

# Arrays to store the prices
prices_mc = []
prices_ant = []
prices_bs = [price_call_bs] * len(n_paths_list)  # Black-Scholes price remains constant

# Simulate for different numbers of paths
for n_paths in n_paths_list:
    Z = np.random.standard_normal(n_paths)
    Zi = np.random.standard_normal(n_paths // 2)

    price_mc = monte_carlo_simulation(S0, r, sigma, T, n_paths, Z)
    prices_mc.append(price_mc)

    price_ant = antithetic_variates(S0, r, sigma, T, n_paths // 2, Zi)
    prices_ant.append(price_ant)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(n_paths_list, prices_bs, label='Black-Scholes', color='blue', linestyle='--')
plt.plot(n_paths_list, prices_mc, label='Monte Carlo', color='green', marker='o')
plt.plot(n_paths_list, prices_ant, label='Antithetic Variates', color='red', marker='x')
plt.title('Comparison of Call Option Pricing Methods')
plt.xlabel('Number of Simulation Paths')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.show()
