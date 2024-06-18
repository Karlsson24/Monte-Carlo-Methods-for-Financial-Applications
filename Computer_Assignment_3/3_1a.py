"""
This script performs a Monte Carlo simulation to price European call options and to evaluate the effectiveness
of delta hedging using different rebalancing frequencies. The script calculates the exact option price using
the Black-Scholes formula, simulates stock price paths, calculates the option value and delta at each step,
and performs delta hedging. The results include the mean and standard deviation of the hedging error (delta).

Steps:
1. Initialize the parameters for the simulation and Black-Scholes model.
2. Define functions for simulating stock paths, calculating exact option prices, calculating option values,
   and performing delta hedging.
3. Loop over different rebalancing frequencies and perform the simulation and hedging.
4. Print the mean and standard deviation of the hedging error for each rebalancing frequency.

Parameters:
- S0: Initial stock price.
- K: Strike price of the option.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- T: Time to maturity in years.
- n_periods: List of different rebalancing frequencies (number of periods in a year).
- n_paths: Number of simulated paths.
"""

import numpy as np
from scipy.stats import norm  # Provides the normal PDF, CDF, and inverse CDF (PPF)

# Parameters for the simulation and Black-Scholes model
S0 = 100
K = 90
r = 0.03
sigma = 0.3
T = 1
n_periods = [12, 52, 252]
n_paths = 10000

def SMC(T, n_paths, n_periods, Z, S0, r, sigma):
    dt = T / n_periods
    Z = np.cumsum(np.sqrt(dt) * Z[:, 1:], axis=1)
    Z = np.hstack((np.zeros((n_paths, 1)), Z))  # Add initial zero column for W
    t = np.linspace(0, T, n_periods + 1)
    S = S0 * np.exp((r - 0.5 * sigma ** 2) * t + sigma * Z)
    return S

def delta(S0, S, T, K, r):
    delta_v = np.exp(-r * T) * S[:, -1] / S0 * (S[:, -1] > K)
    delta = np.mean(delta_v)
    return delta

def exact_price(S0, sigma, T, r, K):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    return C

def calc_value(T, n_periods, S, r, sigma, K):
    dt = T / n_periods
    T_i = T - np.arange(n_periods) * dt
    d1 = (np.log(S[:, :n_periods] / K) + (r + 0.5 * sigma ** 2) * T_i) / (sigma * np.sqrt(T_i))
    d2 = d1 - sigma * np.sqrt(T_i)
    C = S[:, :n_periods] * norm.cdf(d1) - K * np.exp(-r * T_i) * norm.cdf(d2)
    V = np.hstack((C, np.maximum(S[:, -1:] - K, 0)))
    return V

def hedging(V, S, S0, T, n_paths, n_periods, r, dt, sigma, K):
    P = np.zeros((n_paths, n_periods + 1))
    delta_start = np.zeros((n_paths, n_periods + 1))
    delta = np.zeros((n_paths, n_periods + 1))
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta_start[:, 0] = norm.cdf(d1)
    y = V[:, 0] - delta_start[:, 0] * S0
    P[:, 0] = delta_start[:, 0] * S0 + y
    for i in range(1, n_periods):
        d1 = (np.log(S[:, i] / K) + (r + 0.5 * sigma ** 2) * (T - i * dt)) / (sigma * np.sqrt(T - i * dt))
        delta_start[:, i] = norm.cdf(d1)
        P[:, i] = delta_start[:, i-1] * S[:, i] + np.exp(r * dt) * y
        delta[:, i] = P[:, i] - V[:, i]
        y = V[:, i] - delta_start[:, i] * S[:, i]
    P[:, -1] = delta_start[:, -2] * S[:, -1] + np.exp(r * dt) * y
    delta[:, -1] = np.exp(-r * dt) * P[:, -1] - V[:, -1]
    delta_sum = np.sum(delta, axis=1)
    return P, delta_sum

# Loop over different rebalancing frequencies and perform the simulation and hedging
for n_period in n_periods:
    dt = T / n_period
    Z = np.random.normal(size=(n_paths, n_period + 1))
    S = SMC(T, n_paths, n_period, Z, S0, r, sigma)
    delta_value = delta(S0, S, T, K, r)
    V = calc_value(T, n_period, S, r, sigma, K)
    P_hedge, delta_sum = hedging(V, S, S0, T, n_paths, n_period, r, dt, sigma, K)
    Delta_mean = np.mean(delta_sum)
    Delta_sd = np.std(delta_sum)
    print(f"Number of periods: {n_period}")
    print(f"Mean of Delta: {Delta_mean}")
    print(f"Standard deviation of Delta: {Delta_sd}")
