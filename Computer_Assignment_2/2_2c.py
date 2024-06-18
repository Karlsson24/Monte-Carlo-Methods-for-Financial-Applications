"""
This script compares different Monte Carlo simulation methods for pricing Asian call options.
The methods include Standard Monte Carlo, Importance Sampling, Quasi Monte Carlo, and Control Variate.

Steps:
1. Initialize the parameters for the simulation.
2. Define functions for each pricing method.
3. Perform simulations using each method.
4. Calculate the option prices.
5. Plot the results to compare the pricing methods.

Parameters:
- S0: Initial stock price.
- k: Strike price.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- T: Time to maturity in years.
- m: Number of time steps.
- exact_price: Given exact price for comparison.
- n_paths: Number of samples for the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import qmc
from scipy.stats import gmean

# Parameters
S0 = 100
k = 130
r = 0.03
sigma = 0.2
T = 1
n_years = 1
m = 52
exact_price = 1.3582800011480647
n_paths = 20000

def monte_carlo_price(n_paths):
    dt = T / m
    Z = np.random.standard_normal((n_paths, m))
    S = np.zeros((n_paths, m + 1))
    S[:, 0] = S0
    for t in range(1, m + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * Z[:, t - 1])

    average_prices_mean = np.mean(S[:, 1:], axis=1)
    average_prices_gmean = gmean(S[:, 1:], axis=1)
    payoffs_monte = np.maximum(average_prices_mean - k, 0)
    payoff_mc = np.exp(-r * T) * np.mean(payoffs_monte)

    return payoff_mc, average_prices_mean, payoffs_monte

def asian_call_importance_sampling(S0, r, sigma, n_years, n_paths, K, m):
    dt = n_years / m
    mu = 5 / m
    Z = np.random.standard_normal((n_paths, m)) + mu
    S = np.zeros((n_paths, m + 1))
    S[:, 0] = S0
    for t in range(1, m + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    average_prices = np.mean(S[:, 1:], axis=1)
    added_drift = np.exp(-mu * (Z) + 0.5 * mu ** 2)
    weights = np.prod(added_drift, axis=1)

    payoff = np.maximum(average_prices - K, 0)
    price_qmc = np.exp(-r * n_years) * np.mean(payoff * weights)

    return price_qmc

def asian_call_price(S0, r, sigma, n_years, n_paths, k, m):
    sampler = qmc.Sobol(d=m, scramble=True)
    sobol_samples = sampler.random(n_paths)
    Z = norm.ppf(sobol_samples)

    dt = n_years / m
    S = np.zeros((n_paths, m + 1))
    S[:, 0] = S0

    for t in range(1, m + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    average_prices = np.mean(S[:, 1:], axis=1)
    payoffs = np.maximum(average_prices - k, 0)
    C = np.exp(-r * n_years) * np.mean(payoffs)

    return C

monte_carlo_results = []
importance_sampling_results = []
quasi_monte_carlo_results = []

path_sizes = range(10000, 100000, 10000)
for n in path_sizes:
    mc_price, avg_prices, payoffs = monte_carlo_price(n)
    is_price = asian_call_importance_sampling(S0, r, sigma, T, n, k, m)
    qmc_price = asian_call_price(S0, r, sigma, T, n, k, m)

    monte_carlo_results.append(mc_price)
    importance_sampling_results.append(is_price)
    quasi_monte_carlo_results.append(qmc_price)

S = np.zeros((n_paths, m + 1))
dt = T / m
times = np.arange(1, m + 1) * dt
S[:, 0] = S0

t_tak = 1 / m * sum(times)
sum_T = sum((2 * i - 1) * times[m - i] for i in range(1, m + 1))
sigma_tak_kvadrat = sigma**2 / (m**2 * t_tak) * sum_T
delta = (0.5 * sigma**2) - (0.5 * sigma_tak_kvadrat)
d_star = (np.log(S0 / k) +
(r - delta + 0.5 * sigma_tak_kvadrat) * t_tak) / (np.sqrt(sigma_tak_kvadrat * t_tak))
d2_star = d_star - np.sqrt(sigma_tak_kvadrat) * np.sqrt(t_tak)
price_gbm = np.exp(-r * (T - t_tak)) * (np.exp(-delta * t_tak) * 100 * norm.cdf(d_star) - np.exp(-r * t_tak) * k * norm.cdf(d2_star))

Z = np.random.standard_normal((n_paths, m))
for n in range(n_paths):
    for t in range(1, m + 1):
        S[n, t] = S[n, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * sigma * Z[n, t - 1])

average_prices = np.mean(S[:, 1:], axis=1)
payoffs_monte = np.maximum(average_prices - k, 0)
C_mc = np.exp(-r * n_years) * np.mean(payoffs_monte)

average_prices_gmean = gmean(S[:, 1:], axis=1)
payoffs_gmean = np.maximum(average_prices_gmean - k, 0)
C_geo = np.exp(-r * n_years) * np.mean(payoffs_gmean)

cov = np.cov(payoffs_monte, payoffs_gmean)[0, 1]
var_geo = np.var(payoffs_gmean)
b_hat_n = cov / var_geo

adjusted_payoffs = payoffs_monte - b_hat_n * (payoffs_gmean - price_gbm)
C_cv = np.exp(-r * n_years) * np.mean(adjusted_payoffs)

price_data_cv = []

for size in path_sizes:
    payoffs_monte_sample = payoffs_monte[:size]
    payoffs_gmean_sample = payoffs_gmean[:size]
    adjusted_payoffs_sample = payoffs_monte_sample - b_hat_n * (payoffs_gmean_sample - price_gbm)

    C_cv_sample = np.exp(-r * n_years) * np.mean(adjusted_payoffs_sample)
    price_data_cv.append(C_cv_sample)


plt.figure(figsize=(10, 6))
plt.plot(path_sizes, monte_carlo_results, label="Monte Carlo")
plt.plot(path_sizes, importance_sampling_results, label="Importance Sampling")
plt.plot(path_sizes, quasi_monte_carlo_results, label="Quasi Monte Carlo")
plt.plot(path_sizes, price_data_cv, marker='s', linestyle='--', label='Control Variate')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Asian Option Pricing with Various Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()