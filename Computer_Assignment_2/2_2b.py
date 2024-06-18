"""
This code uses Monte Carlo simulations with importance sampling to price Asian call options.
Importance sampling is used to reduce the variance of the estimator and improve the accuracy of the simulation.

Steps:
1. Initialize the parameters for the simulation, including initial stock price, risk-free rate,
   time to maturity, volatility, number of samples, number of time steps, and strike price.
2. Generate standard normal random variables adjusted by the importance sampling mean shift.
3. Simulate the stock price paths using Geometric Brownian Motion.
4. Calculate the average price for each simulation path.
5. Calculate the adjustment factor (likelihood ratio) for importance sampling.
6. Calculate the payoffs for the Asian call options, adjusted by the likelihood ratio.
7. Discount the payoffs to present value and compute the option prices.
8. Print the computed option prices for different numbers of time steps.

Parameters:
- S0: Initial stock price.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- n_years: Time to maturity in years.
- n_paths: Number of samples for the simulation.
- K: Strike price.
- m: Number of time steps.
"""

import numpy as np
from scipy.stats import norm

def asian_call_importance_sampling(S0, r, sigma, n_years, n_paths, K, m):
    dt = n_years / m
    mu = 5 / m
    Z = np.random.standard_normal((n_paths, m)) + mu
    S = np.zeros((n_paths, m + 1))
    S[:, 0] = S0
    for t in range(1, m + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    average_prices = np.mean(S[:, 1:], axis=1)
    added_drift = np.exp(-mu * Z + 0.5 * mu ** 2)
    weights = np.prod(added_drift, axis=1)

    payoff = np.maximum(average_prices - K, 0)
    price_qmc = np.exp(-r * n_years) * np.mean(payoff * weights)

    return price_qmc

# Parameters
S0 = 100
n_paths = 20000
n_years = 1
r = 0.03
sigma = 0.2
K = 130

# Calculate prices for different values of m
for m in [12, 52]:
    option_price = asian_call_importance_sampling(S0, r, sigma, n_years, n_paths, K, m)
    print(f"Results for m = {m}:")
    print(f"  Strike Price: {K}, Asian Call Option Price with Importance Sampling: {option_price:.3f}")
