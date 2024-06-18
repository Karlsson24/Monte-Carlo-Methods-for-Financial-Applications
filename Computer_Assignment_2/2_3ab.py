"""
This script uses the Sobol sequence to perform Monte Carlo simulations for pricing European call and put options 
on a basket of two stocks. The Cholesky decomposition is used to model the correlation between the stocks.

Steps:
1. Initialize the parameters for the simulation.
2. Generate Sobol samples and transform them to standard normal variables.
3. Simulate the stock prices using Geometric Brownian Motion.
4. Calculate the option payoffs for a basket of two stocks.
5. Compute the discounted payoffs to obtain the option prices.
6. Print the computed option prices for different strike prices.

Parameters:
- S0: Initial stock price.
- r: Risk-free interest rate.
- stocks: Number of stocks in the basket.
- rå: Correlation between the stocks.
- sigma: Volatility of the stocks.
- K: List of strike prices.
- n_paths: Number of samples for the simulation.
- dt: Time step.
"""

import numpy as np
from scipy.stats import norm, qmc

# Parameters
S0 = 100
r = 0.03
stocks = 2
rå = 0.5
sigma = 0.2
K = [70, 100, 130]
n_paths = 20000
dt = 1

# Covariance matrix and Cholesky decomposition
cov_matrix = np.array([[sigma ** 2, rå * sigma * sigma], [rå * sigma * sigma, sigma ** 2]])
A = np.linalg.cholesky(cov_matrix)

# Generate Sobol samples and transform to standard normal variables
sampler = qmc.Sobol(d=2, scramble=True)
U = sampler.random_base2(m=15)
Z = norm.ppf(U[:n_paths])
X = A @ Z.T

# Simulate stock prices
S = np.zeros((2, n_paths))
S[0, :] = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * X[0, :])
S[1, :] = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * X[1, :])

# Calculate and print option prices
for strike in K:
    call_payoffs = np.exp(-r * dt) * np.maximum(0.5 * (S[0, :] + S[1, :]) - strike, 0)
    put_payoffs = np.exp(-r * dt) * np.maximum(strike - 0.5 * (S[0, :] + S[1, :]), 0)

    call_price = np.mean(call_payoffs)
    put_price = np.mean(put_payoffs)

    print(f"Call option price for strike {strike}: {call_price:.4f}")
    print(f"Put option price for strike {strike}: {put_price:.4f}")
