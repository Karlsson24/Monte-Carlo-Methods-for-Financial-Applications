"""
This code uses Sobol sequences to perform Monte Carlo simulations for pricing Asian call options.
Sobol sequences provide a quasi-random low-discrepancy sequence that improves the convergence rate of the
Monte Carlo simulations. The code simulates the average price of the stock over time and uses this average
to price the Asian call options.

Steps:
1. Initialize the parameters for the simulation, including initial stock price, risk-free rate,
   time to maturity, volatility, number of samples, number of time steps, and strike prices.
2. Generate Sobol samples for the simulation and transform them to standard normal variables.
3. Simulate the stock price paths using Geometric Brownian Motion.
4. Calculate the average price for each simulation path.
5. Calculate the payoffs for Asian call options and discount them to present value.
6. Print the computed option prices.

Parameters:
- S0: Initial stock price.
- r: Risk-free interest rate.
- T: Time to maturity in years.
- n_time: Number of time steps.
- sigma: Volatility of the stock.
- n_samples: Number of samples for the simulation.
- K: List of strike prices.
"""

import numpy as np
from scipy.stats import norm, qmc

# Parameters
S0 = 100  # Initial stock price
r = 0.03  # Risk-free interest rate
T = 1  # Time to maturity in years
n_time = 52  # Number of time steps (weeks)
sigma = 0.2  # Volatility
n_samples = 20000  # Number of samples
K = [70, 100, 130]  # Strike prices

# Generate Sobol samples and transform to standard normal variables
sobol = qmc.Sobol(d=n_time, scramble=True)
sobol_samples = sobol.random(n_samples)
Z = norm.ppf(sobol_samples)

# Time step
dt = T / n_time

# Simulate stock price paths using Geometric Brownian Motion
S_paths = np.zeros((n_samples, n_time + 1))
S_paths[:, 0] = S0
for t in range(1, n_time + 1):
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Calculate the average price for each simulation path
S_mean = np.mean(S_paths[:, 1:], axis=1)  # Exclude the initial price S0

# Calculate and print prices for Asian call options
for strike in K:
    call_payoff = np.maximum(S_mean - strike, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    print(f"Asian Call option price at strike {strike} with {n_time} time steps is {call_price:.2f}")
