"""
This code uses Sobol sequences to perform Monte Carlo simulations for pricing European call and put options.
Sobol sequences provide a quasi-random low-discrepancy sequence that improves the convergence rate of the
Monte Carlo simulations.

Steps:
1. Initialize the parameters for the simulation, including initial stock price, risk-free rate,
   time to maturity, volatility, number of samples, and strike prices.
2. Generate Sobol samples for the simulation.
3. Transform Sobol samples to standard normal variables.
4. Simulate the stock prices at maturity using Geometric Brownian Motion.
5. Calculate the payoffs for call and put options.
6. Discount the payoffs to present value and compute the option prices.
7. Print the computed option prices.

Parameters:
- S0: Initial stock price.
- r: Risk-free interest rate.
- T: Time to maturity in years.
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
sigma = 0.2  # Volatility
n_samples = 20000  # Number of samples
K = [70, 100, 130]  # Strike prices

# Generate Sobol samples and transform to standard normal variables
sobol = qmc.Sobol(d=1, scramble=True)
sobol_samples = sobol.random_base2(m=int(np.log2(n_samples)))
Z = norm.ppf(sobol_samples[:n_samples])

# Simulate stock prices at maturity using Geometric Brownian Motion
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

# Calculate option prices
for strike in K:
    # Calculate payoffs for call and put options
    call_payoff = np.maximum(ST - strike, 0)
    put_payoff = np.maximum(strike - ST, 0)

    # Discount payoffs to present value
    discounted_call_payoff = np.exp(-r * T) * call_payoff
    discounted_put_payoff = np.exp(-r * T) * put_payoff

    # Calculate the mean of discounted payoffs to get option prices
    call_price = np.mean(discounted_call_payoff)
    put_price = np.mean(discounted_put_payoff)

    # Print the option prices
    print(f"Call option price at strike: {strike} is {call_price:.2f}")
    print(f"Put option price at strike: {strike} is {put_price:.2f}")
