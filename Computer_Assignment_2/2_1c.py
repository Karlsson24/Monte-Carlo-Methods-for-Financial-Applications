"""
This code performs Monte Carlo simulations to estimate the price of a European call option using
Standard Monte Carlo, Antithetic Variates, and Quasi-Monte Carlo methods. The results are compared
with the exact Black-Scholes price.

Steps:
1. Initialize the parameters for the simulation.
2. Define a function to calculate the Black-Scholes price.
3. Perform simulations using Standard Monte Carlo, Antithetic Variates, and Quasi-Monte Carlo methods.
4. Plot the estimated option prices against the number of simulations and compare with the exact Black-Scholes price.

Parameters:
- S0: Initial stock price.
- T: Time to maturity in years.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- K: Strike price.
- n_simulations: Array of different numbers of simulations.
"""

import numpy as np
from scipy.stats import norm, qmc
import math
import matplotlib.pyplot as plt

# Function to calculate the Black-Scholes price
def blackscholes(S0, T, r, sigma, K):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

# Parameters
n_simulations = np.arange(10000, 110000, 10000)
r = 0.03
sigma = 0.2
S0 = 100
T = 1
K = 130  # Strike price

# Calculate exact Black-Scholes call price
exact_call_price = blackscholes(S0, T, r, sigma, K)

C_means_SMC = []
C_means_Variance = []
C_means_Quasi = []

for n_paths in n_simulations:
    # Standard Monte Carlo
    Z = np.random.standard_normal(n_paths)
    S = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    C = np.exp(-r * T) * np.maximum(S - K, 0)
    C_means_SMC.append(np.mean(C))

    # Antithetic Variance Reduction
    S_antithetic = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * -Z)
    C_antithetic = np.exp(-r * T) * np.maximum(S_antithetic - K, 0)
    C_means_Variance.append(np.mean(0.5 * (C + C_antithetic)))

    # Quasi-Monte Carlo
    sobol = qmc.Sobol(d=1)
    sobol_samples = sobol.random_base2(m=int(math.log2(n_paths)))
    Z_quasi = norm.ppf(sobol_samples[:, 0])
    S_quasi = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_quasi)
    C_quasi = np.exp(-r * T) * np.maximum(S_quasi - K, 0)
    C_means_Quasi.append(np.mean(C_quasi))

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(n_simulations, C_means_SMC, label='Standard Monte Carlo Mean', marker='o')
plt.plot(n_simulations, C_means_Variance, label='Antithetic Variate Mean', marker='x')
plt.plot(n_simulations, C_means_Quasi, label='Quasi-Monte Carlo Mean', marker='s')
plt.axhline(y=exact_call_price, color='r', linestyle='-', label=f'Exact Black-Scholes Price: {exact_call_price:.2f}')
plt.xlabel('Number of Simulations')
plt.ylabel('Call Option Price')
plt.title(f'Call Option Simulation Results for K={K}')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
