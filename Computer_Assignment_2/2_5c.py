"""
This script estimates the price of a European call option on a bond using the Vasicek interest rate model
through Monte Carlo simulation. The Vasicek model is used to simulate the interest rate paths, and the 
call option price is estimated based on these simulated paths.

Steps:
1. Initialize the parameters for the Vasicek model and Monte Carlo simulation settings.
2. Simulate the short rate paths using the Vasicek model.
3. Calculate the bond price at maturity T1 for each simulated path.
4. Compute the payoff of the call option and discount it back to present value.
5. Estimate the call option price as the average of the discounted payoffs.

Parameters:
- r0: Initial short rate.
- a: Speed of mean reversion.
- b: Long-term mean rate.
- sigma: Volatility of the short rate.
- K: Strike price of the call option.
- T1: Maturity of the bond.
- T2: Total simulation time.
- n_paths: Number of simulated paths.
- time: Number of time steps for the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Vasicek model
r0 = 0.0137  # Initial short rate
a = 0.2      # Speed of mean reversion
b = 0.02     # Long-term mean rate
sigma = 0.01 # Volatility of the short rate
K = 0.8      # Strike price of the call option
T1 = 5       # Maturity of the bond
T2 = 10      # Total simulation time

# Monte Carlo simulation settings
n_paths = 20000  # Number of simulated paths
time = 52        # Weekly time steps for a year
dt = T2 / time   # Time step size

# Initialize arrays to store the paths
r = np.zeros((n_paths, time + 1))
r[:, 0] = r0  # Initial condition for the short rate

# Generate random numbers for simulation
Z = np.random.standard_normal((n_paths, time))

# Simulate the short rate paths using the Vasicek model
for t in range(1, time + 1):
    dr = a * (b - r[:, t-1]) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
    r[:, t] = r[:, t-1] + dr

# Calculate bond price at maturity T1 for each simulated path
B_T1 = (1 - np.exp(-a * T1)) / a
A_T1 = ((b - (sigma**2) / (2 * a**2)) * (B_T1 - T1) - (sigma**2) * B_T1**2 / (4 * a))
P_T1 = np.exp(A_T1 - B_T1 * r[:, int(T1 / dt)])

# Compute the payoff of the call option and discount it back to present value
payoff = np.maximum(P_T1 - K, 0)
discount_factors = np.exp(-r[:, int(T1 / dt)] * T1)
option_price = np.mean(payoff * discount_factors)

# Print results
print(f"The estimated call option price is: {option_price:.4f}")
