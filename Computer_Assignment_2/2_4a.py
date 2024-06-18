"""
This script simulates the price of a European barrier option using the Heston model.
The Heston model is a stochastic volatility model where the volatility follows a mean-reverting square-root process.
The script calculates the option price by simulating multiple paths and averaging the discounted payoffs.

Steps:
1. Initialize the global parameters for the simulation.
2. Generate random variables for each time step and each path.
3. Simulate the stock price and volatility paths using the Heston model.
4. Check if the barrier condition is met and calculate the payoff accordingly.
5. Compute the discounted average payoff to obtain the option price.

Parameters:
- S0: Initial stock price.
- r: Risk-free interest rate.
- sigma: Initial volatility.
- n_paths: Number of paths to simulate.
- T: Time to maturity in years.
- B: Barrier level.
- m: Number of time steps.
- K: Strike price.
- a: Speed of mean reversion.
- b: Long-term variance.
- sigma_v: Volatility of volatility.
- rho: Correlation between the stock price and its volatility.
"""

import numpy as np

# Global parameters
S0 = 100
r = 0.03
sigma = 0.4
n_paths = 20000
T = 1
B = 80
m = 12
K = 100
a = 0.1
b = sigma ** 2
sigma_v = 0.1
rho = -0.7
dt = T / m

# Initialize matrices for stock prices and volatilities
S = np.zeros((n_paths, m + 1))
v = np.zeros((n_paths, m + 1))
S[:, 0] = S0
v[:, 0] = sigma ** 2
payoffs = np.zeros(n_paths)

# Simulation
for path in range(n_paths):
    barrier = False
    for t in range(1, m + 1):
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        X1 = Z1
        X2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        # Update volatility
        v[path, t] = v[path, t - 1] + a * (b - v[path, t - 1]) * dt + sigma_v * np.sqrt(v[path, t - 1] * dt) * X2
        v[path, t] = np.maximum(v[path, t], 0)  # Ensure volatility remains non-negative

        # Update stock price
        S[path, t] = S[path, t - 1] + r * S[path, t - 1] * dt + np.sqrt(v[path, t - 1] * dt) * S[path, t - 1] * X1

        # Check if barrier condition is met
        if S[path, t] <= B:
            barrier = True

    # Calculate payoff
    if barrier:
        payoff = max(S[path, -1] - K, 0)
    else:
        payoff = 0
    payoffs[path] = payoff

# Calculate option price
option_price = np.exp(-r * T) * np.mean(payoffs)
print(f"Option price: {option_price:.4f}")
