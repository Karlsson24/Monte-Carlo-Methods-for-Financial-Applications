"""
This code calculates the price of a down-and-in barrier option using the continuous-time formula.
It accounts for the continuity correction factor to adjust the barrier level.

Steps:
1. Initialize the parameters for the option, including time to maturity, initial stock price,
   risk-free rate, volatility, strike price, and barrier level.
2. Calculate the adjustment factor for the barrier level.
3. Adjust the barrier level using the continuity correction factor.
4. Calculate the option price using the down-and-in barrier option formula.

Parameters:
- T: Time to maturity of the option in years.
- S0: Initial stock price.
- q: Dividend yield.
- n_month: Number of time steps (months).
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- K: Strike price.
- b: Barrier level.
"""

import numpy as np
from scipy.stats import norm

# Parameters
T = 1  # Time to maturity (in years)
S0 = 100  # Initial stock price
q = 0  # Dividend yield
n_month = 52  # Number of time steps (months)
r = 0.03  # Risk-free interest rate
sigma = 0.4  # Volatility
K = 100  # Strike price
b = 80  # Barrier level

# Time step
dt = T / n_month

# Adjustment factor for continuity correction
adjustment_factor = np.exp(-0.5826 * sigma * np.sqrt(T / n_month))

# Adjusted barrier level
H = b * adjustment_factor

# Calculate lambda and y for the down-and-in barrier option formula
lam = (r - q + sigma**2 * 0.5) / sigma**2
y = np.log(H**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

# Calculate the down-and-in barrier option price
C_di = (S0 * np.exp(-q * T) * (H / S0)**(2 * lam) * norm.cdf(y) -
        K * np.exp(-r * T) * (H / S0)**(2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))

print(C_di)
