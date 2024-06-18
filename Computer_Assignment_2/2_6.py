"""
This script simulates the cash flows and calculates the spread for a bank and a customer using the Vasicek interest rate model.
The model simulates the short rate paths and uses these to compute the cash flows for both the bank and the customer.
The spread is then calculated as the ratio of the bank's mean cash flow to the customer's mean cash flow.

Steps:
1. Initialize the parameters for the Vasicek model and Monte Carlo simulation settings.
2. Simulate the short rate paths using the Vasicek model.
3. Calculate the discount factors and cash flows for the bank and the customer.
4. Compute the mean cash flows and the spread.
5. Print the calculated spread.

Parameters:
- r0: Initial short rate.
- a: Speed of mean reversion.
- b: Long-term mean rate.
- sigma: Volatility of the short rate.
- T: Total simulation time in years.
- amount: Total amount of the cash flow.
- ceil: Ceiling rate for the interest rate.
- n_paths: Number of simulated paths.
- time: Number of time steps for the simulation.
- dt: Time step size.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Vasicek model
r0 = 0.0137  # Initial short rate
a = 0.2      # Speed of mean reversion
b = 0.02     # Long-term mean rate
sigma = 0.01 # Volatility of the short rate
T = 5        # Total simulation time in years
amount = 100000000  # Total amount of the cash flow
ceil = 0.023        # Ceiling rate for the interest rate
n_paths = 100000    # Number of simulated paths
time = 12 * 5       # Number of time steps for the simulation (monthly over 5 years)
dt = 1 / 12         # Time step size (monthly)

# Initialize arrays to store the paths
r = np.zeros((n_paths, time + 1))
r[:, 0] = r0

# Generate random numbers for simulation
Z = np.random.standard_normal((n_paths, time))

# Calculate constants for bond price calculation
B = (1 - np.exp(-a * T)) / a
A = ((b - (sigma ** 2) / (2 * a ** 2)) * (B - T) - (sigma ** 2) * B ** 2 / (4 * a))

# Simulate the short rate paths using the Vasicek model
for t in range(1, time + 1):
    r[:, t] = r[:, t-1] + a * (b - r[:, t-1]) * dt + sigma * np.sqrt(dt) * Z[:, t-1]

# Calculate discount factors and cash flows
discounted_rate = r[:, :-1] + r[:, 1:]
delta = np.maximum(r[:, 1:] - ceil, 0)

bank_cashflow = np.exp(-discounted_rate * dt) * (delta / 12 * amount)
customer_cashflow = dt * amount * np.ones_like(bank_cashflow)

# Calculate bond price adjustment
P = np.exp(A - B * r[:, int(T / dt)][:, np.newaxis])
customer_cashflow *= P

# Compute mean cash flows and spread
mean_bank_cashflow = np.mean(bank_cashflow.sum(axis=1))
mean_customer_cashflow = np.mean(customer_cashflow.sum(axis=1))
spread = mean_bank_cashflow / mean_customer_cashflow

# Print the calculated spread
print(f"The calculated spread is: {spread:.4f}")
