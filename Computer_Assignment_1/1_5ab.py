"""
This code performs Monte Carlo simulations to estimate the cashflows of an autocallable structured product 
with multiple underlying stocks. The simulation accounts for correlation between the stocks using Cholesky 
decomposition.

Steps:
1. Initialize the parameters for the simulation, including nominal value, barriers, coupon rate, 
   capital protection, initial stock price, risk-free rate, volatility, correlation, total time, 
   number of paths, number of stocks, and number of periods.
2. Create the covariance matrix and correlated noise for the simulation using Cholesky decomposition.
3. Simulate stock prices using Geometric Brownian Motion.
4. Calculate the worst-performing stock (WPS) for each path and period.
5. Calculate the cashflows based on the performance relative to the barriers.
6. Compute the average cashflow and print the result.

Parameters:
- nominal: Nominal value of the structured product.
- autocall_barrier: Autocall barrier level.
- coupon_barrier: Coupon barrier level.
- risk_barrier: Risk barrier level.
- accumulated_coupon: Coupon rate.
- capital_protection: Capital protection level.
- S0: Initial stock price.
- r: Risk-free interest rate.
- sigma: Volatility of the stocks.
- rho: Correlation between the stocks.
- T: Total time in years.
- n_paths: Number of simulation paths.
- stocks: Number of stocks.
- n_periods: Number of periods.
- dt: Time step.
"""

import numpy as np

# Parameters
nominal = 100
autocall_barrier = 100
coupon_barrier = 95
risk_barrier = 70
accumulated_coupon = 0.1
capital_protection = 0.9
S0 = 100
r = 0.03
sigma = 0.2
rho = 0.5
T = 7  # Total time in years
n_paths = 20000
stocks = 4
n_periods = 7  # Number of periods
dt = T / n_periods  # Time step

# Create covariance matrix and Cholesky decomposition
cov_matrix = np.full((stocks, stocks), rho * sigma * sigma)
np.fill_diagonal(cov_matrix, sigma * sigma)
L = np.linalg.cholesky(cov_matrix)

# Monte Carlo simulation
Z = np.random.normal(size=(stocks, n_paths, n_periods + 1))
S = np.zeros((stocks, n_paths, n_periods + 1))
S[:, :, 0] = S0
for i in range(1, n_periods + 1):
    X = L @ Z[:, :, i]
    S[:, :, i] = S[:, :, i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * X)

WPS = np.min(S, axis=0)

# Calculate cashflows
cashflow = np.zeros(n_paths)
for i in range(n_paths):
    sum_cashflow = 0
    acc_coupon = 0
    for j in range(1, n_periods + 1):
        if WPS[i, j] > autocall_barrier:
            sum_cashflow += np.exp(-r * j * dt) * (nominal + accumulated_coupon * nominal + acc_coupon)
            cashflow[i] = sum_cashflow
            break
        elif coupon_barrier <= WPS[i, j] <= autocall_barrier:
            sum_cashflow += np.exp(-r * j * dt) * (accumulated_coupon * nominal + acc_coupon)
            acc_coupon = 0
        elif risk_barrier <= WPS[i, j] < coupon_barrier:
            acc_coupon += accumulated_coupon * nominal
        elif j == n_periods:
            sum_cashflow += np.exp(-r * j * dt) * (capital_protection * nominal)
            cashflow[i] = sum_cashflow
            break
    else:
        cashflow[i] = sum_cashflow + np.exp(-r * j * dt) * nominal

print("Mean of cashflow:", np.mean(cashflow))
