"""
This code performs Monte Carlo simulations to estimate the prices of Basket options
on a portfolio of two stocks. The stock prices are simulated using Geometric Brownian Motion and
Cholesky decomposition to account for correlations between the stocks.

Steps:
1. Initialize the parameters for the simulation, including the number of paths, number of stocks,
   risk-free rate, volatility, and strike prices.
2. Construct the covariance matrix based on the volatility and correlation between the stocks.
3. Perform Cholesky decomposition on the covariance matrix.
4. Generate random variables for the simulation.
5. Simulate stock prices using the Cholesky decomposition.
6. Calculate the payoff for call and put options based on the average price of the two stocks.
7. Calculate the average option prices from the simulated payoffs.
8. Store and display the results in a pandas DataFrame.

Parameters:
- S0: Initial stock price.
- n_paths: Number of simulation paths.
- stocks: Number of stocks.
- r: Risk-free interest rate.
- sigma: Volatility of the stock.
- correlation: Correlation between the two stocks.
- K: Strike prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky

# Parameters
S0 = 100  # Initial stock price
n_paths = 20000  # Number of paths to simulate
stocks = 2  # Number of stocks
r = 0.03  # Risk-free rate
sigma = 0.2  # Volatility
correlation = 0  # Correlation between the two stocks
K = [70, 100, 130]  # Strike prices
dt = 1  # Time step

# Covariance matrix based on the volatility and correlation
cov_matrix = np.array([[sigma ** 2, correlation * sigma * sigma],
                       [correlation * sigma * sigma, sigma ** 2]])

# Cholesky decomposition of the covariance matrix
A = cholesky(cov_matrix)

# Generate random variables for the simulation
Z = np.random.standard_normal((n_paths, stocks))

# Simulate stock prices using the Cholesky decomposition
S = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * np.dot(Z, A.T))

# Calculate option prices
results = []
for strike in K:
    # Calculate payoffs for call and put options
    avg_price = np.mean(S, axis=1)
    C_payoff = np.exp(-r * dt) * np.maximum(avg_price - strike, 0)
    P_payoff = np.exp(-r * dt) * np.maximum(strike - avg_price, 0)

    # Average payoffs
    C_n = np.mean(C_payoff)
    P_n = np.mean(P_payoff)

    # Store results
    results.append([strike, C_n, P_n])

# Create a DataFrame to display the results
df = pd.DataFrame(results, columns=['Strike', 'Call Option', 'Put Option'])
print(df)
