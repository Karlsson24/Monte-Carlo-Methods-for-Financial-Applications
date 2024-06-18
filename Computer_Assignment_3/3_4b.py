"""
This script calculates the implied volatilities of TSLA put options using the binomial tree model for American options.
The script performs the following steps:

1. Load the put option data from a CSV file.
2. Filter the put options to include only those with strike prices between 100 and 230.
3. Calculate the mid price of the put options.
4. Define a function to price American put options using the binomial tree model.
5. Define a function to calculate the implied volatility by solving the objective function.
6. Calculate the implied volatilities for the filtered put options.
7. Plot the implied volatilities against the strike prices.
8. Print the filtered put option data with the calculated implied volatilities.

Parameters:
- file_path_put: Path to the CSV file containing put option data.
- S0: Initial stock price.
- r: Risk-free interest rate.
- T: Time to maturity in years.
- N: Number of time steps in the binomial tree.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Load the put option data from a CSV file
file_path_put = r'C:\Users\Andre\OneDrive\Dokument\IE\Monte_Carlo_Simulation\TSLA_puts.csv'
put_data = pd.read_csv(file_path_put, sep=';', index_col='Contract Name')

# Filter the put options to include only those with strike prices between 100 and 230
put_data_filtered = put_data[(put_data['Strike'] >= 100) & (put_data['Strike'] <= 230)].copy()

# Calculate the mid price of the put options
put_data_filtered['Mid Price'] = (put_data_filtered['Bid'] + put_data_filtered['Ask']) / 2


def binomial_tree_american_put(S0, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    ST = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    # Initialize option values at maturity
    option_values = np.maximum(K - ST, 0)

    # Step backwards through the tree
    for i in range(N - 1, -1, -1):
        ST = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        option_values = np.exp(-r * dt) * (p * option_values[1:i + 2] + (1 - p) * option_values[0:i + 1])
        option_values = np.maximum(option_values, K - ST)

    return option_values[0]


def implied_volatility_binomial(S0, K, T, r, market_price, N=52):
    objective_function = lambda sigma: market_price - binomial_tree_american_put(S0, K, T, r, sigma, N)
    implied_vol = brentq(objective_function, 1e-6, 5.0)
    return implied_vol


# Parameters for the calculation
S0 = 165.8
r = 0.05
T = (pd.to_datetime("2024-09-20") - pd.to_datetime("2024-03-15")).days / 365.25

# Calculate the implied volatilities for the filtered put options
put_data_filtered['Implied Volatility'] = put_data_filtered.apply(
    lambda row: implied_volatility_binomial(S0, row['Strike'], T, r, row['Mid Price']), axis=1)

# Plot the implied volatilities against the strike prices
plt.figure(figsize=(10, 6))
plt.plot(put_data_filtered['Strike'], put_data_filtered['Implied Volatility'], marker='o')
plt.title('Implied Volatilities of TSLA Put Options')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.grid(True)
plt.show()

# Print the filtered put option data with the calculated implied volatilities
print(put_data_filtered)
