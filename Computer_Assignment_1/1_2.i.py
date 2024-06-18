"""
This code calculates the prices of European call options using a modified Black-Scholes formula
with adjusted volatility over weekly time steps. The results are presented for different strike prices.

Constants:
1. risk_free_rate: The risk-free interest rate.
2. strike_prices: List of different strike prices.
3. volatility: The volatility of the stock.
4. initial_stock_price: The initial price of the stock.
5. num_weeks: The number of weeks until option maturity.
6. time_to_maturity: The time to maturity of the option in years.

Steps:
1. Calculate time steps and other time-related parameters.
2. Compute the adjusted volatility (sigma_tak_squared).
3. Calculate the delta parameter.
4. Loop through each strike price to calculate the option prices using the Black-Scholes formula.
5. Adjust the option prices using a correction factor.
6. Store and display the results in a pandas DataFrame.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Constants and parameters
risk_free_rate = 0.03  # Risk-free interest rate
strike_prices = [70, 100, 130]  # Strike prices
volatility = 0.2  # Volatility
initial_stock_price = 100  # Initial stock price
num_weeks = 52  # Number of weeks
time_to_maturity = 1  # Time to maturity (in years)
time_step = 1 / num_weeks  # Time step
times = np.arange(1, num_weeks + 1) * time_step  # Time points
mean_time = np.mean(times)  # Mean time
total_time = time_step * sum(times)  # Total time
tau = times  # Time points for calculation

# Calculate sigma_tak_squared
sigma_tak_squared = (volatility ** 2 / (num_weeks ** 2 * total_time) *
                     sum((2 * i - 1) * tau[num_weeks - i] for i in range(1, num_weeks + 1)))

# Delta parameter
delta = 0.5 * (volatility ** 2) - 0.5 * sigma_tak_squared

# List to store results
results = []

# Calculate option prices for each strike price
for strike in strike_prices:
    d1 = (np.log(initial_stock_price / strike) +
          (risk_free_rate - delta + 0.5 * sigma_tak_squared) * total_time) / np.sqrt(sigma_tak_squared * total_time)
    d2 = d1 - np.sqrt(sigma_tak_squared * total_time)

    # Black-Scholes formula
    BS_formula = (np.exp(-delta * total_time) * initial_stock_price * norm.cdf(d1) -
                  np.exp(-risk_free_rate * total_time) * strike * norm.cdf(d2))

    # Correction factor for the option price
    correction_factor = np.exp(-risk_free_rate * (time_to_maturity - total_time))

    # Calculate the final option price
    price = correction_factor * BS_formula

    # Append the result
    results.append([strike, price])

# Create a DataFrame to display the results
df = pd.DataFrame(results, columns=['Strike', 'Call Option'])
print(df)
