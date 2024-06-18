"""
Task: Calculate the European Call and Put option price using the Black-Scholes formula

"""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm  # Normal distribution functions

def blackscholes(S0, T, r, sigma, K):
    """
    Calculate the Black-Scholes price for a European call option.

    Parameters:
    S0 (float): Initial stock price
    T (float): Time to maturity (years)
    r (float): Risk-free interest rate
    sigma (float): Volatility
    K (float): Strike price

    Returns:
    float: Call option price
    """
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = (S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return call_price


# Parameters
K = [70, 100, 130]  # Strike prices
S0 = 100  # Initial stock price
r = 0.03  # Risk-free rate
sigma = 0.2  # Volatility
T = 1  # Time to maturity (years)

# List to store results
results = []

# Calculate and store option prices
for strike in K:
    price_call = blackscholes(S0, T, r, sigma, strike)
    price_put = price_call - S0 + strike * math.exp(-r * T)
    results.append([strike, price_call, price_put])

# Convert list to DataFrame
results_df = pd.DataFrame(results, columns=['Strike Price', 'Call Option Price', 'Put Option Price'])

# Display the results
print(results_df)
