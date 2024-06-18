"""
Task: Study how the price for the call and put option (for K = 100) depends on the volatility, time to maturity and the
interest rate by plotting the option values against these variables in three separate plots. Vary
• the volatility in the range [1%,50%] with steps of size 1%,
• the time to maturity in the range [1/12,5] with steps of size 1/12, and
• the interest rate in the range [0, 0.1] with steps of size 0.01.

"""


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm  # Normal and lognormal pdf, cdf, and inverse cdf/pdf
from scipy.stats.mstats import gmean  # Geometric mean


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
S0 = 100  # Initial stock price
interest = 0.03  # Risk-free interest rate
T = 1  # Time to maturity (years)
K = 100  # Strike price
sigma = 0.2  # Volatility

# Ranges for different parameters
Vola = np.arange(0.01, 0.51, 0.01)
Maturity = np.arange(1 / 12, 5 + 1 / 12, 1 / 12)
rate = np.arange(0, 0.1, 0.01)

# Lists to store calculated option prices
call_prices_Vol = []
put_prices_Vol = []

# Calculate option prices for varying volatility
for i in Vola:
    price_call = blackscholes(S0, T, interest, i, K)
    price_put = price_call - S0 + K * math.exp(-interest * T)
    call_prices_Vol.append(price_call)
    put_prices_Vol.append(price_put)

# Lists to store calculated option prices for varying maturity
call_prices_Mat = []
put_prices_Mat = []

# Calculate option prices for varying maturity
for time in Maturity:
    price_call = blackscholes(S0, time, interest, sigma, K)
    price_put = price_call - S0 + K * math.exp(-interest * time)
    call_prices_Mat.append(price_call)
    put_prices_Mat.append(price_put)

# Lists to store calculated option prices for varying interest rates
call_prices_Rate = []
put_prices_Rate = []

# Calculate option prices for varying interest rates
for r in rate:
    price_call = blackscholes(S0, T, r, sigma, K)
    price_put = price_call - S0 + K * math.exp(-r * T)
    call_prices_Rate.append(price_call)
    put_prices_Rate.append(price_put)

# Plot 1: Volatility vs. Option Prices
plt.figure(1, figsize=(10, 5))
plt.plot(Vola, call_prices_Vol, label='Call Option Price', marker='o')
plt.plot(Vola, put_prices_Vol, label='Put Option Price', marker='x')
plt.title('Option Prices vs. Volatility')
plt.xlabel('Volatility (sigma)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)

# Plot 2: Maturity vs. Option Prices
plt.figure(2, figsize=(10, 5))
plt.plot(Maturity, call_prices_Mat, label='Call Option Price', marker='o')
plt.plot(Maturity, put_prices_Mat, label='Put Option Price', marker='x')
plt.title('Option Prices vs. Maturity')
plt.xlabel('Maturity (Years)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)

# Plot 3: Interest Rate vs. Option Prices
plt.figure(3, figsize=(10, 5))
plt.plot(rate, call_prices_Rate, label='Call Option Price', marker='o')
plt.plot(rate, put_prices_Rate, label='Put Option Price', marker='x')
plt.title('Option Prices vs. Interest Rate')
plt.xlabel('Interest Rate (r)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)

# Display all figures
plt.show()
