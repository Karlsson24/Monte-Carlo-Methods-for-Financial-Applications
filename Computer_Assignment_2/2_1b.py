"""
This code performs Monte Carlo simulations with importance sampling to price European call and put options.
Importance sampling is used to reduce the variance of the estimator and improve the accuracy of the simulation.

Steps:
1. Initialize the parameters for the simulation, including initial stock price, risk-free rate, 
   time to maturity, volatility, and number of samples.
2. Generate standard normal random variables adjusted by the importance sampling mean shifts.
3. Simulate the stock prices at maturity using Geometric Brownian Motion.
4. Calculate the adjustment factor (likelihood ratio) for importance sampling.
5. Calculate the payoffs for call and put options, adjusted by the likelihood ratio.
6. Discount the payoffs to present value and compute the option prices.
7. Print the computed option prices.

Parameters:
- S0: Initial stock price.
- r: Risk-free interest rate.
- T: Time to maturity in years.
- my_call: Mean shift for the call option importance sampling.
- my_put: Mean shift for the put option importance sampling.
- sigma: Volatility of the stock.
- n_samples: Number of samples for the simulation.
"""

import numpy as np
from scipy.stats import norm

# Parameters
S0 = 100  # Initial stock price
r = 0.03  # Risk-free interest rate
T = 1  # Time to maturity in years
my_call = 1.84  # Mean shift for the call option importance sampling
my_put = -2.26  # Mean shift for the put option importance sampling
sigma = 0.2  # Volatility
n_samples = 20000  # Number of samples

# Generate standard normal random variables adjusted by the importance sampling mean shifts
Z_call = np.random.standard_normal(n_samples) + my_call
Z_put = np.random.standard_normal(n_samples) + my_put

# Simulate stock prices at maturity using Geometric Brownian Motion
ST_call = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z_call)
ST_put = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z_put)

# Calculate the adjustment factor (likelihood ratio) for importance sampling
fx_gx_call = np.exp(-my_call * Z_call + 0.5 * my_call**2)
fx_gx_put = np.exp(-my_put * Z_put + 0.5 * my_put**2)

# Calculate call option price at strike 130
strike_call = 130
call_payoff = np.maximum(ST_call - strike_call, 0) * fx_gx_call
discounted_call_payoff = np.exp(-r * T) * call_payoff
call_price = np.mean(discounted_call_payoff)
print(f"Call option price at strike {strike_call} is {call_price:.2f}")

# Calculate put option price at strike 70
strike_put = 70
put_payoff = np.maximum(strike_put - ST_put, 0) * fx_gx_put
discounted_put_payoff = np.exp(-r * T) * put_payoff
put_price = np.mean(discounted_put_payoff)
print(f"Put option price at strike {strike_put} is {put_price:.2f}")
