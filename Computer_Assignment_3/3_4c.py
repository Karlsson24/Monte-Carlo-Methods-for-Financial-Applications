"""
This script calculates the Greeks (Delta, Gamma, and Theta) for TSLA put options using a binomial tree model for American options.
The script performs the following steps:

1. Load the put option data from a CSV file.
2. Filter the put options to find the one with a specified strike price.
3. Construct a binomial tree for the underlying asset price.
4. Calculate the option price using the binomial tree model.
5. Calculate the implied volatility that matches the market price.
6. Calculate the Greeks (Delta, Gamma, and Theta) using the binomial tree.
7. Print the calculated Greeks.

Parameters:
- filväg_put: Path to the CSV file containing put option data.
- S0: Initial stock price.
- r: Risk-free interest rate.
- T: Time to maturity in years.
- K: Strike price of the put option.
- n_perioder: Number of time steps in the binomial tree.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Load the put option data from a CSV file
filväg_put = r'C:\Users\Andre\OneDrive\Dokument\IE\Monte_Carlo_Simulation\TSLA_puts.csv'
puts_data = pd.read_csv(filväg_put, sep=';', index_col='Contract Name')

# Define functions for constructing a binomial tree and pricing options
def konstruera_binomialtrad(S, T, r, vol, N):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    trad = np.zeros((N + 1, N + 1))
    trad[0, 0] = S
    for i in range(1, N + 1):
        trad[i, 0] = trad[0, 0] * u ** i
        trad[i, 1:i + 1] = trad[i - 1, 0:i] * d
    return trad

def optionspris(pris_trad, r, K, T, n_perioder, vol):
    dt = T / n_perioder
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p
    V = np.zeros((n_perioder + 1, n_perioder + 1))
    E = np.maximum(K - pris_trad, 0)
    V[-1, :] = E[-1, :]
    for i in range(n_perioder - 1, -1, -1):
        V[i, :i + 1] = np.exp(-r * dt) * (p * V[i + 1, :i + 1] + q * V[i + 1, 1:i + 2])
        V[i, :i + 1] = np.maximum(V[i, :i + 1], E[i, :i + 1])
    return V[0, 0], V

def implicit_volatilitet(S, K, r, T, marknadspris, n_perioder=52):
    def malfunktion(sigma):
        pris_trad = konstruera_binomialtrad(S, T, r, sigma, n_perioder)
        pris, _ = optionspris(pris_trad, r, K, T, n_perioder, sigma)
        return pris - marknadspris

    try:
        impl_vol = brentq(malfunktion, 1e-6, 5)
    except ValueError:
        impl_vol = np.nan
    return impl_vol

def delta_trad(Vi1, Vi2, P1, P2):
    return (Vi1 - Vi2) / (P1 - P2)

def gamma_trad(Vi, pris):
    d1 = delta_trad(Vi[2, 0], Vi[2, 1], pris[2, 0], pris[2, 1])
    d2 = delta_trad(Vi[2, 1], Vi[2, 2], pris[2, 1], pris[2, 2])
    return (d1 - d2) / ((pris[2, 0] - pris[2, 2]) / 2)

def theta_t0(Vi, dt):
    return (Vi[2, 1] - Vi[0, 0]) / (2 * dt) / 250

# Parameters for the calculation
S0 = 165.8
r = 0.05
T = (pd.to_datetime('2024-09-20') - pd.to_datetime('2024-03-15')).days / 365.0
K = 165
n_perioder = 52

# Calculate the mid price of the put options
puts_data['Mitt Pris'] = (puts_data['Bid'] + puts_data['Ask']) / 2
marknadspris = puts_data[puts_data['Strike'] == K]['Mitt Pris'].values[0]

# Calculate the implied volatility
vol = implicit_volatilitet(S0, K, r, T, marknadspris, n_perioder)

# Construct the binomial tree and calculate the option price
pris_trad = konstruera_binomialtrad(S0, T, r, vol, n_perioder)
optionspris_0, V = optionspris(pris_trad, r, K, T, n_perioder, vol)

# Calculate the Greeks
delta = delta_trad(V[1, 0], V[1, 1], pris_trad[1, 0], pris_trad[1, 1])
gamma = gamma_trad(V, pris_trad)
theta = theta_t0(V, T / n_perioder)

# Print the calculated Greeks
print("Greker för K = 165:")
print(f"Delta: {delta}")
print(f"Gamma: {gamma}")
print(f"Theta (daglig): {theta}")

