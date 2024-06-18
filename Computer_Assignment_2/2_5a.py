"""
This script uses the Vasicek model to simulate and plot the zero-coupon bond prices over time.
The Vasicek model describes the evolution of interest rates and is used to calculate the bond prices.
The script calculates the bond prices for different maturities and plots the results.

Steps:
1. Initialize the parameters for the Vasicek model.
2. Calculate the bond price for each maturity.
3. Compute the continuously compounded zero rate (Zt) for each maturity.
4. Store the results and plot the bond prices over time.

Parameters:
- r0: Initial short rate.
- t: Current time.
- a: Speed of mean reversion.
- b: Long-term mean rate.
- sigma: Volatility of the short rate.
- K: Strike price (not used in this script).
- T1, T2: Maturities (not used in this script).
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Vasicek model
r0 = 0.0137  # Initial short rate
t = 0        # Current time
a = 0.2      # Speed of mean reversion
b = 0.02     # Long-term mean rate
sigma = 0.01 # Volatility of the short rate
K = 0.8      # Strike price (not used in this script)
T1 = 5       # Maturity 1 (not used in this script)
T2 = 10      # Maturity 2 (not used in this script)

# Calculate bond prices and zero rates for different maturities
result = []
result.append(r0)
for T in range(1, 11):
    B = (1 / a) * (1 - np.exp(-a * (T - t)))
    A = (((B - (T - t)) * (a**2 * b - 0.5 * sigma**2)) / (a**2)) - (sigma**2 * B**2 / (4 * a))
    P = np.exp(A - B * r0)
    Zt = -np.log(P) / T
    result.append(Zt)

# Print the results
print(result)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(0, 11), result, marker='o')
plt.title('Continuously Compounded Zero Rates over Time')
plt.xlabel('Time (T)')
plt.ylabel('Zero Rate (Zt)')
plt.grid(True)
plt.show()
