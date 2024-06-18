"""
This script calculates the price of a European call option on a bond using the Vasicek interest rate model.
The script uses the Vasicek model to calculate the bond prices at two different maturities (T1 and T2),
then computes the call option price using the Black-Scholes formula for bonds.

Steps:
1. Initialize the parameters for the Vasicek model and option pricing.
2. Calculate the bond prices at maturities T1 and T2.
3. Compute the standard deviation of the bond price and the d1 term.
4. Use the Black-Scholes formula to calculate the call option price.
5. Print the computed call option price.

Parameters:
- r0: Initial short rate.
- t: Current time.
- a: Speed of mean reversion.
- b: Long-term mean rate.
- sigma: Volatility of the short rate.
- K: Strike price.
- T1: Maturity of the first bond.
- T2: Maturity of the second bond.
"""

import numpy as np
from scipy.stats import norm

# Parameters
r0 = 0.0137  # Initial short rate
t = 0        # Current time
a = 0.2      # Speed of mean reversion
b = 0.02     # Long-term mean rate
sigma = 0.01 # Volatility of the short rate
K = 0.8      # Strike price
T1 = 5       # Maturity of the first bond
T2 = 10      # Maturity of the second bond

# Calculate B(t,T1) and A(t,T1)
Bt1 = (1 / a) * (1 - np.exp(-a * (T1 - t)))
At1 = ((Bt1 - (T1 - t)) * (a ** 2 * b - 0.5 * sigma ** 2)) / a ** 2 - sigma ** 2 * Bt1 ** 2 / (4 * a)
Pt1 = np.exp(At1 - Bt1 * r0)

# Calculate B(t,T2) and A(t,T2)
Bt2 = (1 / a) * (1 - np.exp(-a * (T2 - t)))
At2 = ((Bt2 - (T2 - t)) * (a ** 2 * b - 0.5 * sigma ** 2)) / a ** 2 - sigma ** 2 * Bt2 ** 2 / (4 * a)
Pt2 = np.exp(At2 - Bt2 * r0)

# Calculate sigma_p and d
sigma_p = (1 / a) * (1 - np.exp(-a * (T2 - T1))) * np.sqrt((sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * (T2 - T1))))
d = (1 / sigma_p) * np.log(Pt1 / (Pt2 * K)) + 0.5 * sigma_p

# Calculate the call option price using the Black-Scholes formula
C = Pt2 * norm.cdf(d) - Pt1 * K * norm.cdf(d - sigma_p)
print(f"Call option price: {C:.4f}")
