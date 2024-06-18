"""
This script calculates the price of a strangle option strategy on TSLA using data from CSV files for call and put options. 
The strangle price is the sum of the ask prices of a call option with a given strike price and a put option with a different 
strike price. The script performs the following steps:

1. Load the call and put option data from CSV files.
2. Filter the options data to find the specified call and put options.
3. Calculate the strangle price as the sum of the ask prices of the specified call and put options.
4. Print the calculated strangle price.

Parameters:
- file_path_call: Path to the CSV file containing call option data.
- file_path_put: Path to the CSV file containing put option data.
- S0: Initial stock price.
- r: Risk-free interest rate.
- K_call: Strike price of the call option.
- K_put: Strike price of the put option.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm # Provides the normal pdf, cdf, and inverse cdf (ppf)
from scipy.stats.mstats import gmean # Provides the geometric mean

# File paths for the call and put option data
file_path_call = r'C:\Users\Andre\OneDrive\Dokument\IE\Monte_Carlo_Simulation\TSLA_calls.csv'
file_path_put = r'C:\Users\Andre\OneDrive\Dokument\IE\Monte_Carlo_Simulation\TSLA_puts.csv'

# Load the call and put option data from CSV files
call_data = pd.read_csv(file_path_call, sep=';', index_col='Contract Name')
put_data = pd.read_csv(file_path_put, sep=';', index_col='Contract Name')

# Parameters for the strangle option strategy
S0 = 165.8
r = 0.05
K_put = 135
K_call = 195

# Filter the options data to find the specified call and put options
call_option = call_data[call_data['Strike'] == K_call]
put_option = put_data[put_data['Strike'] == K_put]

# Check if the specified call and put options exist in the data
if call_option.empty:
    print(f"No call option found with strike price {K_call}")
else:
    call_price = call_option['Ask'].iloc[0]

if put_option.empty:
    print(f"No put option found with strike price {K_put}")
else:
    put_price = put_option['Ask'].iloc[0]

# Calculate the strangle price as the sum of the ask prices of the specified call and put options
if not call_option.empty and not put_option.empty:
    strangle_price = call_price + put_price
    print(f"Strangle price: {strangle_price}")
