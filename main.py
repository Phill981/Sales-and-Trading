from models.blackscholes import black_scholes_call
from models.binomial_tree import binomial_tree_call
from models.merton import merton_jump_diffusion_call
from models.monte_carlo import monte_carlo_call

import time
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2


lambda_ = 0.1
mu_J = 0.0
sigma_J = 0.1

actual_option_price = 10.5
N = 1000  

def compare_option_prices(num_runs):
    methods = {
        "Monte Carlo": monte_carlo_call,
        "Merton Jump Diffusion": merton_jump_diffusion_call,
        "Black-Scholes": black_scholes_call,
        "Binomial Tree": binomial_tree_call
    }
    
    methods_names, average_runtime, average_price_discrepancy = [], [], []
    results = {}

    for method_name, method in methods.items():
        total_price, total_runtime = 0, 0

        for _ in range(num_runs):
            start_time = time.time()
            if method_name == "Merton Jump Diffusion":
                price = method(S, K, T, r, sigma, lambda_, mu_J, sigma_J)
            else:
                price = method(S, K, T, r, sigma)
            end_time = time.time()

            total_price += price
            total_runtime += end_time - start_time

        avg_price = total_price / num_runs
        avg_runtime = total_runtime / num_runs

        results[method_name] = (avg_price, avg_runtime)
        
        methods_names.append(method_name)
        average_runtime.append(avg_runtime)
        average_price_discrepancy.append(abs(avg_price - actual_option_price))

    export_dataset = pd.DataFrame({
        "Model": methods_names,
        "Average Runtime": average_runtime,
        "Average Price Discrepancy": average_price_discrepancy
    })
    export_dataset.to_excel("data.xlsx", index=False)

    return results

def plot_runtime_vs_time_steps(max_steps):
    time_steps = range(10, max_steps + 1, 10)
    runtimes = []

    for N in time_steps:
        start_time = time.time()
        binomial_tree_call(S, K, T, r, sigma, N)
        end_time = time.time()
        runtimes.append(end_time - start_time)

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, runtimes, marker='o')
    plt.title('Runtime vs. Number of Time Steps in Binomial Tree Model')
    plt.xlabel('Number of Time Steps (N)')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.savefig("./img/runtime_vs_timesteps.png")
    plt.show()


def plot_pricing_error_vs_volatility(volatility_range):
    methods = {
        "Monte Carlo": monte_carlo_call,
        "Merton Jump Diffusion": merton_jump_diffusion_call,
        "Black-Scholes": black_scholes_call,
        "Binomial Tree": binomial_tree_call
    }

    volatilities = np.linspace(0.0, volatility_range, 100)
    pricing_errors = {name: [] for name in methods.keys()}

    for vol in volatilities:
        for method_name, error_list in pricing_errors.items():
            if method_name == "Merton Jump Diffusion":
                price = merton_jump_diffusion_call(S, K, T, r, vol, lambda_, mu_J, sigma_J)
            else:
                price = methods[method_name](S, K, T, r, vol)
            pricing_error = abs(price - actual_option_price)
            error_list.append(pricing_error)

    plt.figure(figsize=(10, 5))
    for method_name, errors in pricing_errors.items():
        plt.plot(volatilities, errors, label=method_name)

    plt.title('Pricing Error vs. Volatility for Each Model')
    plt.xlabel('Volatility')
    plt.ylabel('Pricing Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("./img/pricing_error_volatility.png")
    plt.show()

compare_option_prices(N)
plot_runtime_vs_time_steps(max_steps=1000)
plot_pricing_error_vs_volatility(volatility_range=1.0)
