import numpy as np #type: ignore

def monte_carlo_call (S, K, T, r, sigma, simulations =10000) :
    """
    Calculate European call option price using Monte Carlo Simulation .

        Parameters :
        S (float) : Current stock price
        K (float) : Strike price
        T (float) : Time to maturity (in years)
        r (float) : Risk - free interest rate
        sigma (float) : Volatility of the underlying asset
        simulations (int) : Number of simulated paths

    Returns :
        float : Call option price
    """
    np. random.seed(0)
    Z = np.random.standard_normal ( simulations )
    ST = S * np.exp((r - 0.5 * sigma **2) * T + sigma * np.sqrt(T) * Z
    )
    payoffs = np.maximum(ST - K, 0)
    call = np.exp(-r * T) * np.mean(payoffs)
    return call