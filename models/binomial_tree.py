import numpy as np #type: ignore

def binomial_tree_call (S, K, T, r, sigma, N =100) :
    """
    Calculate European call option price using Binomial Tree model .
        Parameters :
        S (float) : Current stock price
        K (float) : Strike price
        T (float) : Time to maturity (in years)
        r (float) : Risk - free interest rate
        sigma (float) : Volatility of the underlying asset
        N (int) : Number of time steps
    
    Returns :
        float : Call option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range (N + 1)
    ]
    option_values = [max(price - K , 0) for price in asset_prices]
    for i in range (N - 1, -1, -1) :
        for j in range (i + 1) :
            option_values[j] = np.exp (-r * dt) * (p * option_values [j +
                1] + (1 - p) * option_values [j])
    return option_values [0]