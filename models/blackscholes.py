import numpy as np #type: ignore
from scipy . stats import norm #type: ignore

def black_scholes_call (S , K , T , r , sigma ) :
    """
    Calculate European call option price using Black - Scholes - Merton model .

    Parameters :
        S ( float ) : Current stock price
        K ( float ) : Strike price
        T ( float ) : Time to maturity ( in years )
        r ( float ) : Risk - free interest rate
        sigma ( float ) : Volatility of the underlying asset

    Returns :
        float : Call option price
    """
    d1 = (np.log (S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt (T))
    d2 = d1 - sigma * np . sqrt (T)
    call = S * norm . cdf (d1) - K * np . exp (-r * T) * norm.cdf(d2)
    return call