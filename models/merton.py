import numpy as np#type: ignore

def merton_jump_diffusion_call (S , K , T , r , sigma , lambda_ , mu_J ,
    sigma_J , simulations =10000) :
    """
    Calculate European call option price using Merton Jump Diffusion Model .

    Parameters :
        S (float)  : Current stock price
        K (float) : Strike price
        T (float) : Time to maturity (in years)
        r (float) : Risk - free interest rate
        sigma (float) : Volatility of the underlying asset
        lambda_ (float) : Jump intensity
        mu_J (float) : Mean of jump size 
        sigma_J (float) : Standard deviation of jump size
        simulations (int) : Number of simulated paths
    Returns :
        float : Call option price
    """
    np.random.seed(0) 

    Y = np.random.normal(mu_J , sigma_J , simulations)
    drift = (r - 0.5 * sigma **2 - lambda_ * (np.exp(mu_J + 0.5 * sigma_J **2) -1)) * T
    diffusion = sigma * np.sqrt(T) * np.random.standard_normal(simulations)
    ST = S * np.exp(drift + diffusion + Y)
    payoffs = np.maximum (ST - K, 0)
    call = np.exp(-r * T) * np.mean(payoffs)
    return call
