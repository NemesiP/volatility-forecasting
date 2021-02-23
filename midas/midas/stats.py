# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:09:11 2021

@author: peter
"""

import numpy as np
from scipy.special import gammaln

def loglikelihood_normal(resid, sigma2):
    """
    Negative Log-Likelihood function whereas the underlying distribution is Normal.
    
    Calculation:
        LLs = 0.5 * log(2 * pi) + 0.5 * log(sigma2) + 0.5 * (resid ** 2 / sigma2)
        LLF = Sum(lls) / len(y)
        

    Parameters
    ----------
    resid : Array
        DESCRIPTION.
    sigma2 : Array or Value
        DESCRIPTION.

    Returns
    -------
    float64
        The computed average negative loglikelihood value.

    """
    lls = -0.5 * (np.log(2*np.pi) + np.log(sigma2) + resid ** 2 / sigma2)
    return np.sum(-lls) / len(resid)

def loglikelihood_student_t(resid, sigma2, nu):
    """
    Negative Log-Likelihood function whereas the underlying distribution is Student-t.
    
    Calculation:
        lls = gammaln((nu + 1) / 2) - gammaln( nu / 2) - log(pi * (nu - 2)) / 2
        lls -= 0.5 * log(sigma2)
        lls -= ((nu + 1) / 2) * (log(1 + (resid ** 2) / (sigma2 * (nu - 2))))
        LLF = sum(-lls) / len(y)
        
    Parameters
    ----------
    resid : Array
        DESCRIPTION.
    sigma2 : Array or Value
        DESCRIPTION.
    nu : float64
        Parameter of the Student-t distribution

    Returns
    -------
    float64
        The computed average negative loglikelihood value.

    """
    lls = gammaln((nu + 1) / 2) - gammaln( nu / 2) - np.log(np.pi * (nu - 2)) / 2
    lls -= 0.5 * np.log(sigma2)
    lls -= ((nu + 1) / 2) * (np.log(1 + (resid ** 2) / (sigma2 * (nu - 2))))
    return np.sum(-lls) / len(resid)

def squared_return(df):
    """
    Function that calculate the squared returns.
    
    Calculation:
        r_t^2 = (log(C_t) - log(C_t-1))^2
    
    where,
    C_t: Close price at time t
    
    Parameters
    ----------
    df : DataFrame
        Required a column called 'Close'.

    Returns
    -------
    df : DataFrame
        Return back the original dataframe with a new column called 'Squared_Return'

    """
    r_t = np.log(df.Close) - np.log(df.Close.shift(1))
    volatility = r_t ** 2
    df['Squared_Return'] = volatility.fillna(0)
    return df

def parkinson_high_low(df):
    """
    Function that calculate the Parkinson (1980) high low estimator.
    
    Calculation:
        volatility = (log(H_t) - log(L_t))^2 / 4*log(2)
        
    where,
    H_t: Highest value at time t
    L_t: Lowest value at time t

    Parameters
    ----------
    df : DataFrame
        Required columns called 'High' and 'Low'.

    Returns
    -------
    df : DataFrame
        Return back the original dataframe with a new columns called 'High_Low_Est'.

    """
    high_low_t = np.log(df.High) - np.log(df.Low)
    volatility = (high_low_t ** 2) / (4 * np.log(2))
    df['High_Low_Est'] = volatility.fillna(0)
    return df