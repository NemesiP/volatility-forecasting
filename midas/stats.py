# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:09:11 2021

@author: peter
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import t
import pandas as pd

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


def dm_test(act, pred1, pred2, h = 1, degree = 0, plot = False):
    e1_lst, e2_lst, d_lst = [], [], []
    
    act_lst = np.asarray(act)
    pred1_lst = np.asarray(pred1)
    pred2_lst = np.asarray(pred2)
    
    def family_of_loss_func(actual, predicted, degree):
        """
        Implemented from:
        Patton, A. J., 2011. Volatility forecasting comparison using imperfect 
        volatility proxies, Journal of Econometrics 160, 246-256.
        """
        if degree == -2:
            # QLIKE
            loss = actual / predicted - np.log(actual / predicted) - 1
        elif degree == -1:
            loss = predicted - actual + actual * np.log(actual / predicted)
        else:
            # MSE if degree = 0
            loss = (np.sqrt(actual) ** (2 * degree + 4) - predicted ** (degree + 2)) / ((degree + 1) * (degree + 2))
            loss -= (1 / (degree + 1)) * (predicted ** (degree + 1)) * (actual - predicted)
        return loss
    
    T = float(len(act_lst))
    for a, p1, p2 in zip(act_lst, pred1_lst, pred2_lst):
        e1_lst.append(family_of_loss_func(a, p1, degree))
        e2_lst.append(family_of_loss_func(a, p2, degree))
        
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)
        
    d_mean = np.mean(d_lst)
    
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return autoCov / T
    
    gamma = []
    
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, d_mean))
    
    V_d = (gamma[0] + 2 * np.sum(gamma[1:])) / T
    DM_stat = d_mean * V_d ** -0.5

    
    p_value = 2 * t.cdf(-np.abs(DM_stat), df = T - 1)
    
    if plot == True:
        print('DM = ', DM_stat, '\nDM p_value', p_value)
    return DM_stat, p_value

def family_of_loss_func(actual, predicted, degree):
    """
    Implemented from:
    Patton, A. J., 2011. Volatility forecasting comparison using imperfect 
    volatility proxies, Journal of Econometrics 160, 246-256.
    """
    if degree == -2:
        # QLIKE
        loss = actual / predicted - np.log(actual / predicted) - 1
    elif degree == -1:
        loss = predicted - actual + actual * np.log(actual / predicted)
    else:
        # MSE if degree = 0
        loss = (np.sqrt(actual) ** (2 * degree + 4) - predicted ** (degree + 2)) / ((degree + 1) * (degree + 2))
        loss -= (1 / (degree + 1)) * (predicted ** (degree + 1)) * (actual - predicted)
    return loss

def panel_DM(act, pred1, pred2, degree = 0):
    """
    Implemented from:
    Timmermann, A., Zhu, Y., 2019. Comparing Forecasting Performance with Panel Data
    """
    l1 = family_of_loss_func(act, pred1, degree)
    l2 = family_of_loss_func(act, pred2, degree)
    d12 = l1 - l2
    d12n = np.nansum(d12, axis = 1)
    n = np.sum(~np.isnan(d12), axis = 1)
    T = d12.shape[0]
    nT = np.sum(~np.isnan(d12))
    hat_d12 = np.nansum(d12, axis = 1) / np.sum(~np.isnan(d12), axis = 1)
    R12 = np.sqrt(n) * hat_d12
    hat_R0 = np.nansum(R12) / T
    hat_R1 = np.nansum(R12[1:]) / (T - 1)
    hat_R11 = np.nansum(R12[:-1]) / (T - 1)
    g0 = np.nansum((R12 - hat_R0) * (R12 - hat_R0)) / T
    g1 = 2 * np.nansum((R12[1:] - hat_R11) * (R12[:-1] - hat_R1)) / (T - 1)
    sig = np.sqrt(g0 + g1)
    DM = np.nansum(d12) / (np.sqrt(nT) * sig)
    p_value = 2 * t.cdf(-np.abs(DM), df = nT - 1)
    return DM, p_value
