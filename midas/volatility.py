# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:50:49 2021

@author: peter
"""
import numpy as np
import pandas as pd
from base import BaseModel, GarchBase
from stats import loglikelihood_normal, loglikelihood_student_t
from weights import Beta
from helper_functions import create_matrix
from datetime import datetime, timedelta
import time
import scipy.stats as stats

class MIDAS(BaseModel):
    def __init__(self, lag = 22, plot = True, *args):
        self.lag = lag
        self.plot = plot
        self.args = args

    
    def initialize_params(self, X):
        """
        This function is about to create the initial parameters. 
        Return a sequance of 1.0 value that has the necessary length.

        Parameters
        ----------
        X : DataFrame
            Pandas dataframe that contains all the regressors.

        Returns
        -------
        init_params: numpy.array
            Numpy array that contain the required amount of initial parameters.

        """
        self.init_params = np.linspace(1.0, 1.0, int(1.0 + X.shape[1] * 2.0))
        return self.init_params
    
    def model_filter(self, params, x):
        """
        This function is about to create the model's equation.

        Parameters
        ----------
        params : numpy.array
            Numpy array that contain the required amount of parameters.
        x : Dictionary
            Dictionary that contains all the lagged regressors.

        Returns
        -------
        model : numpy.array
            Numpy array that return the values from the specification.

        """
        model = params[0]
        for i in range(1, len(x) + 1):
            model += params[2 * i - 1] * Beta().x_weighted(x['X{num}'.format(num = i)], [1.0, params[2 * i]])
        
        return model
    
    def loglikelihood(self, params, X, y):
        """
        This function is about to calculate the negative loglikelihood function's value.

        Parameters
        ----------
        params : numpy.array
            Numpy array that contain the required amount of parameters.
        X : DataFrame
            Pandas dataframe that contain all the regressors.
        y : pandas.Series or numpy.array
            Sequance that contains the dependent variable.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        X = create_matrix(X, self.lag)
        return np.sum((y - self.model_filter(params, X)) ** 2)             
    
    def predict(self, X):
        X = create_matrix(X, self.lag)
        return self.model_filter(self.optimized_params, X)
    
    def simulate(self, params = [2.0, 0.5, 5.0], lag = 12, num = 500):
        y = np.zeros(num)
        x = np.exp(np.cumsum(np.random.normal(0.5, 2, num) / 100))
        alpha, beta, theta = params[0], params[1], params[2]
        
        for i in range(num):
            if i < lag:
                y[i] = alpha
            else:
                y[i] = alpha + beta * Beta().x_weighted(x[i - lag : i][::-1].reshape((1, lag)), [1.0, theta])
        
        return x, y
 
class MIDAS_sim(BaseModel):
    def __init__(self, lag = 22, plot = True, *args):
        self.lag = lag
        self.plot = plot
        self.args = args
        
    def initialize_params(self, X):
        self.init_params = np.linspace(1, 1, 3)
        return self.init_params
    
    def model_filter(self, params, X, y):
        if isinstance(y, int) or isinstance(y, float):
            T = y
        else:
            T = len(y)
        model = np.zeros(T)
        
        for i in range(T):
            model[i] = params[0] + params[1] * Beta().x_weighted(X[i * self.lag : (i + 1) * self.lag].reshape((1, self.lag)), [1.0, params[2]])
        
        return model
    
    def loglikelihood(self, params, X, y):
        return np.sum((y - self.model_filter(params, X, y)) ** 2)
    

    def simulate(self, params = [0.1, 0.3, 4.0], num = 500, K = 22):
        X = np.zeros(num * K)
        y = np.zeros(num)
        
        for i in range(num * K):
            if i == 0:
                X[i] = np.random.normal()
            else:
                X[i] = 0.9 * X[i - 1] + np.random.normal()
                
        for i in range(num):
            y[i] = params[0] + params[1] * Beta().x_weighted(X[i * K : (i + 1) * K].reshape((1, K)), [1.0, params[2]]) + np.random.normal(scale = 0.7**2)
        
        return X, y  
    
    def create_sims(self, number_of_sims = 500, length = 500, K = 22, params = [0.1, 0.3, 4.0]):
        lls, b0, b1, th, runtime = np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims)
        
        for i in range(number_of_sims):
            np.random.seed(i)
            X, y = self.simulate(params = params, num = length, K = K)
            start = time.time()
            self.fit(['pos', 'pos', 'pos'], X, y)
            runtime[i] = time.time() - start
            lls[i] = self.opt.fun
            b0[i], b1[i], th[i] = self.optimized_params[0], self.optimized_params[1], self.optimized_params[2]
            
        return pd.DataFrame(data = {'LogLike': lls, 
                                    'Beta0': b0, 
                                    'Beta1': b1, 
                                    'Theta':th})
    
    def forecasting(self, X, k = 10):
        X_n = np.zeros(k * 22)
        
        for i in range(k * 22):
            if i == 0:
                X_n[i] = 0.9 * X[-1] + np.random.normal()
            else:
                X_n[i] = 0.9 * X_n[i - 1] + np.random.normal()
                
        try:
            y_hat = self.model_filter(self.optimized_params, X_n, k)
        except:
            params = input('Please give the parameters:')
            
        return X_n, y_hat    

class GARCH(BaseModel):
    def __init__(self, plot = True, *args):
        self.plot = plot
        self.args = args
        
    def initialize_params(self, y):
        self.init_params = np.asarray([0.0, 0.05, 0.02, 0.95])
        return self.init_params
    
    def model_filter(self, params, y):
        sigma2 = np.zeros(len(y))
        resid = y - params[0]
        for i in range(len(y)):
            if i == 0:
                sigma2[i] = params[1] / (1 - params[2] - params[3])
            else:
                sigma2[i] = params[1] + params[2] * resid[i - 1] ** 2 + params[3] * sigma2[i - 1]
        return sigma2
    
    def loglikelihood(self, params, y):
        sigma2 = self.model_filter(params, y)
        resid = y - params[0]
        return loglikelihood_normal(resid, sigma2)
    
    def simulate(self, params = [0.0, 0.2, 0.2, 0.6], num = 500):
        y = np.zeros(num)
        state = np.zeros(num)
        for i in range(num):
            if i == 0:
                state[i] = params[1] / (1 - params[2] - params[3])
            else:
                state[i] = params[1] + params[2] * y[i - 1] * y[i - 1] + params[3] * state[i - 1]
            y[i] = stats.norm.rvs(loc = params[0], scale = np.sqrt(state[i]))
        return y, state
    
    def predict(self, X):
        return self.model_filter(self.optimized_params, X)
    
class T_GARCH(BaseModel):
    def __init__(self, plot = True, *args):
        self.plot = plot
        self.args = args
        
    def initialize_params(self, y):
        self.init_params = np.asarray([0.1, 0.02, 0.95, 3.75])
        return self.init_params
    
    def model_filter(self, params, y):
        sigma2 = np.zeros(len(y))
        resid = y - params[0]
        for i in range(len(y)):
            if i == 0:
                sigma2[i] = params[1] / (1 - params[2] - params[3])
            else:
                sigma2[i] = params[1] + params[2] * resid[i - 1] ** 2 + params[3] * sigma2[i - 1]
        return sigma2
    
    def loglikelihood(self, params, y):
        sigma2 = self.model_filter(params, y)
        resid = y - params[0]
        nu = params[4]
        return loglikelihood_student_t(resid, sigma2, nu)
    
    def simulate(self, params = [0.0, 0.2, 0.2, 0.6, 3.0], num = 500):
        y = np.zeros(num)
        state = np.zeros(num)
        for i in range(num):
            if i == 0:
                state[i] = params[1] / (1 - params[2] - params[3])
            else:
                state[i] = params[1] + params[2] * y[i - 1] * y[i - 1] + params[3] * state[i - 1]
            y[i] = stats.t.rvs(params[4], loc = params[3], scale = np.sqrt(state[i]))
        return y, state
    
    def predict(self, X):
        return self.model_filter(self.optimized_params, X)
    
class GARCH_MIDAS(BaseModel):
    def __init__(self, lag = 22, plot = True, *args):
        self.lag = lag
        self.plot = plot
        self.args = args
        
    def initialize_params(self, X):
        """
        This function is about to create the initial parameters and
        collect the indexes of monthly and daily data columns.
        

        Parameters
        ----------
        X : DataFrame
            Pandas dataframe that contains all the regressors

        Returns
        -------
        init_params: numpy.array
            Numpy array that contain the required amount of initial parameters.

        """
        # Empty array for the column indexes of daily regressors
        daily_index = np.array([])
        # Empty array for the column indexes of monthly regressors
        monthly_index = np.array([])
        # Initial GARCH parameters
        garch_params = np.array([0.05, 0.05, 0.02, 0.95])
        # An array where there will be the required amount of parameter for the modeling.
        midas_params = np.array([1.0])
        for i in range(X.shape[1]):
            # Calculate the ratio of unique observation divided by the number of whole observations
            ratio = X.iloc[:, i].unique().shape[0] / X.shape[0]
            # Let's assume that the ratio for monthly observation will be close to 12/365,
            # so I set the critial point to 0.05.
            if ratio <= 0.05:
                midas_params = np.append(midas_params, [1.0])
                monthly_index = np.append(monthly_index, i)
            else:
                midas_params = np.append(midas_params, [1.0, 1.0])
                daily_index = np.append(daily_index, i)
        
        self.monthly = monthly_index
        self.daily = daily_index
        self.init_params = np.append(garch_params, midas_params)
        return self.init_params
    
    def model_filter(self, params, X, y):
        """
        This function is about to create the model's equation.
        The short-/long-term component will be calculated as well.

        Parameters
        ----------
        params : numpy.array
            Numpy array that contain the required amount of parameters.
        X : DataFrame
            Pandas dataframe that contains all the regressors
        y : pandas.Series or numpy.array
            Sequance that contains the dependent variable.

        Returns
        -------
        sigma2 : numpy.array
            Numpy array that return the values from the specification.

        """
        # Array of zeros with length of the dependent variable
        self.g = np.zeros(len(y))
        resid = y - params[0]
        sigma2 = np.zeros(len(y))
        # Empty list to collect the ...
        plc = []
        
        uncond_var = params[1] / (1 - params[2] - params[3])
        
        # 'per' is an array of periods (monthly). For example [(2010, 1), ...]
        per = X.index.to_period('M')
        # 'uniq' contains the unique dates (monthly)
        uniq = np.asarray(per.unique())
        
        # Array of zeros with length of the number of unique monthly dates
        self.tau = np.zeros(len(uniq))
        
        for t in range(len(uniq)):
            if t == 0:
                plc.append(np.where((per >= uniq[t].strftime('%Y-%m')) & (per < uniq[t + 1].strftime('%Y-%m')))[0])
                # 'new_d' is a empty array if t equal to zero. I will collect daily regressors in 'new_d'
                # so in the first month I assume didn't have any knowledge about the past.
                new_d = np.array([])
            elif t != len(uniq) - 1:
                plc.append(np.where((per >= uniq[t].strftime('%Y-%m')) & (per < uniq[t + 1].strftime('%Y-%m')))[0])
                # 'dd' contain the values of daily regressors from the previous period.
                dd = X.iloc[plc[t - 1], self.daily].values
                if len(dd) < self.lag:
                    # I create 'pad' variable to build matrixes with the same size. It is a crutial step
                    # because in january we have more observations than february, so I made an assumption,
                    # that each month have a length is equal to the size of lag.
                    pad = np.zeros((self.lag - len(dd), dd.shape[1]))
                    new_d = np.vstack([dd[::-1], pad]).T
                else:
                    # If we have more observation than lag, I dropped out the last (length - lag)
                    new_d = dd[len(dd) - self.lag:][::-1].T
            else:
                plc.append(np.where(per >= uniq[t].strftime('%Y-%m'))[0])
                dd = X.iloc[plc[t - 1], self.daily].values
                if len(dd) < self.lag:
                    pad = np.zeros((self.lag - len(dd), dd.shape[1]))
                    new_d = np.vstack([dd[::-1], pad]).T
                else:
                    new_d = dd[len(dd) - self.lag:][::-1].T
            
            # First, I added monthly variables to tau and the intercept
            self.tau[t] = params[4] + np.dot(X.iloc[plc[t], self.monthly].values[0], params[5 : 5 + len(self.monthly)])
            
            # Finally, I added daily observations from t-1 period specfied with Beta function 
            for j in range(len(new_d)):
                x = new_d[j].reshape((1, self.lag))
                self.tau[t] += params[5 + len(self.monthly) + j] * Beta().x_weighted(x, [1.0, params[(5 + len(self.monthly + self.daily) + j)]])
            
            for i in plc[t]:
                if i == 0:
                    self.g[i] = uncond_var
                    sigma2[i] = self.g[i] * self.tau[t]
                else:
                    self.g[i] = uncond_var * (1 - params[2] - params[3]) + params[2] * ((resid[i-1] ** 2) / self.tau[t]) + params[3] * self.g[i - 1]
                    sigma2[i] = self.g[i] * self.tau[t]
        
        return sigma2
    
    def loglikelihood(self, params, X, y):
        sigma2 = self.model_filter(params, X, y)
        resid = y - params[0]
        return loglikelihood_normal(resid, sigma2)
    
    def predict(self, X, y):
        return self.model_filter(self.optimized_params, X, y)
    
class MGARCH(BaseModel):
    def __init__(self, lag = 12, plot = True, *args):
        self.lag = lag
        self.plot = plot
        self.args = args

    def initialize_params(self, X):
        try:
            X_len = X.shape[1]
        except:
            X_len = 1
        
        garch_params = np.array([0.1, 0.85])
        midas_params = np.linspace(1.0, 1.0, int(1.0 + X_len * 2.0))
        
        self.init_params = np.append(garch_params, midas_params)
        return self.init_params

    def model_filter(self, params, X, y):
        self.g = np.zeros(len(y))
        self.tau = np.zeros(len(X))
        sigma2 = np.zeros(len(y))
        
        try:
            X_len = X.shape[1]
        except:
            X_len = 1
        
        alpha1, beta1 = params[0], params[1]
        
        resid = y
        uncond_var = np.mean(y ** 2)

        for t in range(len(X) - 1):
            if t == 0:
                m = np.where(y.index < X.index[t + 1])[0]
            else:
                m = np.where((y.index >= X.index[t]) & (y.index < X.index[t + 1]))[0]
            
            if t - self.lag < 0:
                self.tau[t] = params[2]
                for par in range(1, X_len + 1):
                    self.tau[t] += params[2 * par + 1] * Beta().x_weighted(X.iloc[ : t, par - 1][::-1].values.reshape((1, X.iloc[ : t, par - 1].shape[0])), [1.0, params[2 * par + 2]])
            else:
                self.tau[t] = params[2] 
                for par in range(1, X_len + 1):
                    self.tau[t] += params[2 * par + 1] * Beta().x_weighted(X.iloc[t - self.lag : t, par - 1][::-1].values.reshape((1, X.iloc[t - self.lag : t, par - 1].shape[0])), [1.0, params[2 * par + 2]])
            
            for i in m:
                if i == 0:
                    self.g[i] = uncond_var
                else:
                    self.g[i] = uncond_var * (1 - alpha1 - beta1) + alpha1 * (resid[i - 1] ** 2) / self.tau[t] + beta1 * self.g[i - 1]
                sigma2[i] = self.g[i] * self.tau[t]
                    
        return sigma2
    
    def loglikelihood(self, params, X, y):
        sigma2 = self.model_filter(params, X, y)
        resid = y
        return loglikelihood_normal(resid, sigma2)
    
    def predict(self, X, y):
        return self.model_filter(self.optimized_params, X, y)
    
    def simulate(self, params = [0.0, 0.1, 0.2, 0.6, 0.4, 0.005, 5.0], lag = 12, num = 100):
        rv = np.zeros(num)
        tau = np.zeros(num)
        g = np.zeros(num * 22)
        sigma2 = np.zeros(num * 22)
        y = np.zeros(num * 22)
        
        mu, omega, alpha, beta, m, pszi, theta = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
        uncond_var = omega / (1 - alpha - beta)
        
        for t in range(num):
            if t - lag < 0:
                rv[t] = np.sum(y[(t - 1 )* 22 : t * 22] ** 2)
                tau[t] = m + pszi * Beta().x_weighted(rv[:t][::-1].reshape((1, rv[:t].shape[0])), [1.0, theta])
            else:
                rv[t] = np.sum(y[(t - 1 )* 22 : t * 22] ** 2)
                tau[t] = m + pszi * Beta().x_weighted(rv[t - lag : t][::-1].reshape((1, rv[t - lag : t].shape[0])), [1.0, theta])
            for i in range(t * 22, (t + 1) * 22):
                if t == 0:
                    if i == 0:
                        g[i] = uncond_var
                    else:
                        g[i] = uncond_var * (1 - alpha - beta) + alpha * (y[i - 1] - mu) ** 2 + beta * g[i - 1]
                    sigma2[i] = g[i]
                    y[i] = stats.norm.rvs(loc = params[0], scale = np.sqrt(sigma2[i]))
                else:
                    g[i] = uncond_var * (1 - alpha - beta) + alpha * ((y[i - 1] - mu) ** 2) / tau[t] + beta * g[i - 1]
                    sigma2[i] = g[i] * tau[t]
                    y[i] = stats.norm.rvs(loc = params[0], scale = np.sqrt(sigma2[i]))
        return rv, tau, g, sigma2, y
    
class GARCH_MIDAS_sim(BaseModel):
    def __init__(self, lag = 36, plot = True, *args):
        self.lag = lag
        self.plot = plot
        self.args = args

    def initialize_params(self, X):
        self.init_params = np.array([0.05, 0.5, 0.5, 0.5, 0.5, 1.0])
        return self.init_params
    
    def model_filter(self, params, X, y):
        self.tau = np.zeros(len(X))
        self.g = np.zeros(len(y))
        sigma2 = np.zeros(len(y))
        
        I_t = int(len(y) / len(X))
        
        mu = params[0]
        alpha1 = params[1]
        beta1 = params[2]
        m = params[3]
        theta = params[4]
        w = params[5]
        
        X_t = np.zeros((len(X), self.lag))
        
        for t in range(len(X)):
            if t < self.lag:
                X_t[t] = np.hstack((X[ : t][::-1], np.zeros(self.lag - t)))
            else:
                X_t[t] = X[t - self.lag : t][::-1]
                
        self.tau = np.exp(m + theta *  Beta().x_weighted(X_t, [1.0, w]))
        
        j = 0
        
        for i in range(len(y)):
            if i % I_t == 0:
                j += 1
                
            if i == 0:
                self.g[i] = 1
            else:
                self.g[i] = (1 - alpha1 - beta1) + alpha1 * ((y[i - 1] - mu) ** 2) / self.tau[j - 1] + beta1 *self.g[i - 1]
                    
            sigma2[i] = self.g[i] * self.tau[j - 1]
                
        return sigma2
    
    def loglikelihood(self, params, X, y):
        sigma2 = self.model_filter(params, X, y)
        resid = y - params[0]
        return loglikelihood_normal(resid, sigma2)
    
    def simulate(self,
                 params = [0.0, 0.06, 0.91, 0.1, 0.3, 4.0, 0.9, 0.09],
                 num = 480,
                 lag = 36,
                 I_t = 22):
        X = np.zeros(num)
        tau = np.zeros(num)
        g = np.zeros(num * I_t)
        sigma2 = np.zeros(num * I_t)
        r = np.zeros(num * I_t)
        X_t = np.zeros((num, lag))
        
        mu = params[0]
        alpha1 = params[1]
        beta1 = params[2]
        m = params[3]
        theta = params[4]
        w = params[5]
        fi = params[6]
        sigma_fi = params[7]
        
        j = 0
        
        for i in range(num):
            if i == 0:
                X[i] = np.random.normal(0.0, sigma_fi)
            else:
                X[i] = fi * X[i - 1] + np.random.normal(0.0, sigma_fi)
        
        for i in range(num):
            if i < lag:
                X_t[i] = np.hstack((X[ : i][::-1], np.zeros(lag - i)))
            else:
                X_t[i] = X[i - lag : i][::-1]
            
        tau = np.exp(m + theta * Beta().x_weighted(X_t, [1.0, w]))
            
        for i in range(num * I_t):
            if i % I_t == 0:
                j += 1
            
            if i == 0:
                g[i] = 1
            else:
                g[i] = 1 - alpha1 - beta1 + alpha1 * (r[i - 1]) ** 2 / tau[j - 1] + beta1 * g[i - 1]
            
            sigma2[i] = g[i] * tau[j - 1]
            r[i] = stats.norm.rvs(loc = mu, scale = np.sqrt(sigma2[i]))
            
        return X, r, tau, g, sigma2
    
class Panel_GARCH(BaseModel):
    def __init__(self, plot = True, dist = 'normal', *args):
        self.plot = plot
        self.dist = dist
        self.args = args
    
    def initialize_params(self, X):
        if self.dist == 'normal':
            self.init_params = np.array([0.4, 0.4])
        elif self.dist == 'student-t':
            self.init_params = np.array([0.4, 0.4, 4.0])
        else:
            raise ValueError("ValueError exception thrown")
        return self.init_params
    
    def model_filter(self, params, X):
        sigma2 = np.zeros_like(X)
        
        alpha, beta = params[0], params[1]
        
        uncond_var = np.nanmean(X ** 2, axis = 0)
        nans = X.isna().sum().values
        X = X.values
        
        for i in range(sigma2.shape[0]):
            for j in range(sigma2.shape[1]):
                if nans[j] == i:
                    sigma2[i][j] = uncond_var[j]
                elif nans[j] < i:
                    sigma2[i][j] = uncond_var[j] * (1 - alpha - beta) + alpha * (X[i - 1][j] ** 2) + beta * sigma2[i - 1][j]
                else:
                    pass
        return sigma2
    
    def loglikelihood(self, params, X):
        sigma2 = self.model_filter(params, X)
        if self.dist == 'normal':
            lls = loglikelihood_normal(X, sigma2).sum()
        elif self.dist == 'student-t':
            lls = loglikelihood_student_t(X, sigma2, params[2]).sum()
        return lls
    
    def simulate(self, params = [0.06, 0.91], num = 100, length = 1000):
        sigma2 = np.zeros((length, num))
        r = np.zeros((length, num))
        
        alpha, beta = params[0], params[1]
        
        for t in range(length):
            if t == 0:
                sigma2[t] = 1.0
            else:
                sigma2[t] = 1 - alpha - beta + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
            r[t] = np.random.normal(0.0, np.sqrt(sigma2[t]))
        return sigma2, r
    
    def forecast(self, X, H):
        X_new = X
        X_new.loc[X.shape[0]] = 0
        
        sigma2 = self.model_filter(self.optimized_params, X_new)
        sigma2 = sigma2 * np.sqrt(H)
        return sigma2[-1]

class Panel_GARCH_CSA(BaseModel):
    """
    Panel GARCH with cross sectional adjustment
    
    $r_{it} = \sigma_{it} c_t \epsilon_{it}$
    $\mu_i = \frac{1}{N} \sum_{i = 1}^N r_{it}^2$
    $c_t = (1 - \phi) + \phi \sqrt{ \frac{1}{N} \sum_{i = 1}^N (\frac{r_{it-1}}{\sigma_{it-1} c_{t-1}} - \frac{1}{N} \sum_{i = 1}^N \frac{r_{it-1}}{\sigma_{it-1} c_{t-1}} )^2}$
    $\sigma_{it}^2 = \mu_i (1 - \alpha - \beta) + \alpha \epsilon_{it-1}^2 + \beta \sigma_{it-1}^2$
    """
    def __init__(self, plot = True, dist = 'normal', *args):
        self.plot = plot
        self.dist = dist
        self.args = args
        
    def initialize_params(self, X):
        if self.dist == 'normal':
            self.init_params = np.array([0.1, 0.5, 0.5])
        elif self.dist == 'student-t':
            self.init_params = np.array([0.1, 0.5, 0.5, 4.0])
        return self.init_params
    
    def model_filter(self, params, y):
        c = np.zeros(y.shape[0])
        sigma2 = np.zeros(y.shape)
        
        T, N = y.shape
        
        mu = np.nanmean(y ** 2, axis = 0)
        
        y = y.values
        
        phi, alpha, beta = params[0], params[1], params[2]
        
        for t in range(T):
            if t == 0:
                c[t] = 1.0
                for i in range(N):
                    if np.isnan(y[t][i]) == True:
                        sigma2[t][i] = np.nan
                    else:
                        sigma2[t][i] = mu[i]
            else:
                c[t] = (1 - phi) + phi * np.nanstd(y[t - 1] / (np.sqrt(sigma2[t - 1]) * c[t - 1]))
                for i in range(N):
                    if np.isnan(y[t][i]) == True:
                        if np.isnan(y[t - 1][i]) == True:
                            sigma2[t][i] = np.nan
                        else:
                            sigma2[t][i] = mu[i]
                    else:
                        if np.isnan(sigma2[t - 1][i]) == False:
                            sigma2[t][i] = mu[i] * (1 - alpha - beta) + alpha * (y[t - 1][i] / (np.sqrt(sigma2[t - 1][i]) * c[t - 1])) ** 2 + beta * sigma2[t - 1][i]
                        else:
                            sigma2[t][i] = mu[i]
                
        return sigma2, c
    
    def loglikelihood(self, params, y):
        sigma2, _ = self.model_filter(params, y)
        lls = 0
        sigma2 = sigma2.T
        for i in range(y.shape[1]):
            idx = np.where(np.isnan(sigma2[i]) == False)[0]
            sig = sigma2[i][idx]
            xx = y.iloc[idx, i].values
            if len(sig) == 0.0:
                lls += 0.0
            else:
                if self.dist == 'normal':
                    lls += loglikelihood_normal(xx, sig)
                elif self.dist == 'student-t':
                    lls += loglikelihood_student_t(xx, sig, params[3])
        return lls
    
    def simulate(self, params = [0.1, 0.2, 0.6], num = 100, length = 500):
        c = np.zeros(length)
        sigma2 = np.zeros((length, num))
        ret = np.zeros((length, num))
        
        phi, alpha, beta = params[0], params[1], params[2]
        
        for t in range(length):
            if t == 0:
                c[t] = 1.0
                sigma2[t] = 1.0
            else:
                c[t] = (1 - phi) + phi * np.std(ret[t - 1] / (sigma2[t - 1] * c[t - 1]))
                mu = np.mean(ret[ : t] ** 2, axis = 0)
                sigma2[t] = mu * (1 - alpha - beta) + alpha * (ret[t - 1] / (sigma2[t - 1] * c[t - 1])) ** 2 + beta * sigma2[t - 1]
            
            ret[t] = stats.norm.rvs(loc = 0.0, scale = np.sqrt(sigma2[t]))
                
        return ret, sigma2, c
    
    def forecast(self, y):
        row_nul = pd.DataFrame([[0]*y.shape[1]], columns = y.columns)
        y = y.append(row_nul)
        
        sigma2, _ = self.model_filter(self.optimized_params, y)
        forecast = sigma2[-1]
        forecast[np.where(forecast == 0)[0]] = np.nan
        return forecast

class Panel_MIDAS(BaseModel):
    def __init__(self, lag = 12, plot = True, exp = True, *args):
        self.lag = lag
        self.plot = plot
        self.exp = exp
        self.args = args
        
    def initialize_params(self, X):
        self.init_params = np.linspace(1, 1, int(1.0 + X.shape[1] * 2.0))
        return self.init_params
    
    def model_filter(self, params, X):
        X = create_matrix(X, self.lag)
        model = params[0]
        for i in range(1, len(X) + 1):
            model += params[2 * i - 1] * Beta().x_weighted(X['X{num}'.format(num = i)], [1.0, params[2 * i]])
        if self.exp == True:
            return np.exp(model)
        else:
            return model
            
    def loglikelihood(self, params, X, y):
        try:
            y_len, y_col = y.shape
        except:
            y_len, y_col = y.shape[0], 1
        
        y_nan = y.isna().sum().values
        self.tau_t = np.zeros(y_len)
        tau = self.model_filter(params, X)
        T = X.shape[0]
        j = 0
        
        for i in range(T - 1):
            if i == 0:
                index = y[y.index < X.index[i + 1]].index
            else:
                index = y[(y.index >= X.index[i]) & (y.index < X.index[i + 1])].index
                
            mat = np.linspace(tau[i], tau[i], index.shape[0])
            self.tau_t[j : j + index.shape[0]] = mat
            j += index.shape[0]
            
        lls = 0
        for i in range(y_col):
            if y_nan[i] >= y_len:
                lls += 0
            else:
                lls += loglikelihood_normal(y.iloc[y_nan[i]:, i].values, self.tau_t[y_nan[i]:])
        return lls
    
    def simulate(self, params = [0.1, 0.3, 4.0], num = 500, K = 12, panel = 100):
        X = np.zeros(num)
        tau = np.zeros(num)
        r = np.zeros((num * 22, panel))
        j = 0
        month = []
        m_dates = []
        y_dates = []
        
        for t in range(num):
            if t == 0:
                X[t] = np.random.normal()
            else:
                X[t] = 0.9 * X[t - 1] + np.random.normal()
                
        for t in range(1, num + 1):
            if t < K + 1:
                tau[t - 1] = np.exp(params[0] + params[1] * Beta().x_weighted(X[:t][::-1].reshape((1, X[:t].shape[0])), [1.0, params[2]]))
            else:
                tau[t - 1] = np.exp(params[0] + params[1] * Beta().x_weighted(X[t - K : t][::-1].reshape((1, K)), [1.0, params[2]]))
                
            r[(t - 1) * 22 : t * 22] = np.random.normal(scale = np.sqrt(tau[t - 1]), size = (22, panel))
            
        
        for i in range(num):
            month.append(i % 12)

        for i in month:
            if i == 0:
                j += 1
                m_dates.append(datetime(2010 + j, 1, 1))
            else:
                m_dates.append(datetime(2010 + j, 1 + i, 1))

        for i in m_dates[:-1]:
            for j in range(22):
                y_dates.append(i + timedelta(j))   
        
        y = pd.DataFrame(data = r[:-22], index = y_dates)
        X = pd.DataFrame(data = X, index = m_dates)
        
        return X, y, tau
    
    def create_sims(self, number_of_sims = 500, length = 100, K = 12, params = [0.1, 0.3, 4.0], panel = 200):
        lls, b0, b1, th, runtime = np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims)
        
        for i in range(number_of_sims):
            np.random.seed(i)
            X, y, _ = self.simulate(params = params, num = length, K = K, panel = panel)
            start = time.time()
            self.fit(['pos', 'pos', 'pos'], X, y)
            runtime[i] = time.time() - start
            lls[i] = self.opt.fun
            b0[i], b1[i], th[i] = self.optimized_params[0], self.optimized_params[1], self.optimized_params[2]
            print("{}st iteration's runTime: {} sec.\n".format(i + 1, round(runtime[i], 4)))
            
        return pd.DataFrame(data = {'LogLike': lls, 
                                    'Beta0': b0, 
                                    'Beta1': b1, 
                                    'Theta':th})
    
class Panel_GARCH_MIDAS(object):
    def __init__(self, lag = 12, plot = True, exp = True, *args):
        self.lag = lag
        self.exp = exp
        self.plot = plot
        self.args = args
        
    def fit(self, restriction_midas, restriction_garch, X, y):
        self.midas = Panel_MIDAS(lag = self.lag, plot = self.plot, exp = self.exp)
        if self.plot == True:
            print('Estimated parameters for the MIDAS equation:\n')
        else:
            pass
        self.midas.fit(restriction_midas, X, y)

        y_hat = self.calculate_y_hat(y, self.midas.tau_t)
        
        self.garch = Panel_GARCH(plot = self.plot)
        if self.plot == True:
            print('\nEstimated parameters for the GARCH equation:\n')
        else:
            pass
        self.garch.fit(restriction_garch, y_hat)

    def calculate_y_hat(self, y, tau):
        y_hat = np.zeros_like(y)

        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y_hat[i][j] = y.iloc[i, j] / np.sqrt(tau[i])
                
        y_hat = pd.DataFrame(data = y_hat, index = y.index, columns = y.columns)

        return y_hat
    
    def simulate(self, midas_params = [0.1, 0.3, 4.0], garch_params = [0.06, 0.8], num = 500, K = 12, panel = 100):
        beta0, beta1, theta = midas_params[0], midas_params[1], midas_params[2]
        alpha, beta = garch_params[0], garch_params[1]
        X = np.zeros(num)
        tau = np.zeros(num)
        r = np.zeros((num * 22, panel))
        g = np.zeros((num * 22, panel))
        j = 0
        month = []
        m_dates = []
        y_dates = []
        
        for t in range(num):
            if t == 0:
                X[t] = np.random.normal()
            else:
                X[t] = 0.9 * X[t - 1] + np.random.normal()
                
        for t in range(1, num + 1):
            if t < K + 1:
                tau[t - 1] = np.exp(beta0 + beta1 * Beta().x_weighted(X[:t][::-1].reshape((1, X[:t].shape[0])), [1.0, theta]))
            else:
                tau[t - 1] = np.exp(beta0 + beta1 * Beta().x_weighted(X[t - K : t][::-1].reshape((1, K)), [1.0, theta]))
            
            for i in range((t - 1) * 22, t * 22):
                if i == 0:
                    g[i] = np.ones(panel)
                else:
                    g[i] = (1 - alpha - beta) + alpha * (r[i - 1] ** 2) / tau[t - 1] + beta * g[i - 1]
                
                r[i] = np.random.normal(scale = np.sqrt(g[i] * tau[t - 1]), size = panel)
        
        for i in range(num):
            month.append(i % 12)

        for i in month:
            if i == 0:
                j += 1
                m_dates.append(datetime(2010 + j, 1, 1))
            else:
                m_dates.append(datetime(2010 + j, 1 + i, 1))

        for i in m_dates[:-1]:
            for j in range(22):
                y_dates.append(i + timedelta(j))   
        
        y = pd.DataFrame(data = r[:-22], index = y_dates)
        X = pd.DataFrame(data = X, index = m_dates)
        
        return X, y, tau, g
    
    def create_sims(self, number_of_sims = 500, length = 100, K = 12, midas_params = [0.1, 0.3, 4.0], garch_params = [0.06, 0.8]):
        b0, b1, th, al, bt, runtime = np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims), np.zeros(number_of_sims)
        
        for i in range(number_of_sims):
            np.random.seed(i)
            X, y, _, _ = self.simulate(midas_params = midas_params, garch_params = garch_params, num = length, K = K, panel = 100)
            start = time.time()
            self.fit(['pos', 'pos', 'pos'], ['01', '01'], X, y)
            runtime[i] = time.time() - start
            b0[i], b1[i], th[i], al[i], bt[i] = self.midas.optimized_params[0], self.midas.optimized_params[1], self.midas.optimized_params[2], self.garch.optimized_params[0], self.garch.optimized_params[1]
            print("{}st iteration's runTime: {} sec.\n".format(i + 1, round(runtime[i], 4)))
            
        return pd.DataFrame(data = {'Beta0': b0, 
                                    'Beta1': b1, 
                                    'Theta':th,
                                    'Alpha': al,
                                    'Beta': bt})
    
            
    def forecast(self, y, H = 5, plotting = True):
        from pandas.tseries.offsets import BDay
        import matplotlib.pyplot as plt
        forecast = np.zeros(H)
        mu = np.mean(y ** 2)
        alpha = self.garch.optimized_params[0]
        beta = self.garch.optimized_params[1]
        y_hat = y / self.midas.tau_t
        sigma2 = self.garch.model_filter(self.garch.optimized_params, y_hat)
        
        for i in range(1, H + 1):
            forecast[i - 1] = (mu * (1 - (alpha + beta) ** (i - 1)) + sigma2[-1] * (alpha + beta) ** (i - 1)) * self.midas.tau_t[-1]
        
        forc = np.zeros(len(y) + H)
        forc[:-H] = sigma2 * self.midas.tau_t
        forc[-H:] = forecast
        
        if isinstance(y, pd.core.series.Series) or isinstance(y, pd.core.frame.DataFrame):
            index = []
            for i in range(len(y) + H):
                if i < len(y):
                    index.append(y.index[i])
                else:
                    index.append(y.index[-1] + BDay(i - len(y.index) + 1))
    
            forecasted_series = pd.Series(data = forc, index = index)
            if plotting == True:
                plt.figure(figsize = (15, 5))
                plt.plot(forecasted_series[forecasted_series.index <= pd.to_datetime(y.index[-1])], 'g')
                plt.plot(forecasted_series[forecasted_series.index > pd.to_datetime(y.index[-1])], 'r')
                plt.title("Volatility Prediction for the next {} days".format(H))
                plt.tight_layout()
                plt.show()
        else:
            forecasted_series = forc
        
        return forecasted_series

class Panel_GARCH_SLSQP(GarchBase):
    def __init__(self, plot = True, dist = 'normal', *args):
        self.plot = plot
        self.dist = dist
        self.args = args
    
    def initialize_params(self):
        if self.dist == 'normal':
            self.init_params = np.array([0.4, 0.4])
        elif self.dist == 'student-t':
            self.init_params = np.array([0.4, 0.4, 4.0])
        else:
            raise ValueError("ValueError exception thrown")
        return self.init_params
    
    def model_filter(self, params, y):
        sigma2 = np.zeros(y.shape)
        alpha, beta = params[0], params[1]
        uncond_var = np.mean(y ** 2)
        
        for i in range(y.shape[0]):
            if i == 0:
                sigma2[i] = uncond_var
            else:
                sigma2[i] = uncond_var * (1 - alpha - beta) + alpha * (y[i - 1] ** 2) + beta * sigma2[i - 1]
        
        return sigma2
    
    def loglikelihood(self, params, y):
        y_len, y_cols = y.shape
        lls = 0
        
        for i in range(y_cols):
            xx = y.iloc[np.where(y.iloc[:, i].isna() == False)[0], i].values
            if len(xx) == 0:
                lls += 0.0
            else:
                sigma2 = self.model_filter(params, xx)
                if self.dist == 'normal':
                    lls += loglikelihood_normal(xx, sigma2)
                elif self.dist == 'student-t':
                    lls += loglikelihood_student_t(xx, sigma2, params[2])
        return lls
    
    def variables(self):
        if self.dist == 'normal':
            return ['Alpha', 'Beta']
        elif self.dist == 'student-t':
            return ['Alpha', 'Beta', 'Nu']
        
    def forecast(self, y, H):
        y_len, y_cols = y.shape
        y_new = y
        y_new[y_len] = 0
        forecast = np.zeros(y_cols)
        
        for i in range(y_cols):
            xx = y_new.iloc[np.where(y_new.iloc[:, i].isna() == False)[0], i].values
            if len(xx) <= 1.0:
                forecast[i] = np.nan
            else:
                sigma2 = self.model_filter(self.optimized_params, xx)
                forecast[i] = sigma2[-1] * np.sqrt(H)
        
        return forecast
    
    def simulate(self, params = [0.06, 0.91], num = 100, length = 1000):
        sigma2 = np.zeros((length, num))
        r = np.zeros((length, num))
        
        alpha, beta = params[0], params[1]
        
        for t in range(length):
            if t == 0:
                sigma2[t] = 1.0
            else:
                sigma2[t] = 1 - alpha - beta + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
            r[t] = np.random.normal(0.0, np.sqrt(sigma2[t]))
        return sigma2, r
    
class Panel_GARCH_CSA_SLSQP(GarchBase):
    def __init__(self, plot = True, dist = 'normal', *args):
        self.plot = plot
        self.dist = dist
        self.args = args
    
    def initialize_params(self):
        if self.dist == 'normal':
            self.init_params = np.array([0.4, 0.4, 0.4])
        elif self.dist == 'student-t':
            self.init_params = np.array([0.4, 0.4, 0.4, 4.0])
        else:
            raise ValueError("ValueError exception thrown")
        return self.init_params
    
    def model_filter(self, params, y):
        c = np.zeros(y.shape[0])
        sigma2 = np.zeros(y.shape)
        
        T, N = y.shape
        
        mu = np.nanmean(y ** 2, axis = 0)
        
        y = y.values
        
        phi, alpha, beta = params[0], params[1], params[2]
        
        for t in range(T):
            if t == 0:
                c[t] = 1.0
                for i in range(N):
                    if np.isnan(y[t][i]) == True:
                        sigma2[t][i] = np.nan
                    else:
                        sigma2[t][i] = mu[i]
            else:
                c[t] = (1 - phi) + phi * np.nanstd(y[t - 1] / (np.sqrt(sigma2[t - 1]) * c[t - 1]))
                for i in range(N):
                    if np.isnan(y[t][i]) == True:
                        if np.isnan(y[t - 1][i]) == True:
                            sigma2[t][i] = np.nan
                        else:
                            sigma2[t][i] = mu[i]
                    else:
                        if np.isnan(sigma2[t - 1][i]) == False:
                            sigma2[t][i] = mu[i] * (1 - alpha - beta) + alpha * (y[t - 1][i] / (np.sqrt(sigma2[t - 1][i]) * c[t - 1])) ** 2 + beta * sigma2[t - 1][i]
                        else:
                            sigma2[t][i] = mu[i]
                
        return sigma2, c
    
    def loglikelihood(self, params, y):
        sigma2, _ = self.model_filter(params, y)
        lls = 0
        sigma2 = sigma2.T
        for i in range(y.shape[1]):
            idx = np.where(np.isnan(y.iloc[:, i]) == False)[0]
            sig = sigma2[i][idx]
            xx = y.iloc[idx, i].values
            if len(sig) == 0.0:
                lls += 0.0
            else:
                if self.dist == 'normal':
                    lls += loglikelihood_normal(xx, sig)
                elif self.dist == 'student-t':
                    lls += loglikelihood_student_t(xx, sigma2, params[3])
        return lls
    
    def variables(self):
        if self.dist == 'normal':
            return ['Phi', 'Alpha', 'Beta']
        elif self.dist == 'student-t':
            return ['Phi','Alpha', 'Beta', 'Nu']
        
    def simulate(self, params = [0.1, 0.2, 0.6], num = 100, length = 500):
        c = np.zeros(length)
        sigma2 = np.zeros((length, num))
        ret = np.zeros((length, num))
        
        phi, alpha, beta = params[0], params[1], params[2]
        
        for t in range(length):
            if t == 0:
                c[t] = 1.0
                sigma2[t] = 1.0
            else:
                c[t] = (1 - phi) + phi * np.std(ret[t - 1] / (sigma2[t - 1] * c[t - 1]))
                mu = np.mean(ret[ : t] ** 2, axis = 0)
                sigma2[t] = mu * (1 - alpha - beta) + alpha * (ret[t - 1] / (sigma2[t - 1] * c[t - 1])) ** 2 + beta * sigma2[t - 1]
            
            ret[t] = stats.norm.rvs(loc = 0.0, scale = np.sqrt(sigma2[t]))
                
        return ret, sigma2, c
    
    def forecast(self, y):
        row_nul = pd.DataFrame([[0]*y.shape[1]], columns = y.columns)
        y = y.append(row_nul)
        
        sigma2, _ = self.model_filter(self.optimized_params, y)
        forecast = sigma2[-1]
        return forecast
    
class Panel_GARCH_MIDAS_SLSQP(object):
    def __init__(self, lag = 12, plot = True, exp = True, two_type = False, *args):
        self.lag = lag
        self.exp = exp
        self.plot = plot
        self.two_type = two_type
        self.args = args
        
    def fit(self, restriction_midas, restriction_garch, X, y):
        self.midas = Panel_MIDAS(lag = self.lag, plot = self.plot, exp = self.exp)
        if self.plot == True:
            print('Estimated parameters for the MIDAS equation:\n')
        else:
            pass
        self.midas.fit(restriction_midas, X, y)

        y_hat = self.calculate_y_hat(y, self.midas.tau_t)
        
        self.garch_1 = Panel_GARCH_SLSQP(plot = self.plot, dist = 'normal')
        if self.plot == True:
            print('\nEstimated parameters for the GARCH equation (Normal):\n')
        else:
            pass
        if self.two_type == True:
            self.garch_2 = Panel_GARCH_SLSQP(plot = self.plot, dist = 'student-t')
            self.garch_1.fit(restriction_garch[:-1], y_hat)
            if self.plot == True:
                print('\nEstimated parameters for the GARCH equation (Student-t):\n')
            else:
                pass
            self.garch_2.fit(restriction_garch, y_hat)
        else:
            self.garch_1.fit(restriction_garch, y_hat)
        return
            
    def calculate_y_hat(self, y, tau):
        y_hat = np.zeros_like(y)

        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y_hat[i][j] = y.iloc[i, j] / np.sqrt(tau[i])
                
        y_hat = pd.DataFrame(data = y_hat, index = y.index, columns = y.columns)

        return y_hat
    
    def simulate(self, midas_params = [0.1, 0.3, 4.0], garch_params = [0.06, 0.8], num = 500, K = 12, panel = 100):
        beta0, beta1, theta = midas_params[0], midas_params[1], midas_params[2]
        alpha, beta = garch_params[0], garch_params[1]
        X = np.zeros(num)
        tau = np.zeros(num)
        r = np.zeros((num * 22, panel))
        g = np.zeros((num * 22, panel))
        j = 0
        month = []
        m_dates = []
        y_dates = []
        
        for t in range(num):
            if t == 0:
                X[t] = np.random.normal()
            else:
                X[t] = 0.9 * X[t - 1] + np.random.normal()
                
        for t in range(1, num + 1):
            if t < K + 1:
                tau[t - 1] = np.exp(beta0 + beta1 * Beta().x_weighted(X[:t][::-1].reshape((1, X[:t].shape[0])), [1.0, theta]))
            else:
                tau[t - 1] = np.exp(beta0 + beta1 * Beta().x_weighted(X[t - K : t][::-1].reshape((1, K)), [1.0, theta]))
            
            for i in range((t - 1) * 22, t * 22):
                if i == 0:
                    g[i] = np.ones(panel)
                else:
                    g[i] = (1 - alpha - beta) + alpha * (r[i - 1] ** 2) / tau[t - 1] + beta * g[i - 1]
                
                r[i] = np.random.normal(scale = np.sqrt(g[i] * tau[t - 1]), size = panel)
        
        for i in range(num):
            month.append(i % 12)

        for i in month:
            if i == 0:
                j += 1
                m_dates.append(datetime(2010 + j, 1, 1))
            else:
                m_dates.append(datetime(2010 + j, 1 + i, 1))

        for i in m_dates[:-1]:
            for j in range(22):
                y_dates.append(i + timedelta(j))   
        
        y = pd.DataFrame(data = r[:-22], index = y_dates)
        X = pd.DataFrame(data = X, index = m_dates)
        
        return X, y, tau, g
    
    def forecast(self, y):
        y_hat = self.calculate_y_hat(y * 100, self.midas.tau_t)
        
        if self.two_type == False:
            forecast = self.garch_1.forecast(y_hat, 1) * self.midas.tau_t[-1]
            return forecast
        else:
            forecast_norm = self.garch_1.forecast(y_hat, 1) * self.midas.tau_t[-1]
            forecast_stud = self.garch_2.forecast(y_hat, 1) * self.midas.tau_t[-1]
            return forecast_norm, forecast_stud

class EWMA(BaseModel):
    def __init__(self, plot = True, lam = 0.94, *args):
        self.plot = plot
        self.lam = 0.94
        self.args = args
    
    def initialize_params(self, y):
        self.init_params = np.array([self.lam])
        return self.init_params
    
    def model_filter(self, params, y):
        T = y.shape[0]
        sigma2 = np.zeros(T)
        lamb = params
        
        for t in range(T):
            if t == 0:
                sigma2[t] = np.nanmean(y ** 2)
            else:
                sigma2[t] = lamb * sigma2[t - 1] + (1 - lamb) * y[t - 1] ** 2
        return sigma2
    
    def loglikelihood(self, params, y):
        sigma2 = self.model_filter(params, y)
        return loglikelihood_normal(y, sigma2)
    
    def simulate(self, lamb, T):
        sigma2 = np.zeros(T)
        ret = np.zeros(T)
        
        for t in range(T):
            if t == 0:
                sigma2[t] = 1.0
            else:
                sigma2[t] = lamb * sigma2[t - 1] + (1 - lamb) * ret[t - 1] ** 2
            ret[t] = np.random.normal(scale = np.sqrt(sigma2[t]))
            
        return ret, sigma2
    
class Panel_EWMA(BaseModel):
    def __init__(self, plot = True, lam = 0.94, *args):
        self.plot = plot
        self.lam = 0.94
        self.args = args
    
    def initialize_params(self, y):
        self.init_params = np.array([self.lam])
        return self.init_params
    
    def model_filter(self, params, y):
        T = y.shape[0]
        sigma2 = np.zeros(T)
        lamb = params
        
        for t in range(T):
            if t == 0:
                sigma2[t] = 1.0
            else:
                sigma2[t] = lamb * sigma2[t - 1] + (1 - lamb) * y[t - 1] ** 2
        return sigma2
    
    def loglikelihood(self, params, y):
        lls = 0
        
        for i in range(y.shape[1]):
            idx = np.where(np.isnan(y.iloc[:, i]) == False)[0]
            sig = self.model_filter(params, y.iloc[idx, i].values)
            if len(sig) == 0:
                lls += 0
            else:
                lls += loglikelihood_normal(y.iloc[idx, i].values, sig)
        return lls
    
    def forecast(self, y):
        row_nul = pd.DataFrame([[0]*y.shape[1]], columns = y.columns)
        y = y.append(row_nul)
        forecast = np.zeros(len(y.columns))
        for i in range(y.shape[1]):
            idx = np.where(np.isnan(y.iloc[:, i]) == False)[0]
            if len(idx) == 0:
                forecast[i] = np.nan
            else:
                sig = self.model_filter(self.optimized_params, y.iloc[idx, i].values)
                forecast[i] = sig[-1]
        return forecast
    
    def simulate(self, lamb = 0.94, T = 500, num = 100):
        sigma2 = np.zeros((T, num))
        r = np.zeros((T, num))
        
        for t in range(T):
            if t == 0:
                sigma2[t] = 1.0
            else:
                sigma2[t] = lamb * sigma2[t - 1] + (1 - lamb) * r[t - 1] ** 2
            r[t] = np.random.normal(0.0, np.sqrt(sigma2[t]), size = num)
        return r, sigma2