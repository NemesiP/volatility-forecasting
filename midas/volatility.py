# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:50:49 2021

@author: peter
"""
import numpy as np
from base import BaseModel
from stats import loglikelihood_normal, loglikelihood_student_t
from weights import Beta
from helper_functions import create_matrix
import scipy.stats as stats

class MIDAS(BaseModel):
    def __init__(self, lag = 22, *args):
        self.lag = lag
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
#            model += params[2 * i - 1] * WeightMethod().x_weighted_beta(x['X{num}'.format(num = i)], [1.0, params[2 * i]])
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
    
    """
    def simulate(self, params = [4.0, 0.1, 1.0], num = 500):
        m, pszi, theta = params[0], params[1], params[2]
        tau = np.zeros(num)
        rv = np.zeros(num)
        y = np.zeros(num * 22)
        
        for t in range(num):
            if t - self.lag < 0:
                if t == 0:
                    rv[t] = 0.0
                else:
                    rv[t] = np.sum(y[(t - 1 )* 22 : t * 22] ** 2)
                tau[t] = m + pszi * Beta().x_weighted(rv[:t][::-1].reshape((1, rv[:t].shape[0])), [1.0, theta])
            else:
                rv[t] = np.sum(y[(t - 1 )* 22 : t * 22] ** 2)
                tau[t] = m + pszi * Beta().x_weighted(rv[t - self.lag : t][::-1].reshape((1, rv[t - self.lag : t].shape[0])), [1.0, theta])
            for i in range(t * 22, (t + 1) * 22):
                if t == 0:
                    y[i] = stats.norm.rvs(loc = 0, scale = 1)
                else:
                    y[i] = stats.norm.rvs(loc = 0, scale = np.sqrt(tau[t]) / np.sqrt(22))
        
        return tau, rv, y
    """
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
    
class GARCH(BaseModel):
    def __init__(self, *args):
        self.args = args
        
    def initialize_params(self, y):
        self.init_params = np.asarray([0.05, 0.02, 0.95])
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
    def __init__(self, *args):
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
    def __init__(self, lag = 22, *args):
        self.lag = lag
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
    def __init__(self, lag = 12, *args):
        self.lag = lag
        self.args = args

    def initialize_params(self, X):
        garch_params = np.array([0.05, 0.1, 0.1, 0.85])
        midas_params = np.array([0.5, 0.1, 1.0])
        """
        try:
            X_len = X.shape[1]
        except:
            X_len = 1
        
        for i in range(X_len):
            midas_params = np.append(midas_params, [1.0, 1.0])
        """
        self.init_params = np.append(garch_params, midas_params)
        return self.init_params

    def model_filter(self, params, X, y):
        self.g = np.zeros(len(y))
        self.tau = np.zeros(len(X))
        sigma2 = np.zeros(len(y))
        
        mu, alpha0, alpha1, beta1 = params[0], params[1], params[2], params[3]
        
        resid = y - mu
        uncond_var = alpha0 / (1 - alpha1 - beta1)

        for t in range(len(X)):
            if t == 0:
                m = np.where(y.index < X.index[t])[0]
            else:
                m = np.where((y.index < X.index[t]) & (y.index >= X.index[t - 1]))[0]
            
            if t - self.lag < 0:
                self.tau[t] = params[4] + params[5] * Beta().x_weighted(X[:t].values.T, [1.0, params[6]])
            else:
                self.tau[t] = params[4] + params[5] * Beta().x_weighted(X[t - self.lag : t].values.T, [1.0, params[6]])
            
            for i in m:
                if t == 0:
                    if i == 0:
                        self.g[i] = uncond_var
                    else:
                        self.g[i] = uncond_var * (1 - alpha1 - beta1) + alpha1 * resid[i - 1] ** 2 + beta1 * self.g[i - 1]
                    sigma2[i] = self.g[i]
                else:
                    self.g[i] = uncond_var * (1 - alpha1 - beta1) + alpha1 * (resid[i - 1] ** 2) / self.tau[t] + beta1 * self.g[i - 1]
                    sigma2[i] = self.g[i] * self.tau[t]
                    
        return sigma2
    
    def loglikelihood(self, params, X, y):
        sigma2 = self.model_filter(params, X, y)
        resid = y - params[0]
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