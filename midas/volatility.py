# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:50:49 2021

@author: peter
"""
import numpy as np
from base import BaseModel
from stats import loglikelihood_normal, loglikelihood_student_t
from weights import beta_
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
            model += params[2 * i - 1] * beta_().x_weighted(x['X{num}'.format(num = i)], [1.0, params[2 * i]])
        
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
        daily_index = np.array([])
        monthly_index = np.array([])
        garch_params = np.array([0.05, 0.05, 0.02, 0.95])
        midas_params = np.array([1.0])
        for i in range(X.shape[1]):
            ratio = X.iloc[:, i].unique().shape[0] / X.shape[0]
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
        g = np.zeros(len(y))
        resid = y - params[0]
        sigma2 = np.zeros(len(y))
        
        per = X.index.to_period('M')
        uniq = np.asarray(per.unique())
        
        self.tau = np.zeros(len(uniq))
        
        uncond_var = params[1] / (1 - params[2] - params[3])
        
        plc = []
        
        for t in range(len(uniq)):
            if t == 0:
                plc.append(np.where((per >= uniq[t].strftime('%Y-%m')) & (per < uniq[t + 1].strftime('%Y-%m')))[0])
                new_d = []
            elif t != len(uniq) - 1:
                plc.append(np.where((per >= uniq[t].strftime('%Y-%m')) & (per < uniq[t + 1].strftime('%Y-%m')))[0])
                dd = X.iloc[plc[t - 1], self.daily].values
                if len(dd) < self.lag:
                    pad = np.zeros((self.lag - len(dd), dd.shape[1]))
                    new_d = np.vstack([dd[::-1], pad]).T
                else:
                    new_d = dd[len(dd) - self.lag:][::-1].T
            else:
                plc.append(np.where(per >= uniq[t].strftime('%Y-%m'))[0])
                dd = X.iloc[plc[t - 1], self.daily].values
                if len(dd) < self.lag:
                    pad = np.zeros((self.lag - len(dd), dd.shape[1]))
                    new_d = np.vstack([dd[::-1], pad]).T
                else:
                    new_d = dd[len(dd) - self.lag:][::-1].T
            
            
            self.tau[t] = params[4] + np.dot(X.iloc[plc[t], self.monthly].values[0], params[5 : 5 + len(self.monthly)])
            
            for j in range(len(new_d)):
                x = new_d[j].reshape((1, self.lag))
                self.tau[t] += params[5 + len(self.monthly) + j] * beta_().x_weighted(x, [1.0, params[(5 + len(self.monthly + self.daily) + j)]])
            
            for i in plc[t]:
                if i == 0:
                    g[i] = uncond_var
                    sigma2[i] = g[i] * self.tau[t]
                else:
                    g[i] = uncond_var * (1 - params[2] - params[3]) + params[2] * ((resid[i-1] ** 2) / self.tau[t]) + params[3] * g[i - 1]
                    sigma2[i] = g[i] * self.tau[t]
        return sigma2
    
    def loglikelihood(self, params, X, y):
        sigma2 = self.model_filter(params, X, y)
        resid = y - params[0]
        return loglikelihood_normal(resid, sigma2)
    
    def predict(self, X, y):
        return self.model_filter(self.optimized_params, X, y)