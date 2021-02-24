# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:50:49 2021

@author: peter
"""
import numpy as np
from base import BaseModel
from stats import loglikelihood_normal, loglikelihood_student_t
from weights import WeightMethod
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
            model += params[2 * i - 1] * WeightMethod().x_weighted_beta(x['X{num}'.format(num = i)], [1.0, params[2 * i]])
        
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
        self.init_params = np.asarray([0.05, 0.1, 0.02, 0.95])
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
        self.init_params = np.asarray([0.05, 0.1, 0.02, 0.95, 3.75])
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
    #To-Do
    pass