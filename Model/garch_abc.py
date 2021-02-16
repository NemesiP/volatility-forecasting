# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:10:52 2021

@author: peter
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.special import gammaln

class BaseModel(object, metaclass = ABCMeta):
    def __init__(self, *args):
        self.args = args
        
    @abstractmethod
    def model(self, params, y):
        pass
    
    @abstractmethod
    def loglikelihood(self, params, y):
        pass
    
    @abstractmethod
    def fit(self):
        pass
    
class GARCH(BaseModel):
    def __init__(self, name = 'GARCH'):
        self.name = name
    
    def constraint(self, params):
        return (1 - params[1] - params[2])
    
    def model(self, params, y):
        sigma2 = np.zeros(len(y))
        
        for i in range(len(y)):
            if i == 0:
                sigma2[i] = params[0] / (1 - params[1] - params[2])
            else:
                sigma2[i] = params[0] + params[1] * y[i - 1] ** 2 + params[2] * sigma2[i - 1]
        return sigma2
    
    def loglikelihood(self, params, y):
        sigma2 = self.model(params, y)
        if self.name == 'GARCH':
            lls = 0.5 * (np.log(2*np.pi) + np.log(sigma2) + (y - params[3])** 2 / sigma2)
        elif self.name == 'T-GARCH':
            nu = params[4]
            lls = gammaln((nu + 1) /2) - gammaln(nu / 2) - np.log(np.pi * (nu - 2)) / 2
            lls -= 0.5 * (np.log(sigma2))
            lls -= ((nu + 1) / 2) * (np.log(1 + ((y - params[3]) ** 2) / (sigma2 * (nu - 2))))
        return sum(-lls) / len(y)
    
    def fit(self, y):
        if self.name == 'GARCH':
            init_params = [0.25, 0.15, 0.8, 0.05]
            opt = minimize(self.loglikelihood, 
                           init_params, 
                           args = (np.asarray(y),), 
                           options = {'disp': False},
                           constraints = {'type': 'ineq', 'fun': self.constraint},
                           bounds = ((1e-8, 1 - 1e-8), (1e-8, 1 - 1e-8), (1e-8, 1 - 1e-8), (-10.0, 10.0)),
                           tol = 1e-8,
                           method = 'SLSQP')
        elif self.name == 'T-GARCH':
            init_params = [0.25, 0.15, 0.8, 0.05, 3.75]
            opt = minimize(self.loglikelihood, 
                           init_params, 
                           args = (np.asarray(y),), 
                           options = {'disp': False},
                           constraints = {'type': 'ineq', 'fun': self.constraint},
                           bounds = ((1e-8, 1 - 1e-8), (1e-8, 1 - 1e-8), (1e-8, 1 - 1e-8), (-10.0, 10.0), (2.05, 500)),
                           tol = 1e-10,
                           method = 'SLSQP')
        return opt
    
    def simulate(self, params = [0.2, 0.2, 0.6, 0.0], num = 500, dist = 'Normal'):
        y = np.zeros(num)
        state = np.zeros(num)
        for i in range(num):
            if i == 0:
                state[i] = params[0] / (1 - params[1] - params[2])
            else:
                state[i] = params[0] + params[1] * y[i - 1] * y[i - 1] + params[2] * state[i - 1]
            if dist == 'Normal':
                y[i] = stats.norm.rvs(loc = params[3], scale = np.sqrt(state[i]))
            elif dist == 'Student-t':
                y[i] = stats.t.rvs(params[4], loc = params[3], scale = np.sqrt(state[i]))
            
        return y, state
    
if __name__ == '__main__':
    np.random.seed(14)
    returns, sigma2 = GARCH().simulate(num = 1000, params = [0.1, 0.2, 0.6, 0.0, 2.5], dist = 'Student-t')

    model = GARCH(name = 'T-GARCH').fit(returns)