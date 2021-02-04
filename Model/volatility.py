# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:46:40 2021

@author: peter
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

class SimulationBase(object, metaclass = ABCMeta):
    def __init__(self,
                 params = [0.0, 0.2, 0.2, 0.6],
                 num = 500,
                 display = False,
                 distribution = 'Normal'):
        self.params = params
        self.num = num
        self.display = display
        self.dist = distribution
        
    @abstractmethod
    def simulate(self):
        pass
  
class ModelBase(object, metaclass = ABCMeta):
    def __init__(self,
                 y = None,
                 restrictions = None,
                 params = None):
        self.y = np.asarray(y)
        self.restrictions = restrictions
        if params == None:
            self.params = [0.05, 0.1, 0.05, 0.92]
        else:
            self.params = params
    
    def transform(self, params, restrictions):
        params_trans = np.zeros_like(params)
        for i in range(len(params)):
            if restrictions[i] == 'pos':
                params_trans[i] = np.log(params[i])
            elif restrictions[i] == '01':
                params_trans[i] = np.log(params[i]) - np.log(1 - params[i])
            else:
                params_trans[i] = params[i]
        return params_trans
    
    def transform_back(self, params_trans, restrictions):
        params = np.zeros_like(params_trans)
        for i in range(len(params_trans)):
            if restrictions[i] == 'pos':
                params[i] = np.exp(params_trans[i])
            elif restrictions[i] == '01':
                params[i] = 1 / (1 + np.exp(-params_trans[i]))
            else:
                params[i] = params_trans[i]
        return params
    
    def gradient(self, param_trans, restrictions):
        g = np.zeros_like(param_trans)
        for i in range(len(g)):
            if restrictions[i] == '':
                g[i] = 1
            elif restrictions[i] == 'pos':
                g[i] = np.exp(param_trans[i])
            else:
                g[i] = np.exp(param_trans[i]) / np.power(1 + np.exp(param_trans[i]), 2)
        return g
    
    def standard_error(self, optimization):
        grad = self.gradient(self.transform_back(optimization.x, self.restrictions), self.restrictions)
        variance = optimization.hess_inv.todense() / len(self.y)
        return np.multiply(np.sqrt(np.diag(variance)), grad)
    
    @abstractmethod
    def model(self, params):
        pass
    
    @abstractmethod
    def loglikelihood(self, params):
        pass
    
    def loglikelihood_trans(self, params_trans, restrictions):
        params = self.transform_back(params_trans, restrictions)
        return self.loglikelihood(params)
    
    def fit(self):
        opt = minimize(self.loglikelihood_trans,
                       self.transform(self.params, self.restrictions),
                       args = (self.restrictions),
                       method = "l-bfgs-b")
        
        
        se = self.standard_error(opt)
        high = self.transform_back(opt.x, self.restrictions) + 1.96 * se
        low = self.transform_back(opt.x, self.restrictions)  - 1.96 * se
        llf = -self.loglikelihood(self.transform_back(opt.x, self.restrictions)) * len(self.y)
        AIC = round(2 * len(opt.x) - 2 * llf, 4)
        BIC = round(len(opt.x) * np.log(len(self.y)) - 2 * llf, 4)
        
        table = pd.DataFrame(data = {'Parameters': self.transform_back(opt.x, self.restrictions),
                                     'Standard_Error': se,
                                     '95% CI lower': low,
                                     '95% CI higher': high})
        print('Loglikelihood: ', llf, '\nAIC: ', AIC, '\nBIC: ', BIC, '\n')
        display(table)
        return opt 