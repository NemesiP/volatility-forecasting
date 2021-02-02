# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:06:52 2021

@author: peter
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

class GARCH11(object):
    def __init__(self, returns, initial_parameters, restrictions):
        self.returns = returns
        self.init_params = initial_parameters
        self.restrictions = restrictions
        self.coeff = self.transform_back(self.fit().x, self.restrictions)
        self.sigma2 = self.garch_filter(self.coeff) 
        self.std_err = self.standard_error()
        self.output = self.fit()
        
    def transform(self, params, restrictions):
        params_trans = np.zeros(params.shape)
        for i in range(len(params)):
            if restrictions[i] == 'pos':
                params_trans[i] = np.log(params[i])
            elif restrictions[i] == '01':
                params_trans[i] = np.log(params[i]) - np.log(1 - params[i])
            else:
                params_trans[i] = params[i]
        return params_trans
    
    def transform_back(self, params_trans, restrictions):
        params = np.zeros(params_trans.shape)
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
    
    def garch_filter(self, params):
        un_var = params[1]/(1 - params[2] - params[3])
        sigma2 = np.zeros_like(self.returns)
        
        for i in range(len(self.returns)):
            if i == 0:
                sigma2[i] = un_var
            else:
                sigma2[i] = np.dot(un_var, (1 - params[2] - params[3])) + np.dot(params[2], (self.returns[i-1] - params[0]) ** 2) + np.dot(params[3], sigma2[i - 1])
        return sigma2
    
    def normal_loglikelihood(self, params):
        sigma2 = self.garch_filter(params)
        lls = 0.5 * (np.log(2*np.pi) + np.log(sigma2) + (self.returns - params[0])**2 / sigma2)
        return sum(lls)/len(self.returns)
    
    def normal_loglikelihood_trans(self, params_trans, restrictions):
        params = self.transform_back(params_trans, restrictions)
        return self.normal_loglikelihood(params)
    
    def fit(self):
        opt = minimize(self.normal_loglikelihood_trans,
                       self.transform(self.init_params, self.restrictions),
                       args = (self.restrictions),
                       method = 'l-bfgs-b')
        return opt

    def standard_error(self):
        grad = self.gradient(self.coeff, self.restrictions)
        variance = self.fit().hess_inv.todense() / len(self.returns)
        return np.multiply(np.sqrt(np.diag(variance)), grad)
    
    def create_table(self):
        high = self.coeff + 1.96 * self.std_err
        low = self.coeff - 1.96 * self.std_err
        llf = -self.normal_loglikelihood(self.coeff)*len(self.returns)
        AIC = round(2 * len(self.coeff) - 2 * llf, 4)
        BIC = round(len(self.coeff) * np.log(len(self.returns)) - 2 * llf , 4)
        
        table = pd.DataFrame(data = {'Parameters': self.coeff,
                                     'Std_Error': self.std_err,
                                     't_score': self.coeff / self.std_err,
                                     'p_value': stats.norm.sf(self.coeff / self.std_err),
                                     '95% CI lower': low,
                                     '95% CI upper': high})
        return print('Loglikelihood:', llf, '\nAIC: ', AIC, '\nBIC: ', BIC, '\n'), display(table)

class GARCH11_sim(object):
    def __init__(self, params, num, display = False):
        self.params = params
        self.num = num
        self.display = display
        self.sigma2, self.returns = self.garch_sim()
        
    def garch_sim(self):
        y = np.zeros(self.num)
        state = np.zeros(self.num)
        for i in range(self.num):
            if i == 0:
                state[i] = np.divide(np.take(self.params, 0, -1), (1 - np.take(self.params, 1, -1) - np.take(self.params, 2, -1)))
            else:
                state[i] = np.take(self.params, 0, -1) + np.take(self.params, 1, -1) * y[i-1] ** 2.0 + np.take(self.params, 2, -1) * state[i - 1]
            if display == True:
                print('State at {} is {}'.format(i, state[i]))
            y[i] = stats.norm.rvs(scale = np.sqrt(state[i]))
        return state, y

if __name__ == '__main__':
    np.random.seed(14)

    optimal_params = np.array([0.2, 0.2, 0.6])
    simulate = GARCH11_sim(optimal_params, 1000)

    init_params = np.array([0.05, 0.1, 0.05, 0.92])
    restrictions = np.array(['', '01', '01', '01'])
        
    model = GARCH11(simulate.returns, init_params, restrictions)
    model.create_table()
    