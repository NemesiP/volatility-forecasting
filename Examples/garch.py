import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

class garch:
    def __init__(self, init_params, x, model = 'GARCH'):
        self.model = model
        self.params = init_params
        self.x = x
        
    def sgarch(self, init_params, x):
        alpha0 = self.params[0]
        alpha1 = self.params[1]
        beta1 = self.params[2]
        eps = self.x
        
        iT = len(eps)
        sigma_2 = np.zeros(iT)
        for i in range(iT):
            if i == 0:
                sigma_2[i] = alpha0/(1 - alpha1 - beta1)
            else:
                sigma_2[i] = alpha0 + alpha1 * eps[i - 1]**2 + beta1 * sigma_2[i - 1]
                
        return sigma_2
    
    def gjr_garch(self, init_params, x):
        alpha0 = self.params[0]
        alpha1 = self.params[1]
        beta1 = self.params[2]
        omega = self.params[3]
        eps = self.x
        
        iT = len(eps)
        sigma_2 = np.zeros(iT)
        for i in range(iT):
            if i == 0:
                sigma_2[i] = alpha0/(1 - alpha1 - beta1)
            else:
                sigma_2[i] = alpha0 + alpha1 * eps[i - 1]**2 + beta1 * sigma_2[i - 1] + omega * eps[i - 1]**2 * (eps[i - 1] < 0)
                
        return sigma_2
    
    def loglike(self, init_params, x):
        if self.model == 'GARCH':                
            sigma_2 = self.sgarch(init_params, x)
            
        elif self.model == 'GJR-GARCH': 
            sigma_2 = self.gjr_garch(init_params, x)
            
        logL = -np.sum(-np.log(sigma_2) - self.x**2/sigma_2)
        
        return logL
    
    def fit(self, init_params, x):
        if self.model == 'GARCH':
            res = minimize(self.loglike, init_params, args = (x, ), 
                           bounds = ((0.0001, None), (0.0001, None), (0.0001, None)), 
                           options = {'disp': True})
        elif self.model == 'GJR-GARCH':
            res = minimize(self.loglike, init_params, args = (x,), 
                           bounds = ((0.0001, None), (0.0001, None), (0.0001, None), (0.0001, None)), 
                           options = {'disp': True})
        
        return res
    
if __name__ == '__main__':
    print('A fit() függvénybe még van valami hiba!, így a modell nem tud lefutni')
    df = pd.read_csv('AMD.csv')
    df.Date = pd.to_datetime(df.Date)
    df['Chg'] = np.log(df.Close).diff().fillna(0)
    returns = np.array(df.Chg[1:].values)
    init_params = (0.01, 0.05, 0.9)
    
    model = garch(init_params, returns)
    res = model.fit(init_params, returns)