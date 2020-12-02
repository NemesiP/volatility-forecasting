import numpy as np
from scipy.stats import beta

class WeightMethod:
    def __init__(self):
        self.theta = []
        
    def BetaWeights(self, params, nlags):
        self.theta.append(params[0])
        self.theta.append(params[1])
        eps = np.spacing(1)
        x = np.linspace(eps, 1.0 - eps, nlags)
        beta_vals = beta.pdf(x, self.theta[0], self.theta[1])/sum(beta.pdf(x, self.theta[0], self.theta[1]))
        return beta_vals
    
    def x_weighted_beta(self, x, params):
        w = WeightMethod().BetaWeights(params, x.shape[1])
        xw = np.matmul(x, w)
        return xw
    
    def ExpAlmonWeights(self, params, nlags):
        self.theta.append(params[0])
        self.theta.append(params[1])
        ith = np.arange(1, nlags + 1)
        almon_vals = np.exp(self.theta[0] * ith + self.theta[1] * ith ** 2)
        almon_vals = almon_vals / sum(almon_vals)
        return almon_vals
    
    def x_weighted_almon(self, x, params):
        w = WeightMethod().ExpAlmonWeights(params, x.shape[1])
        xw = np.matmul(x, w)
        return xw