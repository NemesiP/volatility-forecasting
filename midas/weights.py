import numpy as np
from scipy.stats import beta
from abc import ABCMeta, abstractmethod

class WeightMethod(object, metaclass = ABCMeta):
    def __init__(self, *args):
        self.args = args
        
    @abstractmethod
    def weights(self):
        pass

    def x_weighted(self, x, params):
        try:
            w = self.weights(params, x.shape[1])
        except:
            w = self.weights(params, 1)
        
        return np.matmul(x, w)
        
class beta_(WeightMethod):
    def weights(self, params, nlags):
        eps = 1e-6
        x = np.linspace(eps, 1 - eps, nlags)
        beta_vals = beta.pdf(x, params[0], params[1]) / np.sum(beta.pdf(x, params[0], params[1]))
        return beta_vals
    
class exp_almon_(WeightMethod):
    def weights(self, params, nlags):
        ith = np.arange(1, 1 + nlags)
        almon_vals = np.exp(params[0] * ith + params[1] * ith ** 2) / np.sum(np.exp(params[0] * ith + params[1] * ith ** 2))
        return almon_vals

class ewma_(WeightMethod):
    def weights(self, params, nlags):
        lmd = params
        ith = np.arange(1, 1 + nlags)
        ewma = ((1 - lmd) * lmd ** (ith - 1)) / (1 - lmd ** nlags)
        return ewma

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    w = beta_().weights(0.00, 10)
    
    plt.plot(w)
    plt.show()
    
    x = np.random.normal(size = (10, 10))
    
    xw = beta_().x_weighted(x, 0.0)
    
    plt.plot(xw)
    plt.show()
