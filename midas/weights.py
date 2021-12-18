import numpy as np
from scipy.stats import beta as b
from abc import ABCMeta, abstractmethod

class WeightMethod(object, metaclass = ABCMeta):
    def __init__(self, *args):
        self.args = args
        
    @abstractmethod
    def weights(self):
        pass

    def x_weighted(self, x, params, arg = False):
        if arg == False:
            try:
                w = self.weights(params, x.shape[1])
            except:
                w = self.weights(params, 1)
        else:
            try:
                w = self.weights(params, x.shape[0])
            except:
                w = self.weights(params, 1)
        return np.matmul(x, w)
        
class Beta(WeightMethod):
    def weights(self, params, nlags):
        eps = 1e-6
        x = np.linspace(eps, 1 - eps, nlags)
        beta_vals = b.pdf(x, params[0], params[1]) / np.sum(b.pdf(x, params[0], params[1]))
        return beta_vals
    
class ExpAlmon(WeightMethod):
    def weights(self, params, nlags):
        ith = np.arange(1, 1 + nlags)
        almon_vals = np.exp(params[0] * ith + params[1] * ith ** 2) / np.sum(np.exp(params[0] * ith + params[1] * ith ** 2))
        return almon_vals

class Exp(WeightMethod):
    def weights(self, params, nlags):
        ith = np.arange(1, 1 + nlags)
        ew = params ** ith / np.sum(params ** ith)
        return ew

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    w = Beta().weights([1., 500.], 50)
    
    plt.plot(w)
    plt.show()
    
    x = np.random.normal(size = (12, 12))
    
    xw = Beta().x_weighted(x, [1., 500.])
    
    plt.plot(xw)
    plt.show()
