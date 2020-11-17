import numpy as np
import pandas as pd
from scipy.optimize import least_squares

class Weights():
    def __init__(self, theta):
        self.theta1 = theta[0]
        self.theta2 = theta[1]
        
    def weights(self, nlags):
        eps = np.spacing(1)
        u = np.linspace(eps, 1.0 - eps, nlags)
        beta_vals = u ** (self.theta1 - 1) * (1 - u) ** (self.theta2 - 1)
        beta_vals = beta_vals / sum(beta_vals)
        return beta_vals
    
    def x_weighted(self, x, params):
        self.theta1, self.theta2 = params
        w = self.weights(x.shape[1])
        xw = np.matrix(x*w).sum(axis = 1)
        df = pd.DataFrame(xw)
        return df.iloc[:, 0].values
    
class Lags():
    def buildLag(s, lag=2, dropna=True):
        if type(s) is pd.DataFrame:
            new_dict = {}
            for col_name in s:
                new_dict[col_name] = s[col_name]
                for l in range(1, lag + 1):
                    new_dict['%s_lag%d' %(col_name, l)]=s[col_name].shift(l)
            res = pd.DataFrame(new_dict, index = s.index)
            
        elif type(s) is pd.Series:
            the_range = range(lag+1)
            res = pd.concat([s.shift(i) for i in the_range], axis = 1)
            red.columns = ['lag_%d' %i for i in the_range]
        else:
            print('Only works for DataFrame or Series')
            return None
        if dropna:
            return res.dropna()
        else:
            return res
        

## Defining the feature matrix
df1 = pd.read_csv('data/Stocks/AMD.csv')
ret_1 = np.log(df1.close).diff().fillna(0)
ret_1 = pd.DataFrame(np.matrix(ret_1).T)
X = Lags.buildLag(s = ret_1, lag = 4, dropna = False).fillna(0)

## Defining the output vector
df2 = pd.read_csv('data/Stocks/NVDA.csv')
output = np.log(df2.close).diff().fillna(0)

## Defining thwe initiale parameters
params = [1, 2, 1, 5]

## Defining the Model
def y(params, x):
    return params[0] + params[1]*Weights([1., 5.]).x_weighted(x, params[2:])

def fun(params):
    return y(params, X) - output

## Optimization
model = least_squares(fun, params)


