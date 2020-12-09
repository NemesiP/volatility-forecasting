import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/peter/Desktop/volatility-forecasting/data/Stocks/AMD.csv')
df = df.iloc[-1500:, :]
df['Chg'] = np.log(df.close).diff().fillna(0)
returns = df.Chg[1:].values
df['Date'] = pd.to_datetime(df.iloc[:, 0])


def garch_filter(alpha0, alpha1, beta1, omega, eps):
    iT = len(eps)
    sigma_2 = np.zeros(iT)
    
    for i in range(iT):
        if i == 0:
            sigma_2[i] = alpha0/(1 - alpha1 - beta1)
        else:
            sigma_2[i] = alpha0 + alpha1*eps[i - 1]**2 + beta1*sigma_2[i - 1] + omega * eps[i - 1]**2 * (eps[i - 1] < 0)
            
    return sigma_2

def garch_loglike(vP, eps):
    alpha0 = vP[0]
    alpha1 = vP[1]
    beta1 = vP[2]
    omega = vP[3]
    
    sigma_2 = garch_filter(alpha0, alpha1, beta1, omega, eps)
    
    logL = -np.sum(-np.log(sigma_2) - eps**2/sigma_2)
    
    return logL

cons = ({'type': 'ineq', 'func': lambda x: np.array(x)})
vP0 = (0.1, 0.05, 0.92, 0.2)

res = minimize(garch_loglike, vP0, args = (returns), 
               bounds = ((0.0001, None), (0.0001, None), (0.0001, None), (0.0001, None)), 
               options = {'disp': True})

alpha0_est = res.x[0]
alpha1_est = res.x[1]
beta1_est = res.x[2]
omega_est = res.x[3]
sigma2 = garch_filter(alpha0_est, alpha1_est, beta1_est, omega_est, returns)

plt.plot(df.Date[1:], sigma2, label = 'GJR-GARCH')
plt.legend(loc = 'best')
plt.show()

def garch_filter2(alpha0, alpha1, beta1, eps):
    iT = len(eps)
    sigma_2 = np.zeros(iT)
    
    for i in range(iT):
        if i == 0:
            sigma_2[i] = alpha0/(1 - alpha1 - beta1)
        else:
            sigma_2[i] = alpha0 + alpha1*eps[i - 1]**2 + beta1*sigma_2[i - 1]
            
    return sigma_2

def garch_loglike2(vP, eps):
    alpha0 = vP[0]
    alpha1 = vP[1]
    beta1 = vP[2]
    
    sigma_2 = garch_filter2(alpha0, alpha1, beta1, eps)
    
    logL = -np.sum(-np.log(sigma_2) - eps**2/sigma_2)
    
    return logL

cons = ({'type': 'ineq', 'func': lambda x: np.array(x)})
vP0 = (0.1, 0.05, 0.92)

res2 = minimize(garch_loglike2, vP0, args = (returns), 
               bounds = ((0.0001, None), (0.0001, None), (0.0001, None)), 
               options = {'disp': True})

alpha0_est2 = res2.x[0]
alpha1_est2 = res2.x[1]
beta1_est2 = res2.x[2]
sigma22 = garch_filter2(alpha0_est2, alpha1_est2, beta1_est2, returns)

plt.plot(df.Date[1:], sigma22, label = 'GARCH')
plt.legend(loc = 'best')
plt.show()

plt.plot(df.Date[1:], sigma22, label = 'GARCH')
plt.plot(df.Date[1:], sigma2, label = 'GJR-GARCH')
plt.legend(loc = 'best')
plt.show()

plt.scatter(sigma2, sigma22)
plt.show()