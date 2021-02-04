# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:29:22 2021

@author: peter
"""
from volatility import SimulationBase, ModelBase

class GARCH_simulation(SimulationBase):
    def simulate(self):
        y = np.zeros(self.num)
        state = np.zeros(self.num)
        for i in range(self.num):
            if i == 0:
                state[i] = np.divide(np.take(self.params, 1, -1), (1 - np.take(self.params, 2, -1) - np.take(self.params, 3, -1)))
            else:
                state[i] = np.take(self.params, 1, -1) + np.take(self.params, 2, -1) * y[i-1] ** 2.0 + np.take(self.params, 3, -1) * state[i - 1]
            if display == True:
                print('State at {} is {}'.format(i, state[i]))
            if self.dist == 'Normal':
                y[i] = stats.norm.rvs(loc = np.take(self.params, 0, -1), scale = np.sqrt(state[i]))
            elif self.dist == 'Student-t':
                y[i] = stats.t.rvs(self.df, loc = np.take(self.params, 0, -1), scale = np.sqrt(state[i]))
        
        return state, y

class GARCH11(ModelBase):
    def model(self, params):
        un_var = params[1]/(1 - params[2] - params[3])
        sigma2 = np.zeros_like(self.y)
        
        for i in range(len(self.y)):
            if i == 0:
                sigma2[i] = un_var
            else:
                sigma2[i] = np.dot(un_var, (1 - params[2] - params[3])) + np.dot(params[2], (self.y[i-1] - params[0]) ** 2) + np.dot(params[3], sigma2[i - 1])
        return sigma2
    
    def loglikelihood(self, params):
        sigma2 = self.model(params)
        lls = 0.5 * (np.log(2*np.pi) + np.log(sigma2) + (self.y - params[0])**2 / sigma2)
        return sum(lls)/len(self.y)
    
if __name__ == '__main__':
    np.random.seed(14)
    sim = GARCH_simulation(num = 1000)
    sigma2, returns = sim.simulate()

    model = GARCH11(returns, ['', '01', '01', '01'])
    model.fit()