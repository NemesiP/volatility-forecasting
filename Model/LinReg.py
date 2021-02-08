# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:07:44 2021

@author: peter
"""

from abc import ABCMeta, abstractmethod
from scipy.linalg import lstsq
import numpy as np


class LinearModel(object, metaclass = ABCMeta):
    """Base class for Linear Models"""
    
    def __init__(self, *args):
        self.args = args
        
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
        
    def get_params(self):
        """
        Return the model parameters

        Returns
        -------
        C : array, shape (n_samples, )
            Return parameters
        """
        return self.params
    
    def score(self):
        """
        Return the coefficient of determination R^2 of the prediction

        Returns
        -------
        score: float
            Return the R^2 of self.predict(X).
        """
        u = ((self.y - self.predict(self.X)) ** 2).sum()
        v = ((self.y - self.y.mean()) ** 2).sum()
        return 1 - u / v
    
class LinReg(LinearModel):
    def fit(self, X, y):
        """
        Fit linear model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape(n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, )
            Target vales.

        Returns
        -------
        self
            returns an instance of self.

        """
        self.X, self.y = X, y
        self.params, self.residuals, self.rank, self.singular = lstsq(self.X, self.y) 
        self.coef, self.intercept = self.params[0], self.params[1:]
        return self
        
    def predict(self, X):
        """
        Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape(n_samples, n_features)
            Samples.

        Returns
        -------
        predict : array, shape (n_samples,)
            Return predicted values.
            
        """
        predict = np.dot(X, self.params)
        return predict
    
if __name__ == '__main__':
    num = 100
    X = np.random.normal(0, 1, size = num).reshape((num, 1))
    X = np.concatenate((np.ones_like(X), X), axis = 1)
    y = np.dot(X, np.array([1, 1.75])) + 3
    reg = LinReg().fit(X, y)
    print(reg.score())
    print(reg.coef)
    print(reg.intercept)
    print(reg.predict(np.array([[3, 5]])))
    
    
    