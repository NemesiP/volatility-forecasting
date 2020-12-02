import numpy as np

class LinReg:
    def __init__(self, X, y):
        self.features = X
        self.output = y
        
    def coeff(self):
        ones = np.ones((len(self.output), 1))
        self.features = np.concatenate((ones, self.features), axis = 1)
        
        s1 = np.matmul(self.features.T, self.features)
        s2 = np.linalg.inv(s1)
        s3 = np.matmul(s2, self.features.T)
        self.beta = np.matmul(s3, self.output)
        
        return self.beta
    
    def performance(self):
        y_hat = np.matmul(self.features, self.beta)
        error = sum(self.output - y_hat)
            
        y_mean = sum(self.output)/len(self.output)
        r_square = sum((y_hat - y_mean)**2)/(sum((self.output - y_mean)**2))
            
        print('The R^2:', r_square)