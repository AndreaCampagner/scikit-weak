from scipy.linalg import eigh
import numpy as np
from skweak.classification import SupersetKNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class DELIN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3, d=2, iters = 10):
        self.k = k
        self.d = d
        self.iters = iters
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.clf = SupersetKNearestNeighbors(k=self.k)
        self.classes_ = range(y.shape[1])
        
        self.y_prime = self.y
        for it in range(self.iters):
            self.mu = np.mean(self.X, axis = 0)
            self.Mu = np.zeros((t.shape[1], self.X.shape[1]))

            for j in range(t.shape[1]):
                self.Mu[j, :] = np.mean(np.multiply(self.X, np.transpose([self.y[:,j]]) ), axis=0)

            X_hat = self.X - self.mu
            S_t = X_hat.T.dot(X_hat)
            C = np.zeros((self.y.shape[1], self.y.shape[1]))
            for i in range(self.y.shape[1]):
                C[i,i] = np.sum(self.y[:,i])
            S_b = X_hat.T.dot(self.y_prime).dot(np.linalg.inv(C)).dot(self.y_prime.T).dot(X_hat)
            S_w = S_t - S_b

            Mat = np.linalg.inv(S_w).dot(S_b)
            w, self.vr= eigh(Mat)
            
            X_prime = self.X.dot(self.vr[:,-self.d:])
            self.y_prime = self.clf.fit(X_prime, self.y_prime).predict_proba(X_prime)
            self.y_prime = self.y_prime*self.y
            for i in range(self.y_prime.shape[0]):
                check = True
                for j in range(self.y_prime.shape[1]):
                    if self.y_prime[i,j] > 0:
                        check = False
                        break
                if check:
                    self.y_prime[i,:] = self.y[i,:]
            self.y_prime = self.y_prime/self.y_prime.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        X_prime = X.dot(self.vr[:,-self.d:])
        return self.clf.predict(X_prime)
    
    def predict_proba(self, X):
        X_prime = X.dot(self.vr[:,-self.d:])
        return self.clf.predict_proba(X_prime)
