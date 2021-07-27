from scipy.linalg import eigh
from ._neighbors import SupersetKNeighborsClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..utils import to_probs, prob_format

class DELIN(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification and dimensionality reduction for weakly supervised data,
    based on the DELIN algorithm [1].
    The y input to the fit method should be given as either:
     - a ndarray of lists
     - a n*m matrix, where n is the same size as X.shape[0] and m is the number of classes, where each row sums to 1 (prob format)

    Parameters
    ----------
    :param k: The number of neighbors
    :type k: int, default=3
    
    :param d: The number of dimensions to be kept after reduction
    :type d: int, default = 2
        
    :param iters: The number of iterations to be performed
    :type iters: int, default = 10

    Attributes
    ----------
    :ivar y: If y is in prob format, then target is a copy of y. Otherwise it is y in prob format
    :vartype y: ndarray
            
    :ivar clf: A SupersetKNeighborsClassifier classifier to be used during fitting of the algorithm
    :vartype clf: SupersetKNeighborsClassifier object

    :ivar vr: A square ndarray with the same dim as X.shape[1]. Used to perform dimensionality reduction
    :vartype vr: ndarray
            
    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int

    :ivar classes_: The unique classes in y
    :vartype classes: ndarray
    '''
    def __init__(self, k=3, d=2, iters = 10):
        self.k = k
        self.d = d
        self.iters = iters
        
    def fit(self, X, y):
        """
        Fit the DELIN model
        """
        self.X = X
        self.y = y
        if not prob_format(self.y):
            self.y = to_probs(y)
        self.clf = SupersetKNeighborsClassifier(k=self.k)
        self.classes_ = np.unique(np.add.reduce(y))
        self.n_classes = len(self.classes_)
        
        self.y_prime = self.y
        for it in range(self.iters):
            self.mu = np.mean(self.X, axis = 0)
            self.Mu = np.zeros((self.y.shape[1], self.X.shape[1]))

            for j in range(self.y.shape[1]):
                self.Mu[j, :] = np.mean(np.multiply(self.X, np.transpose([self.y[:,j]]) ), axis=0)

            X_hat = self.X - self.mu
            S_t = X_hat.T.dot(X_hat)
            C = np.zeros((self.y.shape[1], self.y.shape[1]))
            for i in range(self.y.shape[1]):
                C[i,i] = np.sum(self.y[:,i])
            S_b = X_hat.T.dot(self.y_prime).dot(np.linalg.inv(C)).dot(self.y_prime.T).dot(X_hat)
            S_w = S_t - S_b

            Mat = np.linalg.inv(S_w).dot(S_b)
            _, self.vr= eigh(Mat)
            
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
        """
        Returns predictions for the given X
        """
        X_prime = X.dot(self.vr[:,-self.d:])
        return self.clf.predict(X_prime)
    
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        X_prime = X.dot(self.vr[:,-self.d:])
        return self.clf.predict_proba(X_prime)
