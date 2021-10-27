from scipy.linalg import eigh
from ..classification import WeaklySupervisedKNeighborsClassifier
from ..data_representation import DiscreteFuzzyLabel
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class DELIN(BaseEstimator, TransformerMixin, ClassifierMixin):
    '''
    A class to perform classification and dimensionality reduction for weakly supervised data,
    based on the DELIN algorithm [1].
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel

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
            
    :ivar clf: A WeaklyKNeighborsClassifier classifier to be used during fitting of the algorithm
    :vartype clf: WeaklyKNeighborsClassifier object

    :ivar vr: A square ndarray with the same dim as X.shape[1]. Used to perform dimensionality reduction
    :vartype vr: ndarray
            
    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int

    :ivar classes_: The unique classes in y
    :vartype classes: ndarray
    '''
    def __init__(self, k=3, d=2, n_iters = 10):
        self.k = k
        self.d = d
        self.n_iters = n_iters
        
    def fit(self, X, y):
        """
        Fit the DELIN model
        """
        self.__X = X
        self.__y = np.array(y)
        
        
        self.__clf = WeaklySupervisedKNeighborsClassifier(k=self.k)
        self.__n_classes = self.__y[0].n_classes
        
        self.__y_probs = np.empty((self.__X.shape[0], self.__n_classes))

        for i in range(len(self.__y)):
            self.__y_probs[i] = self.__y[i].to_probs()
        self.__y_prime = self.__y_probs.copy()

        for it in range(self.n_iters):
            self.__mu = np.mean(self.__X, axis = 0)
            self.__Mu = np.zeros((self.__n_classes, self.__X.shape[1]))

            for j in range(self.__n_classes):
                self.__Mu[j, :] = np.mean(np.multiply(self.__X, np.transpose([self.__y_probs[:,j]]) ), axis=0)

            X_hat = self.__X - self.__mu
            S_t = X_hat.T.dot(X_hat)
            C = np.zeros((self.__n_classes, self.__n_classes))
            for i in range(self.__n_classes):
                C[i,i] = np.sum(self.__y_probs[:,i])
            S_b = X_hat.T.dot(self.__y_prime).dot(np.linalg.inv(C)).dot(self.__y_prime.T).dot(X_hat)
            S_w = S_t - S_b

            Mat = np.linalg.inv(S_w).dot(S_b)
            _, self.__vr= eigh(Mat)
            
            X_prime = self.__X.dot(self.__vr[:,-self.d:])
            y_temp = np.empty(self.__y_prime.shape[0], DiscreteFuzzyLabel)

            for i in range(self.__y_prime.shape[0]):
                y_temp[i] = DiscreteFuzzyLabel(self.__y_prime[i], self.__n_classes)

            self.__y_prime = self.__clf.fit(X_prime, y_temp).predict_proba(X_prime)
            

            self.__y_prime = self.__y_prime*self.__y_probs
            
            for i in range(self.__y_prime.shape[0]):
                check = True
                for j in range(self.__y_prime.shape[1]):
                    if self.__y_prime[i,j] > 0:
                        check = False
                        break
                if check:
                    self.__y_prime[i] = self.__y_probs[i]

            self.__y_prime = self.__y_prime/self.__y_prime.sum(axis=1, keepdims=True)

        return self

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    
    def predict(self, X):
        """
        Returns predictions for the given X
        """
        X_prime = X.dot(self.__vr[:,-self.d:])
        return self.__clf.predict(X_prime)
    
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        X_prime = X.dot(self.__vr[:,-self.d:])
        return self.__clf.predict_proba(X_prime)

    def transform(self, X, y=None):
        X_prime = X.dot(self.__vr[:,-self.d:])
        return X_prime

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
