from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
import numpy as np

class ImpreciseKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification for imprecise data, based on k-nearest neighbors.
    The X input to the fit method should be given as an iterable of ContinuousWeakLabel that is compatible
    with the given metric.
    
    Parameters
    ----------
    :param k: The number of neighbors
    :type k: int, default=3

    :param metric: The metric for neighbors queries
    :type metric: callable, default None

    Attributes
    ----------
    :ivar y: A copy of the input y
    :vartype y: ndarray

    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int
    '''
    def __init__(self, k=3, metric=None):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        """
        Fit the WeaklySupervisedKNeighborsClassifier model
        """
        self.__X = X
        self.__y = y
        self.__n_classes = len(np.unique(y))
        return self

    
    def predict(self, X):
        """
        Returns predictions for the given X
        """
        y_pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            distances = np.zeros(self.__X.shape[0])
            for j in range(self.__X.shape[0]):
              distances[j] = self.metric(X[i,:], self.__X[j,:])
            indices = np.argsort(distances)[:self.k]

            y_pred[i] = mode(self.__y[indices]).mode
        return y_pred
    
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        y_pred = np.zeros((X.shape[0], self.__n_classes))
        for i in range(X.shape[0]):
            distances = np.zeros(self.__X.shape[0])
            for j in range(self.__X.shape[0]):
              distances[j] = self.metric(X[i,:], self.__X[j,:])
            indices = np.argsort(distances)[:self.k]

            for y in self.__y[indices]:
              y_pred[i,y] += 1/self.k
        return y_pred