import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin

class WeaklySupervisedKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification for weakly supervised data, based on k-nearest neighbors.
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel

    Parameters
    ----------
    :param k: The number of neighbors
    :type k: int, default=3

    :param metric: The metric for neighbors queries
    :type metric: str or callable, default 'minkowski'

    Attributes
    ----------
    :ivar __n_classes: The number of unique classes in y
    :vartype __n_classes: int

    :ivar __classes: The unique classes in y
    :vartype __classes: list of int
    '''
    def __init__(self, k=3, metric='minkowski'):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        """
        Fit the WeaklySupervisedKNeighborsClassifier model
        """
        self.__X = X
        self.__y = np.zeros((len(y), y[0].n_classes))
        for i in range(len(y)):
            self.__y[i] = y[i].classes

        self.__n_classes = y[0].n_classes
        self.__classes = range(self.__n_classes)
        
        self.__tree = NearestNeighbors(metric=self.metric, n_neighbors=self.k)
        self.__tree.fit(self.__X)
        return self

    
    def predict(self, X):
        """
        Returns predictions for the given X
        """
        y_pred = np.zeros(X.shape[0])
        _, indices = self.__tree.kneighbors(X)
        for i in range(len(indices)):
            y_pred[i] = np.argmax(np.add.reduce(self.__y[indices[i]]))
        return y_pred
    
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        y_pred = np.zeros((X.shape[0], self.__n_classes))
        _, indices = self.__tree.kneighbors(X)
        for i in range(len(indices)):
            y_pred[i, :] = np.add.reduce(self.__y[indices[i]])
            y_pred[i, :] /= np.sum(y_pred[i,:])
        return y_pred







class WeaklySupervisedRadiusClassifier(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification for weakly supervised data, based on radius neighbors.
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel

    Parameters
    ----------
    :param radius: The size of the radius
    :type radius: float, default=1.0

    :param metric: The metric for neighbors queries
    :type metric: str or callable, default 'minkowski'

    Attributes
    ----------
    :ivar y: A copy of the input y
    :vartype y: ndarray

    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int
    '''
    def __init__(self, radius=1.0, metric='minkowski'):
        self.radius = radius
        self.metric = metric
        
    def fit(self, X, y):
        """
        Fit the WeaklySupervisedRadiusClassifier model
        """
        self.__X = X
        self.__y = np.zeros((len(y), y[0].n_classes))
        for i in range(len(y)):
            self.__y[i] = y[i].classes
        
        self.__tree = NearestNeighbors(metric=self.metric, radius=self.radius)
        self.__tree.fit(self.__X)
        self.__n_classes = y[0].n_classes
        return self

    
    def predict(self, X):
        """
        Returns predictions for the given X
        """
        y_pred = np.zeros(X.shape[0])
        _, indices = self.__tree.radius_neighbors(X)
        for i in range(len(indices)):
            y_pred[i] = np.argmax(np.add.reduce(self.__y[indices[i]]))
        return y_pred
    
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        y_pred = np.zeros((X.shape[0], self.__n_classes))
        _, indices = self.__tree.radius_neighbors(X)
        for i in range(len(indices)):
            y_pred[i, :] = np.add.reduce(self.__y[indices[i]])
            y_pred[i, :] /= np.sum(y_pred[i,:])
        return y_pred