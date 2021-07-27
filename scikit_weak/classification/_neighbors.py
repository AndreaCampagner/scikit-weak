import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin

from ..utils import to_probs, prob_format

class SupersetKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification for weakly supervised data, based on k-nearest neighbors.
    The y input to the fit method should be given as either:
     - a ndarray of lists
     - a n*m matrix, where n is the same size as X.shape[0] and m is the number of classes, where each row sums to 1 (prob format)

    Parameters
    ----------
    :param k: The number of neighbors
    :type k: int, default=3

    Attributes
    ----------
    :ivar y: If y is in prob format, then target is a copy of y. Otherwise it is y in prob format
    :vartype y: ndarray

    :ivar tree: A tree object for nearest neighbors queries speed-up
    :vartype tree: KDTree object

    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int

    :ivar classes_: The unique classes in y
    :vartype classes: ndarray
    '''
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        """
        Fit the SupersetKNeighborsClassifier model
        """
        self.X = X
        self.y = y
        if not prob_format(self.y):
            self.y = to_probs(y)
        self.tree = KDTree(X)
        self.classes_ = np.unique(np.add.reduce(y))
        self.n_classes = len(self.classes_)
        return self

    
    def predict(self, X):
        """
        Returns predictions for the given X
        """
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            _, indices = self.tree.query(X[i], self.k)
            classes = np.zeros(self.y.shape[1])
            classes += np.add.reduce(self.y[indices])
            y_pred[i] = np.argmax(classes)
        return y_pred
    
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        y_pred = np.zeros((X.shape[0], self.y.shape[1]))
        for i in range(X.shape[0]):
            _, indices = self.tree.query(X[i], self.k)
            for j in indices:
                y_pred[i, :] += self.y[j]
            y_pred[i, :] /= np.sum(y_pred[i,:])
        return y_pred