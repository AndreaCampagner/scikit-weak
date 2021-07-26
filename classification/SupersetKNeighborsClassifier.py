import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin

class SupersetKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = KDTree(X)
        self.classes_ = range(y.shape[1])
        return self
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            _, indices = self.tree.query(X[i], self.k)
            classes = np.zeros(self.y.shape[1])
            for j in indices:
                classes += self.y[j]
            y_pred[i] = np.argmax(classes)
        return y_pred
    
    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0], self.y.shape[1]))
        for i in range(X.shape[0]):
            _, indices = self.tree.query(X[i], self.k)
            for j in indices:
                y_pred[i, :] += self.y[j]
            y_pred[i, :] /= np.sum(y_pred[i,:])
        return y_pred