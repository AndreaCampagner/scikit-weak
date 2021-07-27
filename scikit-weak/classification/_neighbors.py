import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.linalg import eigh
from ..utils import to_probs, prob_format

class SupersetKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification for weakly supervised data, based on k-nearest neighbors.
    The y input to the fit method should be given as either:
     - a ndarray of lists
     - a n*m matrix, where n is the same size as X.shape[0] and m is the number of classes, where each row sums to 1 (prob format)

    Parameters
    ----------
        k : int, default=3
            The number of neighbors

    Attributes
    ----------
        y: ndarray
            If y is in prob format, then target is a copy of y. Otherwise it is y in prob format
        tree: KDTree object
            A tree object for nearest neighbors queries speed-up
        n_classes: int
            The number of unique classes in y
        classes_: int
            The unique classes in y

    Methods
    -------
        fit(X, y)
            Fit the SupersetKNeighborsClassifier model
        transform(X, y)
            Transform the data (only X, y is ignored) using the support_ attribute of the fitted model
        fit_transform(X, y)
            Fit to data, then transform it
    '''
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        if not prob_format(self.y):
            self.y = to_probs(y)
        self.tree = KDTree(X)
        self.classes_ = np.unique(np.add.reduce(y))
        self.n_classes = len(self.classes_)
        return self

    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            _, indices = self.tree.query(X[i], self.k)
            classes = np.zeros(self.y.shape[1])
            classes += np.add.reduce(self.y[indices])
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

class DELIN(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification and dimensionality reduction for weakly supervised data,
    based on the DELIN algorithm [1].
    The y input to the fit method should be given as either:
     - a ndarray of lists
     - a n*m matrix, where n is the same size as X.shape[0] and m is the number of classes, where each row sums to 1 (prob format)

    Parameters
    ----------
        k : int, default=3
            The number of neighbors
        d : int, default = 2
            The number of dimensions to be kept after reduction
        iters: int, default = 10
            The number of iterations to be performed

    Attributes
    ----------
        y: ndarray
            If y is in prob format, then target is a copy of y. Otherwise it is y in prob format
        clf: SupersetKNeighborsClassifier object
            A SupersetKNeighborsClassifier classifier to be used during fitting of the algorithm
        vr: ndarray
            A square ndarray with the same dim as X.shape[1]. Used to perform dimensionality reduction
        n_classes: int
            The number of unique classes in y
        classes_: int
            The unique classes in y

    Methods
    -------
        fit(X, y)
            Fit the DELIN model
        transform(X, y)
            Transform the data (only X, y is ignored) using the support_ attribute of the fitted model
        fit_transform(X, y)
            Fit to data, then transform it

    References
    ----------
    [1] Wu, J. H., & Zhang, M. L. (2019).
        Disambiguation enabled linear discriminant analysis for partial label dimensionality reduction.
        In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19), 416-424.
        https://doi.org/10.1145/3292500.3330901
    '''
    def __init__(self, k=3, d=2, iters = 10):
        self.k = k
        self.d = d
        self.iters = iters
        
    def fit(self, X, y):
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
        X_prime = X.dot(self.vr[:,-self.d:])
        return self.clf.predict(X_prime)
    
    def predict_proba(self, X):
        X_prime = X.dot(self.vr[:,-self.d:])
        return self.clf.predict_proba(X_prime)
