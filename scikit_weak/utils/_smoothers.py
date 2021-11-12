from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from ..data_representation import *

class DiscreteEstimatorSmoother(TransformerMixin, BaseEstimator):

    def __init__(self, estimator, type='set', epsilon=1.0):
        self.estimator=estimator
        self.type=type
        self.epsilon=epsilon

    def fit(self, X, y):
        self.__base_type = None
        if self.type == 'set':
            self.__base_type = DiscreteSetLabel
        elif self.type == 'fuzzy':
            self.__base_type = DiscreteFuzzyLabel
        else:
            raise ValueError("%s is not an allowed value for 'type' parameter" % self.type)

        self.estimator.fit(X,y)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        y_fuzzy = self.estimator.predict_proba(X)
        y_fuzzy = y_fuzzy/np.max(y_fuzzy, axis=1)[:, np.newaxis]
        y_soft = np.empty(y_fuzzy.shape[0], dtype=self.__base_type)

        for i in range(y_fuzzy.shape[0]):
            if self.type == 'set':
                y_fuzzy[i] = (y_fuzzy[i] > self.epsilon)
            y_soft[i] = self.__base_type(y_fuzzy[i], 3)
        
        return y_soft

class DiscreteRandomSmoother(TransformerMixin, BaseEstimator):

    def __init__(self, p_err = 0.1, p_incl = 1.0, type='set', samples=100, epsilon=0.0):
        self.p_err = p_err
        self.p_incl = p_incl
        self.type = type
        self.samples = samples
        self.epsilon = epsilon

    def fit(self, X, y):
        self.__base_type = None
        if self.type == 'set':
            self.__base_type = DiscreteSetLabel
        elif self.type == 'fuzzy':
            self.__base_type = DiscreteFuzzyLabel
        else:
            raise ValueError("%s is not an allowed value for 'type' parameter" % self.type)
        
        self.n_classes_ = len(np.unique(y))
        return self

    def fit_transform(self, X, y):
        self.fit(X,y)

        y_soft = np.empty(y.shape[0], dtype=self.__base_type)
        for i in range(X.shape[0]):
            y_temp = np.zeros(self.n_classes_)
            y_temp[y[i]] = 1.0
            for k in range(self.n_classes_):
                for s in range(self.samples):
                    if k == y[i]:
                        y_temp[k] += np.random.choice([0,1], size=None, replace=True, p=[1-self.p_incl, self.p_incl])
                    else:
                        y_temp[k] += np.random.choice([0,1], size=None, replace=True, p=[1-self.p_err, self.p_err])
            
            y_temp /= self.samples
            y_temp = y_temp/np.max(y_temp)
            if self.type == 'set':
                y_temp = (y_temp > self.epsilon)*1.0
                
            y_soft[i] = self.__base_type(y_temp, self.n_classes_)

        return y_soft

