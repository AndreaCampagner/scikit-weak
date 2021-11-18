from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from ..data_representation import *

class DiscreteEstimatorSmoother(TransformerMixin, BaseEstimator):
    '''
    A class to transform a supervised learning problem into a weakly supervised one, based on the output of a classifier. It
    currently supports transformation to superset and fuzzy label learning. Note that DiscreteEstimatorSmoother does not implement
    the transform method: therefore, usage should be based on calling fit_transform.

    Parameters
    ----------
    :param estimator: Base estimator objects to be fitted. Should support predict_proba
    :type estimator: estimator class

    :param type: Type of weakly supervised labels to transform into
    :type n_estimators: {'set', 'fuzzy'}, default=set

    :param epsilon: Parameter to select the minimum allowed label degree. Only used when type == 'set'. Should be between 0 and 1
    :type epsilon: float, default=1.0

    Attributes
    ----------
    :ivar n_classes_: The number of unique classes in y
    :vartype n_classes: int
    '''
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

        self.n_classes_ = len(np.unique(y))
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
            y_soft[i] = self.__base_type(y_fuzzy[i], self.n_classes_)
        
        return y_soft

class DiscreteRandomSmoother(TransformerMixin, BaseEstimator):
    '''
    A class to transform a supervised learning problem into a weakly supervised one, based on random sampling. It
    currently supports transformation to superset and fuzzy label learning. Note that DiscreteRandomSmoother does not implement
    the transform method: therefore, usage should be based on calling fit_transform.

    Parameters
    ----------
    :param p_err: The probability to include any single wrong label in a sample draw. Should be between 0 and 1.
    :type p_err: float, default=0.1

    :param p_incl: The probability to include the correct label in a sample draw. Should be between 0 and 1.
    :type p_incl: float, default=1.0

    :param prob_ranges: Array of possible membership degrees to be sampled. If not None, overrides both p_err and p_incl
    :type p_err: enumerable of float, default=None

    :param type: Type of weakly supervised labels to transform into
    :type n_estimators: {'set', 'fuzzy'}, default=set

    :param samples: Number of samples to be generated
    :type samples: int, default=100

    :param epsilon: Parameter to select the minimum allowed label degree. Only used when type == 'set'. Should be between 0 and 1
    :type epsilon: float, default=1.0

    Attributes
    ----------
    :ivar n_classes_: The number of unique classes in y
    :vartype n_classes: int
    '''
    def __init__(self, p_err = 0.1, p_incl = 1.0, prob_ranges=None, type='set', samples=100, epsilon=0.0):
        self.p_err = p_err
        self.p_incl = p_incl
        self.type = type
        self.samples = samples
        self.epsilon = epsilon
        self.prob_ranges = prob_ranges

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
        y_temp = np.zeros((y.shape[0], self.n_classes_))

        if self.prob_ranges is not None:
            y_temp = np.random.choice(self.prob_ranges, size=(y.shape[0], self.n_classes_))

        for i in range(X.shape[0]):
            if self.prob_ranges is None:
                y_temp[i, y[i]] = 1.0
                for k in range(self.n_classes_):
                    for s in range(self.samples):
                        if k == y[i]:
                            y_temp[i, k] += np.random.choice([0,1], p=[1-self.p_incl, self.p_incl])
                        else:
                            y_temp[i, k] += np.random.choice([0,1], p=[1-self.p_err, self.p_err])
                
                y_temp[i] /= self.samples
                y_temp[i] = y_temp[i]/np.max(y_temp[i])
                if self.type == 'set':
                    y_temp[i] = (y_temp[i] > self.epsilon)*1.0
                
            y_soft[i] = self.__base_type(y_temp[i], self.n_classes_)

        return y_soft

