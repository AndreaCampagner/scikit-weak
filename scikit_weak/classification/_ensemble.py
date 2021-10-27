from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import resample
import numpy as np
from ..utils import to_probs, prob_format

class RRLClassifier(BaseEstimator, ClassifierMixin):
  '''
    A class to perform classification for weakly supervised data, based on the RRL algorithm [1].
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel

    Parameters
    ----------
    :param estimator: Base estimator objects to be fitted. Should support predict and predict_proba
    :type estimator: estimator class, default=ExtraTreeClassifier

    :param n_estimators: The number of trees to be fitted
    :type n_estimators: int, default=100

    :param resample: Whether to perform bootstrapping or not
    :type resample: bool, default=False
        
    :param random_state: Random seed
    :type random_state: int, default=None

    Attributes
    ----------

    :ivar classifiers: The collection of fitted estimators
    :vartype classifiers: list of estimators

    :ivar ys: The collection of sampled target labels. Each ndarray in ys has the same shape as y
    :vartype ys: list of ndarrays
    
    :ivar Xs: The collection of bootstrapped datasets. Each ndarray in Xs has the same shape as X. If resample=False, then Xs is empty.
    :vartype Xs: list of ndarrays
    
    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int

    :ivar classes: The unique classes in y
    :vartype n_classes: list of int
    '''
  def __init__(self, estimator=ExtraTreeClassifier(), n_estimators=100, resample=False, random_state=None):
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.estimator = estimator
    self.resample = resample

  def fit(self, X, y):
    """
        Fit the RRLClassifier model
    """
    state = np.random.get_state()
    if not (self.random_state is None):
      np.random.seed(self.random_state)

    self.__X = X
    self.__y = np.array(y)

    self.__n_classes = self.__y[0].n_classes
    self.__classes = range(self.__n_classes)
    
    self.__classifiers = [ clone(self.estimator) for i in range(self.n_estimators)]
    
    self.__ys = []
    self.__Xs = []
    for i in range(self.n_estimators):
      temp_ys = self.__sample_labels()
      seed = np.random.randint(np.iinfo('int32').max)
      self.__classifiers[i].set_params(**{'random_state': seed})

      if self.resample:
        seed = np.random.randint(np.iinfo('int32').max)
        temp_Xs, temp_ys = resample(self.__X, temp_ys, random_state = seed)
        self.__Xs.append(temp_Xs)
        self.__ys.append(temp_ys)
        self.__classifiers[i].fit(self.__Xs[i], self.__ys[i])
      else:
        self.__ys.append(temp_ys)
        self.__classifiers[i].fit(self.__X, self.__ys[i])

    if not (self.random_state is None):
      np.random.set_state(state)
    return self

  def __sample_labels(self):
    y_res = np.empty(len(self.__y))
    for i in range(self.__X.shape[0]):
      y_res[i] = self.__y[i].sample_value()
    return y_res
  
  def predict(self, X):
    """
        Returns predictions for the given X
    """
    output = np.zeros((X.shape[0], self.__n_classes))
    sorter = np.argsort(self.__classes)

    for i in range(self.n_estimators):
      indices = sorter[np.searchsorted(self.__classes, self.__classifiers[i].classes_, sorter=sorter)]
      output[:, indices] += self.__classifiers[i].predict_proba(X)
    return np.argmax(output, axis= 1)

  def predict_proba(self, X):
    """
        Returns probability distributions for the given X
    """
    output = np.zeros((X.shape[0], self.__n_classes))
    sorter = np.argsort(self.__classes)
    for i in range(self.n_estimators):
      indices = sorter[np.searchsorted(self.__classes, self.__classifiers[i].classes_, sorter=sorter)]
      output[:, indices] += self.__classifiers[i].predict_proba(X)
    return normalize(output, axis=1, norm='l1')

  def fit_predict(self, X, y):
    self.fit(X,y)
    return self.predict(X)