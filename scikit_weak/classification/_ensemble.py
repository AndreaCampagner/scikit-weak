from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import resample
import numpy as np
from ..utils import to_probs, prob_format

class RRLClassifier(BaseEstimator, ClassifierMixin):
  '''
    A class to perform classification for weakly supervised data, based on the RRL algorithm [1].
    The y input to the fit method should be given as either:
     - a ndarray of lists
     - a n*m matrix, where n is the same size as X.shape[0] and m is the number of classes, where each row sums to 1 (prob format)
     - in sklearn format, with some entries possibly set to -1 (semi-supervised format)

    Parameters
    ----------
    :param estimator: Base estimator objects to be fitted. Should support predict and predict_proba
    :type estimator: estimator class, default=ExtraTreeClassifier

    :param n_estimators: The number of trees to be fitted
    :type n_estimators: int, default=100

    :param missing: Whether the input y will be passed in semi-supervised format or not
    :type missing: bool, default=False

    :param probs: Default probability distribution to be used. Only used if missing=True. If probs is None then use uniform distribution
    :type probs: ndarray or None, default=None

    :param resample: Whether to perform bootstrapping or not
    :type resample: bool, default=False
        
    :param random_state: Random seed
    :type random_state: int, default=0

    Attributes
    ----------
    :ivar y_probs: If y is in prob format, then target is a copy of y. Otherwise it is y in prob format
    :vartype probs: ndarray

    :ivar classifiers: The collection of fitted estimators
    :vartype classifiers: list of estimators

    :ivar ys: The collection of sampled target labels. Each ndarray in ys has the same shape as y
    :vartype ys: list of ndarrays
    
    :ivar Xs: The collection of bootstrapped datasets. Each ndarray in Xs has the same shape as X. If resample=False, then Xs is empty.
    :vartype Xs: list of ndarrays
    
    :ivar n_classes: The number of unique classes in y
    :vartype n_classes: int

    :ivar classes_: The unique classes in y
    :vartype classes_: ndarray
    '''
  def __init__(self, estimator=ExtraTreeClassifier, n_estimators=100, missing=False, probs=None, resample=False, random_state=0):
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.estimator = estimator
    self.probs = probs
    self.resample = resample
    self.missing = missing

  def fit(self, X, y):
    """
        Fit the RRLClassifier model
    """
    np.random.seed(self.random_state)
    self.X = X
    self.y = y

    if self.missing:
      self.n_classes_ = len(np.unique(self.y[self.y != -1]))
      self.__pre_process()
      self.classes_ = np.array(range(self.y_probs.shape[1]))
    else:
      self.y_probs = self.y
      if not prob_format(self.y_probs):
        self.y_probs = to_probs(y)
      self.n_classes_ = self.y_probs.shape[1]
      self.classes_ = np.array(range(self.y_probs.shape[1]))
    
    self.classifiers = [ self.estimator() for i in range(self.n_estimators)]
    
    self.ys = []
    self.Xs = []
    for i in range(self.n_estimators):
      temp_ys = self.__sample_labels()
      if self.resample:
        temp_Xs, temp_ys = resample(self.X, temp_ys, random_state = self.random_state+i)
        self.Xs.append(temp_Xs)
        self.ys.append(temp_ys)
        self.classifiers[i].fit(self.Xs[i], self.ys[i])
      else:
        self.ys.append(temp_ys)
        self.classifiers[i].fit(self.X, self.ys[i])
    return self

  def __sample_labels(self):
    y_res = np.zeros([len(self.y)])
    for i in range(self.y_probs.shape[0]):
      
      y_res[i] = np.random.choice(self.classes_, p=self.y_probs[i])
    return y_res

  def __pre_process(self):
    self.y_probs = np.zeros((len(self.y),self.n_classes_))
    for t in range(len(self.y)):
      if self.y[t] == -1:
        if self.probs is None:
          self.y_probs[t,:] = 1.0/self.n_classes_
        else:
          self.y_probs[t,:] = np.array( tuple( self.probs.values() ) )
      else:
        self.y_probs[t,self.y[t]] = 1.0
  
  def predict(self, X):
    """
        Returns predictions for the given X
    """
    output = np.zeros((X.shape[0], self.n_classes_))
    sorter = np.argsort(self.classes_)
    for i in range(self.n_estimators):
      indices = sorter[np.searchsorted(self.classes_, self.classifiers[i].classes_, sorter=sorter)]
      output[:, indices] += self.classifiers[i].predict_proba(X)
    return self.classes_[np.argmax(output, axis= 1)]

  def predict_proba(self, X):
    """
        Returns probability distributions for the given X
    """
    output = np.zeros((X.shape[0], self.n_classes_))
    sorter = np.argsort(self.classes_)
    for i in range(self.n_estimators):
      indices = sorter[np.searchsorted(self.classes_, self.classifiers[i].classes_, sorter=sorter)]
      output[:, indices] += self.classifiers[i].predict_proba(X)
    return normalize(output, axis=1, norm='l1')