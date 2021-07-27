from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
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
        estimator: object
            Base estimator objects to be fitted. Should support predict and predict_proba
        n_estimators : int, default=100
            The number of trees to be fitted
        missing : bool, default=False
            Whether the input y will be passed in semi-supervised format or not
        probs : ndarray or None, default=None
            Default probability distribution to be used. Only used if missing=True. If probs is None
            then use uniform distribution
        resample : bool, default=False
            Whether to perform bootstrapping or not
        random_state : int, default=0
            Random seed

    Attributes
    ----------
        y_probs: ndarray
            If y is in prob format, then target is a copy of y. Otherwise it is y in prob format
        classifiers: list of estimators
            The collection of fitted estimators
        ys: list of ndarrays
            The collection of sampled target labels. Each ndarray in ys has the same shape as y
        Xs: list of ndarrays
            The collection of bootstrapped datasets. Each ndarray in Xs has the same shape as X.
            If resample=False, then Xs is empty.
        n_classes: int
            The number of unique classes in y
        classes_: int
            The unique classes in y

    Methods
    -------
        fit(X, y)
            Fit the RRLClassifier model
        transform(X, y)
            Transform the data (only X, y is ignored) using the support_ attribute of the fitted model
        fit_transform(X, y)
            Fit to data, then transform it

    References
    ----------
    [1] Campagner, A., Ciucci, D., Svensson, C. M., Figge, M. T., & Cabitza, F. (2021).
        Ground truthing from multi-rater labeling with three-way decision and possibility theory.
        Information Sciences, 545, 771-790.
        https://doi.org/10.1016/j.ins.2020.09.049
    '''
  def __init__(self, estimator=DecisionTreeClassifier, n_estimators=100, missing=False, probs=None, resample=False, random_state=0):
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.estimator = estimator
    self.probs = probs
    self.resample = resample
    self.missing = missing

  def fit(self, X, y):
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
    output = np.zeros((X.shape[0], self.n_classes_))
    sorter = np.argsort(self.classes_)
    for i in range(self.n_estimators):
      indices = sorter[np.searchsorted(self.classes_, self.classifiers[i].classes_, sorter=sorter)]
      output[:, indices] += self.classifiers[i].predict_proba(X)
    return self.classes_[np.argmax(output, axis= 1)]

  def predict_proba(self, X):
    output = np.zeros((X.shape[0], self.n_classes_))
    sorter = np.argsort(self.classes_)
    for i in range(self.n_estimators):
      indices = sorter[np.searchsorted(self.classes_, self.classifiers[i].classes_, sorter=sorter)]
      output[:, indices] += self.classifiers[i].predict_proba(X)
    return normalize(output, axis=1, norm='l1')