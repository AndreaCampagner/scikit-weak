from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import resample
import numpy as np

class RRLClassifier(BaseEstimator, ClassifierMixin):

  def __init__(self, n_estimators=100, random_state=0, max_depth = None, criterion = 'gini',
               max_features=None, class_weight='balanced', probs=None, resample=False, missing=True):
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.max_depth = max_depth
    self.criterion = criterion
    self.max_features = max_features
    self.class_weight = class_weight
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
      self.y_probs = y
      self.n_classes_ = self.y_probs.shape[1]
      self.classes_ = np.array(range(self.y_probs.shape[1]))
    
    self.classifiers = [ DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state + i,
                         max_features=self.max_features, class_weight=self.class_weight, criterion=self.criterion)
                         for i in range(self.n_estimators)]
    
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