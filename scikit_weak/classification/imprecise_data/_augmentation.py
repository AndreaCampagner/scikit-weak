from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class AugmentedClassifier(BaseEstimator, ClassifierMixin):
  '''
  A class to perform classification for imprecise data, based on Data Augmentation.
  The X input to the fit method should be given as an iterable of GenericWeakLabel  that implements the sample_value method.

  Parameters
  ----------
  :param estimator: Base estimator objects to be fitted. Should support predict and predict_proba
  :type estimator: estimator class, default=None

  :param num_samples: The number of samples for data augmentation
  :type num_samples: int, default=100
    
  :param random_state: Random seed
  :type random_state: int, default=None

  Attributes
  ----------
  :ivar augm_X: The augmented X data
  :vartype augm_X: ndarray

  :ivar augm_y: The augmented y data
  :vartype augm_y: ndarray

  :ivar estimator: The fitted model
  :vartype estimator: estimator
  '''
  def __init__(self, estimator=None, num_samples=100, random_state=None):
    self.estimator = estimator
    self.num_samples = num_samples
    self.random_state = random_state

  def fit(self, X, y):
    self.X = X
    self.y = y
    
    state = np.random.get_state()
    if not (self.random_state is None):
      np.random.seed(self.random_state)

    self.augm_X = np.zeros((X.shape[0]*self.num_samples, X.shape[1]))
    self.augm_y = np.zeros(len(y)*self.num_samples)
    self.__sample_data()
    self.estimator.fit(self.augm_X, self.augm_y)
    
    if not (self.random_state is None):
      np.random.set_state(state)
    return self

  def __sample_data(self):
    for i in range(self.X.shape[0]):
      for j in range(self.X.shape[1]):
        for t in range(self.num_samples):
          self.augm_X[i*self.num_samples + t, j] = self.X[i,j].sample_value()
          self.augm_y[i*self.num_samples + t] = self.y[i]

  def predict(self, X):
    return self.estimator.predict(X)

  def predict_proba(self, X):
    return self.estimator.predict_proba(X)
