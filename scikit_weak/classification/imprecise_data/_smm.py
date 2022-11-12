from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SupportMeasureClassifier(BaseEstimator, ClassifierMixin):
  '''
	A class to perform classification for imprecise data, based on a generalization of the Support Measure Machine
    algorithm [1]. Supports only binary classification with labels in {-1, 1}.
    The X input to the fit method should be given as an iterable of RandomLabel.

    Parameters
	----------
	:param kernel: A kernel function, must be passed
	:type estimator: function, default=None

	:param n_iters: Number of optimization iterations
	:type n_estimators: int, default=100

	:param l: Regularization coefficient
	:type l: float, default=0.5
		
	:param probability: Whether the classifier supports the predict_proba method. If True, after model fitting, fits a Logistic Regression to calibrate the confidence scores
	:type probability: bool, default=False

    :param approximate: If False, performs fitting on full data. If True, performs fitting based on mini-batch stochastic gradient descent
    :type approximate: bool, default=False

    :param batch_size: Batch size for approximation based on stochastic gradient descent. Used only if approximate=True
    :type batch_size: int, default=1

	Attributes
	----------

	:ivar alpha_cum: The weights of the instances
	:vartype alpha_cum: ndarray

	:ivar beta: Accumulates the sub-gradients in kernel space
	:vartype beta: ndarray
  '''
  def __init__(self, kernel=None, n_iters=100, l = 0.5, probability=False, approximate=False, batch_size=1):
    self.kernel = kernel
    self.n_iters = n_iters
    self.l = l
    self.probability = probability
    self.approximate = approximate
    self.batch_size = batch_size

  def fit(self, X, y):
    self.__X = X
    self.__y = y
    self.beta = np.zeros(X.shape[0])
    self.alpha_cum = np.zeros(X.shape[0]) 

    for t in range(1,self.n_iters+1):
      alpha = self.beta/(self.l*t)
      i = np.random.randint(0, X.shape[0])
      
      accum = 0
      if self.approximate:
        for k in range(int(self.batch_size*X.shape[0])):
          j = np.random.randint(0, X.shape[0])
          accum += alpha[j]*self.kernel(self.__X[j,:], self.__X[i,:])
      else:
        for j in range(X.shape[0]):
          accum += alpha[j]*self.kernel(self.__X[j,:], self.__X[i,:])

      if y[i]*accum < 1:
        self.beta[i] = self.beta[i] + y[i]

      self.alpha_cum += alpha

    self.alpha_cum /= self.n_iters

    if self.probability:
      scores = np.zeros(self.__X.shape[0])
      for i in range(self.__X.shape[0]):
        for j in range(self.__X.shape[0]):
          scores[i] += self.alpha_cum[j]*self.kernel(self.__X[j,:], self.__X[i,:])

      self.cal = LogisticRegression()
      self.cal.fit(scores[:,None], y)
    return self

  def predict(self, X):
    y_pred = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
      accum = 0
      for j in range(self.__X.shape[0]):
        accum += self.alpha_cum[j]*self.kernel(self.__X[j,:], X[i,:])
      if accum >= 0:
        y_pred[i] = 1
      else:
        y_pred[i] = -1

    return y_pred

  def predict_proba(self, X):
    scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      for j in range(self.__X.shape[0]):
        scores[i] += self.alpha_cum[j]*self.kernel(self.__X[j,:], X[i,:])
    self.cal.predict_proba(scores[:,None])