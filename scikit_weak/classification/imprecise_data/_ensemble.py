from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import resample
import numpy as np

class WSF(BaseEstimator, ClassifierMixin):
	'''
	A class to perform classification for imprecise data, based on a generalization of the RRL algorithm.
	The X input to the fit method should be given as an iterable of GenericWeakLabel that implements the sample_value method.

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

	:ivar __classifiers: The collection of fitted estimators
	:vartype __classifiers: list of estimators

	:ivar __ys: The collection of sampled target labels. Each ndarray in ys has the same shape as y
	:vartype __ys: list of ndarrays
	
	:ivar __Xs: The collection of bootstrapped datasets. Each ndarray in Xs has the same shape as X. If resample=False, then Xs is empty.
	:vartype __Xs: list of ndarrays
	
	:ivar __n_classes: The number of unique classes in y
	:vartype __n_classes: int

	:ivar __classes: The unique classes in y
	:vartype __classes: list of int
	'''


	def __init__(self, estimator=ExtraTreeClassifier(), n_estimators=100, resample=False, random_state=None):
		self.n_estimators = n_estimators
		self.random_state = random_state
		self.estimator = estimator
		self.resample = resample

	def fit(self, X, y):
		"""
			Fit the WSF model
		"""
		state = np.random.get_state()
		if not (self.random_state is None):
			np.random.seed(self.random_state)

		self.__X = X
		self.__y = np.array(y)

		self.__n_classes = len(np.unique(self.__y))
		self.__classes = range(self.__n_classes)

		self.__classifiers = [ clone(self.estimator) for i in range(self.n_estimators)]

		self.__ys = []
		self.__Xs = []
		for i in range(self.n_estimators):
			temp_Xs = self.__sample_data(self.__X)
			seed = np.random.randint(np.iinfo('int32').max)
			self.__classifiers[i].set_params(**{'random_state': seed})

			if self.resample:
				seed = np.random.randint(np.iinfo('int32').max)
				temp_Xs, temp_ys = resample(temp_Xs, self.__y, random_state = seed)
				self.__Xs.append(temp_Xs)
				self.__ys.append(temp_ys)
				self.__classifiers[i].fit(self.__Xs[i], self.__ys[i])
			else:
				self.__Xs.append(temp_Xs)
				self.__classifiers[i].fit(self.__Xs[i], self.__y)

		if not (self.random_state is None):
			np.random.set_state(state)
		return self

	def __sample_data(self, data):
		X_res = np.empty(data.shape)
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				X_res[i,j] = self.__X[i,j].sample_value()
		return X_res

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