from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
import numpy as np

from scikit_weak.classification._relaxation import LabelRelaxationNNClassifier


class PseudoLabelsClassifier(BaseEstimator, ClassifierMixin):
    '''
      A class to perform classification for weakly supervised data, based on the pseudo-labels strategy.
      The y input to the fit method should be given as an iterable of GenericWeakLabel

      Parameters
      ----------
      :param estimator: Base estimator objects to be fitted. Should support predict and predict_proba
      :type estimator: estimator class, default=LogisticRegression

      :param n_restarts: The number of restarts
      :type n_restarts: int, default = 5

      :param n_iterations: The number of iterations for fitting
      :type n_iterations: int, default=10

      :param threshold: The threshold for pseudo-label selection
      :type threshold: float, default=0.5

      :param random_state: Random seed
      :type random_state: int, default=None

      Attributes
      ----------

      :ivar estimator: The last fitted estimator
      :vartype estimator: estimator

      :ivar __n_classes: The number of unique classes in y
      :vartype __n_classes: int

      :ivar __classes: The unique classes in y
      :vartype __classes: list of int
      '''

    def __init__(self, estimator=LogisticRegression(), n_iterations=10, n_restarts=5, threshold=0.5, random_state=None):
        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        """
            Fit the PseudoLabelsClassifier model
        """
        state = np.random.get_state()
        if not (self.random_state is None):
            np.random.seed(self.random_state)

        self.__X = X
        self.__y = np.array(y)

        self.__n_classes = self.__y[0].n_classes
        self.__classes = range(self.__n_classes)

        acc = 0

        for r in range(self.n_restarts):

            self.__fix = [False] * len(self.__y)
            self.__imputed_y = np.random.randint(0, self.__n_classes, self.__y.shape)

            temp_est = clone(self.estimator)

            for i in range(self.n_iterations):
                self.__sample_labels()
                temp_est.fit(self.__X, self.__imputed_y)

                y_proba = temp_est.predict_proba(self.__X)
                for t in range(y_proba.shape[0]):
                    if y_proba[t, self.__imputed_y[t]] > self.threshold:
                        self.__fix[t] = True

            y_pred = temp_est.predict(self.__X)
            temp_acc = 0
            for t in range(y_pred.shape[0]):
                temp_acc += self.__y[t][y_pred[t]]
            if temp_acc > acc:
                self.estimator = temp_est

        if not (self.random_state is None):
            np.random.set_state(state)
        return self

    def __sample_labels(self):
        for i in range(self.__y.shape[0]):
            if self.__fix[i] == False:
                self.__imputed_y[i] = self.__y[i].sample_value()

    def predict(self, X):
        """
            Returns predictions for the given X
        """
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """
            Returns probability distributions for the given X
        """
        return self.estimator.predict_proba(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class CSSLClassifier(BaseEstimator, ClassifierMixin):
    '''
      A credal self-supervised learning classifiers as described in "Credal Self-Supervised Learning" by
        Julian Lienen and Eyke Huellermeier, NeurIPS 2021. The model iteratively refines its beliefs about the true
        conditional class probability distribution for weakly labeled instances by maintaining credal sets induced by
        possibility distributions.

      Parameters
      ----------
      :param estimator: Base estimator objects to be fitted. Should support predict and predict_proba
      :type estimator: estimator class, default=LabelRelaxationNNClassifier

      :param n_iterations: The number of iterations for fitting
      :type n_iterations: int, default=10

      :param random_state: Random seed
      :type random_state: int, default=None

      :param p_data: Optional prior probability distribution of the class frequencies
      :type p_data: np.ndarray, default=None

      :param p_hist_buffer_size: Buffer size of the model prediction history
      :type p_hist_buffer_size: int, default=64

      Attributes
      ----------

      :ivar estimator: The last fitted estimator
      :vartype estimator: estimator

      :ivar _n_classes: The number of unique classes in y
      :vartype _n_classes: int
      '''

    def __init__(self, estimator=LabelRelaxationNNClassifier(provide_alphas=True), n_iterations=10,
                 n_classes=10, random_state=None, p_data=None, p_hist_buffer_size=64):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.estimator = estimator
        self.estimator.n_classes = n_classes

        self.n_classes = n_classes
        if p_data is not None:
            self.p_data = p_data
        else:
            self.p_data = np.ones(n_classes) / n_classes
        self.p_hist = np.ones([p_hist_buffer_size, self.n_classes]) / self.n_classes

    def fit(self, X, y):
        """
        y is list of discrete fuzzy labels
        """
        state = np.random.get_state()
        if not (self.random_state is None):
            np.random.seed(self.random_state)

        self.__X = X

        # Prepare targets and alphas
        self.__y = []
        self.__alphas = []
        for y_val in y:
            if y_val.is_agnostic():
                self.__y.append(-1)
                self.__alphas.append(1.)
            else:
                self.__y.append(np.argmax(y_val.classes))
                self.__alphas.append(0.)
        self.__y = np.array(self.__y)
        self.__alphas = np.array(self.__alphas)

        # Iteratively refine alpha values until there is no imprecisiation left or number of iterations is exhausted
        for i in range(self.n_iterations):
            # Fit model on current alphas
            train_mask = self.__y != -1
            train_X = self.__X[train_mask]
            train_y = self.__y[train_mask]
            train_alphas = self.__alphas[train_mask]

            self.estimator = self.estimator.fit(train_X, (train_y, train_alphas))

            # Update alphas
            update_mask = self.__alphas < 1.
            preds = self.estimator.predict(self.__X[update_mask])

            p_model = self.p_hist.mean(axis=0)
            p_model /= p_model.sum()

            norm_preds = preds * (self.p_data + 1e-7) / (p_model + 1e-7)
            norm_preds /= norm_preds.sum()

            self.__y[update_mask] = np.argmax(norm_preds, axis=-1)
            self.__alphas[update_mask] = 1. - np.max(norm_preds, axis=-1)

            # Update p_hist
            self.p_hist = np.concatenate((self.p_hist[1:], np.expand_dims(norm_preds.mean(axis=0), axis=0)), axis=0)

        if not (self.random_state is None):
            np.random.set_state(state)
        return self

    def predict(self, X):
        """
            Returns predictions for the given X
        """
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """
            Returns probability distributions for the given X
        """
        return self.estimator.predict_proba(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
