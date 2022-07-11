from functools import partial
import numpy as np
from keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from keras.metrics import accuracy
from keras.wrappers.scikit_learn import BaseWrapper
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.regularizers import l2
from keras.losses import kullback_leibler_divergence
import keras.backend as keras_backend
from sklearn.preprocessing import OneHotEncoder


class LabelRelaxationLoss(object):
    """
    Wrapper object that provides the label relaxation loss as described in "From Label Smoothing to Label Relaxation" by
    Julian Lienen and Eyke Huellermeier, AAAI 2021.

    Parameters
    ----------
    :param alpha: Imprecisiation degree alpha of the label relaxation loss
    :type alpha: float, default=0.1
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()

        self._alpha = alpha

    def loss(self, y_true, y_pred):
        if self._alpha is None:
            y_true, alphas = y_true
        else:
            alphas = self._alpha

        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, keras_backend.epsilon(), 1. - keras_backend.epsilon())

        sum_y_hat_prime = tf.reduce_sum((1. - y_true) * y_pred, axis=-1)
        y_pred_hat = alphas * y_pred / (tf.expand_dims(sum_y_hat_prime, axis=-1) + keras_backend.epsilon())
        y_true_credal = tf.where(tf.greater(y_true, 0.1), 1. - alphas, y_pred_hat)

        divergence = kullback_leibler_divergence(y_true_credal, y_pred)

        preds = tf.reduce_sum(y_pred * y_true, axis=-1)

        return tf.where(tf.greater_equal(preds, 1. - alphas), tf.zeros_like(divergence),
                        divergence)


def accuracy_metric(y_true, y_pred, tuple_target=False, n_classes=10):
    if tuple_target:
        targets = tf.reshape(y_true[0], (-1, n_classes))
    else:
        targets = y_true

    # Argmax as we face probability distributions
    y_pred = tf.argmax(y_pred, axis=-1)
    targets = tf.argmax(targets, axis=-1)
    return accuracy(targets, y_pred)


class LabelRelaxationNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple MLP-based classifier using the label relaxation loss as optimization criterion.

    Parameters
    ----------
    :param lr_alpha: Imprecisiation degree alpha of the label relaxation loss
    :type lr_alpha: float, default=0.1

    :param hidden_layer_sizes: Tuple consisting of the individual hidden layer sizes used by the underlying NN model
    :type hidden_layer_sizes: tuple, default=(100,)

    :param activation: Activation function applied to the hidden layers' activations
    :type activation: str, default 'relu'

    :param l2_penalty: L2 norm regularization parameter applied to all neural network layers
    :type l2_penalty: float, default=1e-4

    :param learning_rate: Learning rate used by the SGD optimizer
    :type learning_rate: float, default=1e-3

    :param momentum: Momentum used by the SGD optimizer
    :type momentum: float, default=0.0

    :param epochs: Number of training epochs
    :type epochs: int, default=100

    :param batch_size: Batch size for training
    :type batch_size: int, default=None

    :param provide_alphas: Indicator whether we consider tuples as targets consisting of the classes and their
        imprecisiation
    :type provide_alphas: bool, default=False

    :param n_classes: Number of classes in case we want to be certain about the dimensionality of the one-hot encoding
    :type n_classes: int default=None
    """

    def __init__(self, lr_alpha: float = 0.1, hidden_layer_sizes: tuple = (100,), activation: str = "relu",
                 l2_penalty: float = 1e-4, learning_rate: float = 1e-3, momentum: float = 0.0, epochs: int = 100,
                 batch_size: int = None, provide_alphas: bool = False, n_classes: int = None):
        super().__init__()

        self.lr_alpha = lr_alpha
        self._lr_loss = LabelRelaxationLoss(self.lr_alpha)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        if self.n_classes is not None:
            self.classes_ = np.arange(int(self.n_classes))

        self._internal_model = None

        self.provide_alphas = provide_alphas

    def _one_hot_encoding(self, targets):
        if self.n_classes is None:
            enc = OneHotEncoder(sparse=False)
            return enc.fit_transform(targets.reshape(-1, 1))
        else:
            one_hot_encoded = np.zeros((len(targets), self.n_classes))
            one_hot_encoded[np.arange(len(targets)), targets] = 1
            return one_hot_encoded

    def fit(self, X, y):
        """
        Fits the label relaxation model. The targets y are one-hot encoded in case a simple list is provided.
        """
        input_dim = X.shape[1]

        targets = y if not self.provide_alphas else y[0]
        if len(targets.shape) < 2:
            targets = self._one_hot_encoding(targets)

        if self.n_classes is None:
            self.n_classes = targets.shape[1]
        if self.classes_ is None:
            self.classes_ = np.arange(int(self.n_classes))

        if self.provide_alphas:
            y = (targets, y[1])
        else:
            y = targets

        def _create_model():
            model = Sequential()
            model.add(Input(input_dim))
            for hl_size in self.hidden_layer_sizes:
                model.add(
                    Dense(hl_size, activation=self.activation, kernel_regularizer=l2(self.l2_penalty)))
            model.add(Dense(self.n_classes, activation="softmax", kernel_regularizer=l2(self.l2_penalty)))

            optimizer = SGD(lr=self.learning_rate, momentum=self.momentum)
            acc_metric = partial(accuracy_metric, tuple_target=self.provide_alphas, n_classes=self.n_classes)
            acc_metric.__name__ = "accuracy"
            model.compile(loss=self._lr_loss.loss, optimizer=optimizer,
                          metrics=[acc_metric])
            return model

        self._internal_model = BaseWrapper(build_fn=_create_model, epochs=self.epochs, batch_size=self.batch_size)
        self._internal_model.fit(X, y)
        return self

    def predict(self, X):
        assert self._internal_model is not None, "Model needs to be fit before prediction."

        return self._internal_model.model.predict(X)

    def predict_proba(self, X):
        # As last layer is a softmax layer
        return self.predict(X)
