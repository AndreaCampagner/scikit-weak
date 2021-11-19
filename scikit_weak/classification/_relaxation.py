from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import kullback_leibler_divergence
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.optimizers import SGD
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
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, keras_backend.epsilon(), 1. - keras_backend.epsilon())

        sum_y_hat_prime = tf.reduce_sum((1. - y_true) * y_pred, axis=-1)
        y_pred_hat = self._alpha * y_pred / (tf.expand_dims(sum_y_hat_prime, axis=-1) + keras_backend.epsilon())
        y_true_credal = tf.where(tf.greater(y_true, 0.1), 1. - self._alpha, y_pred_hat)

        divergence = kullback_leibler_divergence(y_true_credal, y_pred)

        preds = tf.reduce_sum(y_pred * y_true, axis=-1)

        return tf.where(tf.greater_equal(preds, 1. - self._alpha), tf.zeros_like(divergence),
                        divergence)


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
    """

    def __init__(self, lr_alpha: float = 0.1, hidden_layer_sizes: tuple = (100,), activation: str = "relu",
                 l2_penalty: float = 1e-4, learning_rate: float = 1e-3, momentum: float = 0.0, epochs: int = 100,
                 batch_size: int = None):
        super().__init__()

        self._lr_loss = LabelRelaxationLoss(lr_alpha)
        self._hidden_layer_sizes = hidden_layer_sizes
        self._hidden_layer_activation = activation
        self._l2_penalty = l2_penalty
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._epochs = epochs
        self._batch_size = batch_size

        self._internal_model = None

    def fit(self, X, y):
        """
        Fits the label relaxation model. The targets y are one-hot encoded in case a simple list is provided.
        """
        input_dim = X.shape[1]

        if len(y.shape) < 2:
            enc = OneHotEncoder(sparse=False)
            y = enc.fit_transform(y.reshape(-1, 1))
        n_classes = y.shape[1]

        def _create_model():
            model = Sequential()
            model.add(Input(input_dim))
            for hl_size in self._hidden_layer_sizes:
                model.add(
                    Dense(hl_size, activation=self._hidden_layer_activation, kernel_regularizer=l2(self._l2_penalty)))
            model.add(Dense(n_classes, activation="softmax", kernel_regularizer=l2(self._l2_penalty)))

            optimizer = SGD(lr=self._learning_rate, momentum=self._momentum)
            model.compile(loss=self._lr_loss.loss, optimizer=optimizer,
                          metrics=["accuracy"])
            return model

        self._internal_model = KerasClassifier(build_fn=_create_model, epochs=self._epochs, batch_size=self._batch_size)
        self._internal_model.fit(X, y)
        return self

    def predict(self, X):
        assert self._internal_model is not None, "Model needs to be fit before prediction."

        return self._internal_model.predict(X)

    def predict_proba(self, X):
        assert self._internal_model is not None, "Model needs to be fit before prediction."

        return self._internal_model.predict_proba(X)
