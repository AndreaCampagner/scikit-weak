import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from ..utils import fuzzy_hinge, fuzzy_cross_entropy

class GRMLinearClassifier(BaseEstimator, ClassifierMixin):
    '''
    A class to perform classification for weakly supervised data, based on the Generalized Risk Minimization Paradigm
    applied to linear classifiers (either, logistic regression or linear SVM).
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel

    Parameters
    ----------
    :param loss: The loss function to optimize
    :type loss: str or callable, default 'logistic'

    :param max_epochs: The number of epochs to train the model
    :type max_epochs: int, default 100

    :param optimizer: The optimizer algorithm
    :type optimizer: str or callable, default 'sgd'

    :param regularizer: The type of regularization to apply
    :type regularizer: str or callable, default None

    :param l1: The regularization coefficient for l1 regularization, used only if regularizer='l1'
    :type l1: float, default 0.01

    :param l2: The regularization coefficient for l2 regularization, used only if regularizer='l2'
    :type l2: float, default 0.01

    :param batch_size: Size of the mini-batches
    :type batch_size: int, default 32



    Attributes
    ----------
    :ivar regularization: The regularization method instance
    :vartype y: tf.keras.regularizers.Regularizer

    :ivar model: The fitted linear model
    :vartype model: tf.keras.Model

    :ivar loss_funct: The loss function callable
    :vartype loss_funct: callable

    :ivar __n_classes: The number of unique classes in y
    :vartype __n_classes: int

    :ivar __classes: The unique classes in y
    :vartype __classes: list of int
    '''
    
    def __init__(self, loss="logistic", max_epochs = 100, optimizer='sgd', random_state=None,
                 regularizer=None, l1=0.01, l2=0.01, batch_size=32):
        self.loss = loss
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.random_state = random_state
        self.regularizer = regularizer
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        
    def fit(self, X, y):
        """
        Fit the GRMLinearClassifier model
        """
        tf.config.run_functions_eagerly(True)
        np_state = np.random.get_state()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
        
        self.loss_funct = None
        self.model = None
        self.__X = X
        self.__y = y

        self.__n_classes = self.__y[0].n_classes

        self.__y_probs = np.empty((self.__X.shape[0], self.__n_classes))

        for i in range(len(self.__y)):
            self.__y_probs[i] = self.__y[i].to_probs()
        
        self.__n_features = self.__X.shape[1]
        
        self.regularization = None
        if self.regularizer == "l1":
            self.regularization = tf.keras.regularizers.l1(self.l1)
        elif self.regularizer == "l2":
            self.regularization = tf.keras.regularizers.l1(self.l2)
        elif self.regularizer == "l1_l2":
            self.regularization = tf.keras.regularizers.l1_l2(self.l1, self.l2)
        
        self.model = None
        if self.loss == "logistic":
            self.loss_funct = fuzzy_cross_entropy
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.__n_classes,
                                      activation='softmax',
                                      kernel_regularizer=self.regularization,
                                      input_shape=[self.__n_features])
            ])
        elif self.loss == "hinge":
            self.loss_funct = fuzzy_hinge
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.__n_classes,
                                      activation='linear',
                                      kernel_regularizer=self.regularization,
                                      input_shape=[self.__n_features])
            ])
        print(tf.executing_eagerly())
        self.model.compile(loss = self.loss_funct,
                           optimizer = self.optimizer)
        self.model.fit(self.__X, self.__y_probs, batch_size = self.batch_size, epochs = self.max_epochs, verbose=0)

        if self.random_state is not None:
            np.random.set_state(np_state)
            tf.random.set_seed(np.random.randint(10000000))

        tf.config.run_functions_eagerly(False)
        
        return self
        
    def predict_proba(self, X):
        """
        Returns probability distributions for the given X
        """
        return self.model.predict(X)

    def predict(self, X):
        """
        Returns predictions for the given X
        """
        y_pred = np.zeros(X.shape[0])
        preds = self.predict_proba(X)
        for i in range(len(y_pred)):
            y_pred[i] = np.argmax(np.add.reduce(preds[i,:]))
        return y_pred