import numpy as np
import tensorflow as tf
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin

class GRMLinearClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, loss="logistic", max_iter = 100, optimizer='sgd', random_state=0,
                 regularizer=None, l1=0.01, l2=0.01, batch_size=32):
        self.loss = loss
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.random_state = random_state
        self.regularizer = regularizer
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        
    def fit(self, X, y):
        np_state = np.random.get_state()
        np.random.seed(self.random_state)
        
        tf.random.set_seed(self.random_state)
        
        self.loss_funct = None
        self.model = None
        self.X = X
        self.y = y
        
        self.n_features = X.shape[1]
        self.n_classes = self.y.shape[1]
        
        self.regularization = None
        if self.regularizer == "l1":
            self.regularization = tf.keras.regularizers.l1(self.l1)
        elif self.regularizer == "l2":
            self.regularization = tf.keras.regularizers.l1(self.l2)
        elif self.regularizer == "l1_l2":
            self.regularization = tf.keras.regularizers.l1_l2(self.l1, self.l2)
        
        self.model = None
        if self.loss == "logistic":
            self.loss_funct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.n_classes,
                                      activation='softmax',
                                      kernel_regularizer=self.regularization,
                                      input_shape=[self.n_features])
            ])
        elif self.loss == "hinge":
            self.loss_funct = tf.keras.losses.CategoricalHinge(reduction=tf.keras.losses.Reduction.NONE)
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.n_classes,
                                      activation='linear',
                                      kernel_regularizer=self.regularization,
                                      input_shape=[self.n_features])
            ])
        
        self.grm_loss_funct = partial(superset_loss, loss_function = self.loss_funct)
        self.model.compile(loss = self.grm_loss_funct,
                           optimizer = self.optimizer)
        self.model.fit(self.X, self.y, batch_size = self.batch_size, epochs = self.max_iter)
        
        np.random.set_state(np_state)
        tf.random.set_seed(np.random.randint(10000000))
        return self
        
    def predict(self, X):
        return self.model.predict(X)