import numpy as np
import itertools
from sklearn.base import BaseEstimator, TransformerMixin, clone
from ..classification import WeaklySupervisedKNeighborsClassifier, WeaklySupervisedRadiusClassifier
from sklearn.metrics import accuracy_score


class RoughSetSelector(TransformerMixin, BaseEstimator):
    """
    A class to perform Feature Selection based on Rough Sets by searching for reducts [1].
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel.
    Supports both discrete (using Pawlak Rough Sets) and continuous (using Neighborhood Rough Sets) datasets.

    Parameters
    ----------
    :param search_strategy: The search strategy to be used. 'approximate' is similar to RFE, having complexity O(n^2). 'brute' is a brute-force search strategy, all possible combinations of features are evaluated, with complexity O(2^n)
    :type search_strategy: {'approximate', 'brute'}, default='approximate'

    :param epsilon: The approximation factor. Should be a number between 0.0 and 1.0 (excluded)
    :type epsilon: float, default=0.0

    :param n_iters: Number of iterations to be used when search_strategy='approximate'. Not used if search_strategy='brute'
    :type n_iters: int, default=100

    :param discrete: Whether the input X is discrete or not. If discrete=True then use equivalence-based
        (i.e. Pawlak) Rough Sets. If discrete=False use neighborhood-based Rough Sets.
    :type discrete: bool, default=True
        
    :param metric: Metric to be used with neighborhood-based Rough Sets. Only used if discrete=False. If discrete=True, then metric="hamming"
    :type metric: string or function, default='minkowski'
        
    :param neighborhood: Type of neighborhood-based Rough Sets to be used. If neighborhood='delta', then
        use delta-neighborhood Rough Sets: all neigbhors with distance <= radius are selected.
        If neighborhood='nearest', then use k-nearest-neighbors Rough Sets: only the k
        nearest neighbors are selected. Only used if discrete=False
    :type neighborhood: {'delta', 'nearest'}, default='nearest'

    :param n_neighbors: Number of nearest neighbors to select. Only used if discrete=False and
        neighborhood='nearest'
    :type n_neighbors: int, default=3

    :param radius: Radius to select neighbors. Only used if discrete=False and neighborhood='delta'
    :type radius: float, default=1.0

    :param random_state: Randomization seed. Used only if search_strategy='approximate'
    :type random_state: int, default=None

    Attributes
    ----------
    :ivar __n_classes: The number of unique classes in y
    :vartype n_classes: int

    :ivar __reducts: The list of minimal reducts. If search_strategy='approximate', reducts always contains a single set of features. If search_strategy='brute', reducts contains the list of all minimal reducts.
    :vartype __reducts: list
    """
    def __init__(self, search_strategy='approximate', epsilon = 0.0,
                 n_iters = 100, discrete = True, metric='minkowski',
                 neighborhood='nearest', n_neighbors=3, radius=1.0, random_state = None):
        self.search_strategy = search_strategy
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.random_state = random_state
        self.discrete = discrete
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.neighborhood = neighborhood
        self.metric = metric
        
    def fit(self, X, y):
        """
        Fit the RoughSetSelector model
        """
        state = np.random.get_state()
        if not (self.random_state is None):
            np.random.seed(self.random_state)


        self.__X = X
        self.__dim = self.__X.shape[1]
        self.__reducts = []
        self.__y = np.array(y)
        self.__n_classes = self.__y[0].n_classes
            
        self.__nn = None
        if self.discrete:
            self.__nn = WeaklySupervisedRadiusClassifier(radius = 0.0,
                                          metric = 'hamming')
        elif self.neighborhood == 'delta':
            self.__nn = WeaklySupervisedRadiusClassifier(radius=self.radius,
                                           metric=self.metric)
        elif self.neighborhood == 'nearest':
            self.__nn = WeaklySupervisedKNeighborsClassifier(k = self.n_neighbors,
                                      metric = self.metric)
        else:
            raise ValueError("%s is not an allowed value for 'neighborhood' parameter" % self.neighborhood)

        self.__nn_all = clone(self.__nn)
        self.__nn_all.fit(self.__X, self.__y)
        
        if self.search_strategy == 'approximate':
            self.__reducts = list(self.__find_approx_reducts())
        elif self.search_strategy == 'brute':
            self.__reducts = self.__find_reducts()
        else:
            raise ValueError("%s is not an allowed value for 'search_strategy' parameter." % self.search_strategy)

        if not (self.random_state is None):
            np.random.set_state(state)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the data (only X, y is ignored) selecting a reduct at random
        """
        state = np.random.get_state()
        if not (self.random_state is None):
            np.random.seed(self.random_state)
            
        feature_set = np.random.choice(len(self.__reducts))

        if not (self.random_state is None):
            np.random.set_state(state)

        return X[:, self.__reducts[feature_set]]
    
    def fit_transform(self, X, y):
        """
        Fit and then transform data
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def __find_reducts(self):
        reducts = []
        min_len = self.__dim
        for k in range(self.__dim, 0, -1):
            for i in itertools.combinations(range(self.__dim), k):
                red = np.array(list(i))
                temp_X = self.__X[:, red].copy()
                self.__nn.fit(temp_X, self.__y)

                accuracy = accuracy_score(self.__nn_all.predict(self.__X), self.__nn.predict(temp_X))
                check = (accuracy >= (1 - self.epsilon))
               
                if check:
                    if len(red) < min_len:
                        min_len = len(red)
                        reducts = [red]
                    elif len(red) == min_len:
                        reducts.append(red)
        return np.array(reducts)
    
    def __find_approx_reducts(self):
        total_red = range(self.__dim)

        for iter in range(self.n_iters):      
            red = range(self.__dim)
            check = True
            while check == True and len(red) > 1:
                check = False
                to_keep = red
                combs = itertools.combinations(red, len(red) - 1)
                combs = np.random.permutation(list(combs))
                for i in combs:

                    temp_red = np.array(list(i))
                    temp_X = self.__X[:, temp_red].copy()
                    self.__nn.fit(temp_X, self.__y)

                    accuracy = accuracy_score(self.__nn_all.predict(self.__X), self.__nn.predict(temp_X))
                    check_acc = (accuracy >= (1 - self.epsilon))
                
                    if check_acc:
                        to_keep = temp_red
                    
                if len(to_keep) < len(red):
                    red = to_keep
                    check = True
            if len(red) < len(total_red):
                total_red = red
        return total_red