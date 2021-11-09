import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from sklearn.base import clone


class FitnessVal(ABC):
    @abstractmethod
    def __gt__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass
    
class SimpleVal(FitnessVal):
    def __init__(self, val):
        self.val = val

    def __gt__(self, other):
        return  self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

class SimplePair(FitnessVal):
    def __init__(self, reduct_size, poss_value):
        self.reduct_size = reduct_size
        self.poss_value = poss_value

    def __gt__(self, other):
        check = False

        if (self.reduct_size < other.reduct_size) and (self.poss_value >= other.poss_value):
            check = True
        elif (self.reduct_size <= other.reduct_size) and (self.poss_value > other.poss_value):
            check = True
        
        return check

    def __eq__(self, other):
        return not((self > other) or (other > self))

class ArrayPair(FitnessVal):
    def __init__(self, reduct_size, poss_array):
        self.reduct_size = reduct_size
        self.poss_array = poss_array

    def __gt__(self, other):
        check = False

        weak = np.all(self.poss_array >= other.poss_array)
        strong = np.any(self.poss_array > other.poss_array)

        if (self.reduct_size < other.reduct_size) and (weak):
            check = True
        elif (self.reduct_size == other.reduct_size) and (weak and strong):
            check =True

        return check

    def __eq__(self, other):
        return not((self > other) or (other > self))


class GeneticRoughSetSelector(TransformerMixin, BaseEstimator):
    """
    A class to perform Rough Set-based feature selection, by searching for reducts, using Genetic Algorithms.
    The y input to the fit method should be given as an iterable of DiscreteWeakLabel.
    Supports both discrete (using Pawlak Rough Sets) and continuous (using Neighborhood Rough Sets) datasets.

    Parameters
    ----------
    :param epsilon: The approximation factor. Should be a number between 0.0 and 1.0 (excluded)
    :type epsilon: float, default=0.0

    :param method: The method used to compute the fitness. If 'lambda', then the algorithm solves a single objective optimization problem
        If 'conservative' or 'dominance' solves a multiple objective optimization problem: in particular, if 'conservative' a 2-objectives problem
        and if 'dominance' (n+1)-objectives problem, where n is the number of instances
    :type method: {'lambda', 'conservative', 'dominance'}, default='conservative'

    :param discrete: Whether the input X is discrete or not. If discrete=True then use equivalence-based
        (i.e. Pawlak) Rough Sets. If discrete=False use neighborhood-based Rough Sets.
    :type discrete: bool, default=True
        
    :param l: Lambda interpolation factor. Only used if method='lambda'
    :type l: float in [0,1], default=0.5

    :param tournament_size: Proportion of population to select from in tournament selection.
    :type tournament_size: float in [0,1], default=0.1

    :param p_mutate: Proability of point mutation
    :type p_mutate: float in [0,1], default=0.1

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

    :param population_size: Size of the population for the Genetic Algorithm
    :type population_size: int, default=100

    :param n_iters: Number of generations for the Genetic Algorithm
    :type n_iters: int, default=100

    Attributes
    ----------
    :ivar n_classes: The number of unique classes in y
    :vartype n_classes_: int

    :ivar best_features_: The unique most fit feature sets.
    :vartype best_features_: ndarray

    :ivar best_targets_: The disambiguated targets corresponing to the most fit feature sets. Can be used for transductive learning or training a downstream model.
    :vartype best_target_: ndarray
    """
    def __init__(self, epsilon = 0.0, method = 'conservative',
                 discrete = False, l = 0.5, tournament_size = 0.1, p_mutate=0.1,
                 metric='minkowski', neighborhood='nearest', n_neighbors=3, radius=1.0,
                 random_state = None, population_size = 100, n_iters = 100):
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.random_state = random_state
        self.discrete = discrete
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.neighborhood = neighborhood
        self.metric = metric
        self.population_size = population_size
        self.method = method
        self.l = l
        self.tournament_size = tournament_size
        self.p_mutate = p_mutate

    def _initialize(self):
        self.__population_features = np.random.choice(a=[False, True], size=(self.population_size, self.__dim))
        self.__population_targets = np.empty((self.population_size, self.__n_instances), dtype=self.__basic_type)
        self._true_targets = np.empty((self.population_size, self.__n_instances), dtype=object)
        for i in range(self.population_size):
            self.__population_targets[i, :] = [self.y[j].sample_value() for j in range(self.__n_instances)]
            self._true_targets[i, :] = self.y

    def __breed(self):
        i = 0
        features_temp = self.__population_features.copy()
        targets_temp = self.__population_targets.copy()
        true_targets_temp = self._true_targets.copy()
        for c in range(int(self.population_size/2)):
            parent_1 = self.__select()
            parent_2 = self.__select()

            child_1, child_2 = self.__crossover(parent_1, parent_2)
            
            child_1 = self.__mutate(child_1)
            child_2 = self.__mutate(child_2)

            features_temp[i,:] = child_1[0]
            targets_temp[i, :] = child_1[1]
            true_targets_temp[i, :] = child_1[2]

            features_temp[i+1,:] = child_2[0]
            targets_temp[i+1, :] = child_2[1]
            true_targets_temp[i+1, :] = child_2[2]

            i+=2

        self.__population_features = features_temp
        self.__population_targets = targets_temp
        self._true_targets = true_targets_temp

    def __select(self):
        indices = np.random.choice(self.population_size,
                                   int(self.population_size*self.tournament_size),
                                   replace=False)

        ind = indices[0]
        for i in indices:
            if self.__fitnesses[i] > self.__fitnesses[ind]:
                ind = i
        
        return ind

    def __mutate(self, child):
        features = child[0]
        targets = child[1]

        draw = np.random.choice([True, False],
                                self.__dim,
                                p=[self.p_mutate, 1 - self.p_mutate])
        features = features ^ draw

        draw = np.random.choice([True, False],
                                self.__n_instances,
                                p=[self.p_mutate, 1 - self.p_mutate])

        indices = np.array(range(self.__n_instances))[draw]
        for i in indices:
            targets[i] = child[2][i].sample_value()

        return (features, targets, child[2])

    def __crossover(self, a, b):
        features_split_ind = np.random.choice(self.__dim)
        target_split_ind = np.random.choice(self.__n_instances)

        child_1_features = self.__population_features[a].copy()
        child_1_features[features_split_ind:] = self.__population_features[b, features_split_ind:]
        child_1_targets = self.__population_targets[a].copy()
        child_1_targets[target_split_ind:] = self.__population_targets[b, target_split_ind:]
        child_1_true_targets = self._true_targets[a].copy()
        child_1_true_targets[target_split_ind:] = self._true_targets[b, target_split_ind:]

        child_2_features = self.__population_features[b].copy()
        child_2_features[features_split_ind:] = self.__population_features[a, features_split_ind:]
        child_2_targets = self.__population_targets[b].copy()
        child_2_targets[target_split_ind:] = self.__population_targets[a, target_split_ind:]
        child_2_true_targets = self._true_targets[b].copy()
        child_2_true_targets[target_split_ind:] = self._true_targets[a, target_split_ind:]

        child_1 = (child_1_features, child_1_targets, child_1_true_targets)
        child_2 = (child_2_features, child_2_targets, child_2_true_targets)
        return child_1, child_2
                

    def __compute_fitness(self, p):
        features = self.__population_features[p,:]
        
        temp_X = self.X[:, features]
        temp_y = self.__population_targets[p]

        reduct_val = self.__evaluate_reduct(temp_X, temp_y)
        poss_val = self.__evaluate_poss(temp_y)
        
        if self.method == 'conservative':
            return SimplePair(reduct_val, poss_val)
        elif self.method == 'lambda':
            return (1 - self.l)*poss_val - self.l*reduct_val/self.__dim
        elif self.method == 'dominance':
            return ArrayPair(reduct_val, poss_val)
        else:
            raise ValueError("%s is not an allowed value for 'method' parameter" % self.method)



    def __evaluate_poss(self, temp_y):
        poss_values = np.zeros(self.__n_instances)
        for i in range(self.__n_instances):
            poss_values[i] = self.y[i][temp_y[i]]

        if self.method == 'conservative' or self.method == 'lambda':
            return np.amin(poss_values)
        elif self.method == 'dominance':
            return poss_values
        else:
            raise ValueError("%s is not an allowed value for 'method' parameter" % self.method)



    def __evaluate_reduct(self, temp_X, temp_y):
        nn = None
        if self.discrete:
            nn = RadiusNeighborsClassifier(radius = 0.0,
                                          metric = 'hamming')
        elif self.neighborhood == 'delta':
            nn = RadiusNeighborsClassifier(radius=self.radius,
                                           metric=self.metric)
        elif self.neighborhood == 'nearest':
            nn = KNeighborsClassifier(n_neighbors = self.n_neighbors,
                                      metric = self.metric)
        else:
            raise ValueError("%s is not an allowed value for 'neighborhood' parameter" % self.neighborhood)
        
        if temp_X.shape[1] == 0:
            return np.inf

        nn_all = clone(nn)
        nn.fit(temp_X, temp_y)
        nn_all.fit(self.X, temp_y)

        accuracy = accuracy_score(nn_all.predict(self.X), nn.predict(temp_X))
        check = (accuracy >= (1 - self.epsilon))
        
        return temp_X.shape[1] if check else np.inf
        
    def fit(self, X, y):
        """
        Fit the GeneticRoughSetSelector model
        """
        state = np.random.get_state()
        if not (self.random_state is None):
            np.random.seed(self.random_state)


        self.X = X
        self.y = np.array(y)
        self.__basic_type = y[0].basic_type()
        self.__dim = X.shape[1]
        self.__n_instances = X.shape[0]
        self.n_classes_ = y[0].n_classes
        self._initialize()

        self.__fitnesses = np.empty(self.population_size, dtype=FitnessVal)
        for i in range(self.n_iters):
            self.__best_value = self.__compute_fitness(0)
            self.__fitnesses[0] = self.__best_value
            self.best_features_ = [self.__population_features[0,:]]
            self.best_targets_ = [self.__population_targets[0]]

            
            for p in range(1,self.population_size):
                self.__fitnesses[p] = self.__compute_fitness(p)
                if self.__fitnesses[p] == self.__best_value:
                    self.best_features_.append(self.__population_features[p,:])
                    self.best_targets_.append(self.__population_targets[p])
                elif self.__fitnesses[p] > self.__best_value:
                    self.__best_value = self.__fitnesses[p]
                    self.best_features_ = [self.__population_features[p,:]]
                    self.best_targets_ = [self.__population_targets[p]]

            self.__breed()

        if not (self.random_state is None):
            np.random.set_state(state)

        self.best_features_ = np.array(self.best_features_)
        self.best_targets_ = np.array(self.best_targets_)
        _, idx = np.unique(self.best_features_, axis=0, return_index=True)
        self.best_features_ = self.best_features_[idx]
        self.best_targets_ = self.best_targets_[idx]

        return self

    def transform(self, X, y=None):
        """
        Transform the data (only X, y is ignored) selecting a reduct at random
        """
        state = np.random.get_state()
        if not (self.random_state is None):
            np.random.seed(self.random_state)
            
        feature_set = np.random.choice(len(self.best_features_))

        if not (self.random_state is None):
            np.random.set_state(state)

        return X[:, self.best_features_[feature_set]]

    def fit_transform(self, X, y):
        """
        Fit and then transform data
        """
        self.fit(X,y)
        return self.transform(X, y)

