import numpy as np
import itertools
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors

def oau_entropy(orthop, n_classes):
    orthop_copy = np.copy(orthop)
    probs = np.zeros([n_classes, len(orthop)])
    i = 0
    for elem in orthop:
        for cl in elem:
            probs[cl, i] = 1.0
        i += 1

    tots = np.sum(probs, axis=1)
    indices = np.argsort(tots)[::-1]
    for i in indices:
        for j in range(len(orthop)):
            if i in orthop_copy[j]:
                orthop_copy[j] = [i]
    orthop_copy = np.array([t[0] for t in orthop_copy])
    probs = np.zeros([n_classes])
    for elem in orthop_copy:
        probs[elem] += 1.0
    probs /= sum(probs)
    return stats.entropy(probs, base=2)

def bet_entropy(orthop, n_classes):
        probs = np.zeros([n_classes])
        unif = np.ones([n_classes])
        for elem in orthop:
            for cl in elem:
                probs[cl] += 1.0/len(elem)
        probs /= sum(probs)
        unif = unif * (probs > 0)
        unif /= sum(unif)
        num =  stats.entropy(probs, base=2)
        den = stats.entropy(unif, base=2)
        h = 0 if num == 0 else num/den
        return h


class RoughSetSelector(TransformerMixin, BaseEstimator):
    '''
    A class to perform Feature Selection based on Rough Sets by searching for entropy reducts [1].
    Support both fully supervised and weakly supervised data. In the latter case, the y input to
    the fit method should be given as a ndarray of lists, in which each list contains all candidate
    labels for the corresponding instance.
    Supports both discrete (using Pawlak Rough Sets) and continuous (using Neighborhood Rough Sets) datasets.

    Parameters
    ----------
        entropy : {'oau', 'bet'}, default='oau'
            The entropy measure to be used
        method : {'approximate', 'brute'}, default='approximate'
            The search strategy to be used. 'approximate' is similar to RFE, having complexity O(n^2).
            'brute' is a brute-force search strategy, all possible combinations of features are evaluated, with complexity O(2^n)
        epsilon : float, default=0.0
            The approximation factor. Should be a number between 0.0 and 1.0 (excluded)
        n_iters : int, default=100
            Number of iterations to be used when method='approximate'. Not used if method='brute'
        discrete: bool, default=True
            Whether the input X is discrete or not. If discrete=True then use equivalence-based
            (i.e. Pawlak) Rough Sets. If discrete=False use neighborhood-based Rough Sets.
        metric : strinf or function, default='minkowski'
            Metric to be used with neighborhood-based Rough Sets. Only used if discrete=False
        neighborhood: {'delta', 'nearest'}, default='nearest'
            Type of neighborhood-based Rough Sets to be used. If neighborhood='delta', then
            use delta-neighborhood Rough Sets: all neigbhors with distance <= radius are selected.
            If neighborhood='nearest', then use k-nearest-neighbors Rough Sets: only the k
            nearest neighbors are selected. Only used if discrete=False
        n_neighbors: int, default=3
            Number of nearest neighbors to select. Only used if discrete=False and
            neighborhood='nearest'
        radius: float, default=1.0
            Radius to select neighbors. Only used if discrete=False and neighborhood='delta'
        random_state: int, default=0
            Randomization seed. Used only if method='approximate'

    Attributes
    ----------
        target: ndarray of objects
            If y is weakly supervised, then target is a copy of y. Otherwise all single-valued elements are mapped to arrays
        entropy_function: function
            The entropy function to be used
        n_classes: int
            The number of unique classes in y
        reducts : list
            The list of reducts (sets of selected features with minimal entropy). 
            If method='approximate', reducts always contains a single set of features
        support_: ndarray of type Bool
            The mask of selected features. If method='brute' contains the mask corresponding to the first
            reduct in attribute reducts


    Methods
    -------
        fit(X, y)
            Fit the RoughSetSelector model
        transform(X, y)
            Transform the data (only X, y is ignored) using the support_ attribute of the fitted model
        fit_transform(X, y)
            Fit to data, then transform it

    References
    ----------
    [1] Campagner, A., Ciucci, D., Hüllermeier, E. (2021). 
        Rough set-based feature selection for weakly labeled data.
        International Journal of Approximate Reasoning, 136, 150-167.
        https://doi.org/10.1016/j.ijar.2021.06.005.
    '''
    def __init__(self, entropy='oau', method='approximate', epsilon = 0.0,
                 n_iters = 100, discrete = True, metric='minkowski',
                 neighborhood='nearest', n_neighbors=3, radius=1.0, random_state = 0):
        self.entropy = entropy
        self.method = method
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.random_state = random_state
        self.discrete = discrete
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.neighborhood = neighborhood
        self.metric = metric
        
    def fit(self, X, y):
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.reducts_ = []
        self.target = y

        self.entropy_function = np.nan
        if self.entropy == 'oau':
            self.entropy_function = oau_entropy
        elif self.entropy == 'bet':
            self.entropy_function = bet_entropy
        else:
            raise ValueError("%s is not an allowed value for 'entropy' parameter." % self.entropy)
        
        self.target = y
        if y.dtype != np.dtype('O'):
            self.target = [[val] for val in y]
            
        self.data = pd.DataFrame(X)
        self.data['target'] = self.target
        self.classes = np.unique(np.add.reduce(self.data["target"].values))
        self.n_classes = len(self.classes)
        self.attributes = list(self.data.columns[:-1].values)
        
        if self.method == 'approximate':
            self.reducts_ = list(self.__find_approx_reducts__())
            self.support_[self.reducts_] = True
        elif self.method == 'brute':
            self.reducts_ = self.__find_reducts__()
            self.support_[list(self.reducts_[0])] = True
        else:
            raise ValueError("%s is not an allowed value for 'method' parameter." % self.method)
        return self
    
    def transform(self, X, y):
        return X[:, self.support_]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
    def __find_reducts__(self):
        h = np.inf
        reducts = []
        min_len = len(self.attributes)
        for k in range(len(self.attributes),0,-1):
            for i in itertools.combinations(range(len(self.attributes)), k):
                h_temp = 0
                
                if self.discrete:
                    grouped_df = self.data.groupby(list(i))

                    for key, _ in grouped_df:
                        classes = grouped_df.get_group(key)['target']
                        ent = self.entropy_function(classes, self.n_classes)
                        h_temp += len(classes)/self.data.shape[0]*ent
                else:
                    nn = NearestNeighbors(n_neighbors = self.n_neighbors,
                                          radius = self.radius,
                                          metric = self.metric)
                    nn.fit(self.data.loc[:,i])
                    indices = []
                    if self.neighborhood == 'nearest':
                        _ , indices = nn.kneighbors(self.data.loc[:,i])
                    elif self.neighborhood == 'delta':
                        _, indices = nn.radius_neighbors(self.data.loc[:,i])
                    else:
                        raise ValueError("%s is not an allowed value for 'neighborhood' parameter" % self.neighborhood)
                    for ind in indices:
                        x = self.data["target"][ind]
                        h_temp += self.entropy_function(x, self.n_classes)

                if h_temp <= h - np.log2([1 - self.epsilon]):
                    reducts.append(i)
                    h = h_temp
                    if len(i) < min_len:
                        min_len = len(i)
        reducts = [red for red in reducts if len(red) <= min_len]
        return reducts
    
    def __find_approx_reducts__(self):
        total_red = self.attributes
        np.random.seed(self.random_state)
        for iter in range(self.n_iters):
            best_h = 0
            
            if self.discrete:
                grouped_df = self.data.groupby(list(self.attributes))
                for key, _ in grouped_df:
                    classes = grouped_df.get_group(key)['target']
                    ent = self.entropy_function(classes, self.n_classes)
                    best_h += len(classes)/self.data.shape[0]*ent
            else:
                nn = NearestNeighbors(n_neighbors = self.n_neighbors,
                                      radius = self.radius,
                                      metric = self.metric)
                nn.fit(self.data.loc[:,self.attributes])
                indices = []
                if self.neighborhood == 'nearest':
                    _ , indices = nn.kneighbors(self.data.loc[:,self.attributes])
                elif self.neighborhood == 'delta':
                    _, indices = nn.radius_neighbors(self.data.loc[:,self.attributes])
                else:
                    raise ValueError("%s is not an allowed value for 'neighborhood' parameter" % self.neighborhood)
                
                for ind in indices:
                    x = self.data["target"][ind]
                    best_h += self.entropy_function(x, self.n_classes)
                    
            red = self.attributes
            check = True
            while check == True and len(red) > 1:
                check = False
                to_keep = red
                combs = itertools.combinations(red, len(red) - 1)
                combs = np.random.permutation(list(combs))
                for i in combs:
                    h_temp = 0
                    
                    if self.discrete:
                        grouped_df = self.data.groupby(list(i))
                        for key, _ in grouped_df:
                            classes = grouped_df.get_group(key)['target']
                            ent = self.entropy_function(classes, self.n_classes)
                            h_temp += len(classes)/self.data.shape[0]*ent
                    else:
                        nn = NearestNeighbors(n_neighbors = self.n_neighbors,
                                              radius = self.radius,
                                              metric = self.metric)
                        nn.fit(self.data.loc[:,list(i)])
                                          
                        indices = []
                        if self.neighborhood == 'nearest':
                            _ , indices = nn.kneighbors(self.data.loc[:,i])
                        elif self.neighborhood == 'delta':
                            _, indices = nn.radius_neighbors(self.data.loc[:,i])
                            indices = [list(ind) for ind in indices]
                        else:
                            raise ValueError("%s is not an allowed value for 'neighborhood' parameter" % self.neighborhood)
                        
                        for ind in indices:
                            x = self.data["target"][ind]
                            h_temp += self.entropy_function(x, self.n_classes)
                    
                    if h_temp <= best_h - np.log2([1 - self.epsilon]):
                        to_keep = i
                        best_h = h_temp
                if len(to_keep) < len(red):
                    red = to_keep
                    check = True
            if len(red) < len(total_red):
                total_red = red
        return total_red