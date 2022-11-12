
from abc import ABC, abstractmethod
import numpy as np

class GenericWeakLabel(ABC):
    '''
      A generic class to represent weak labels or weak data
    '''
    @abstractmethod
    def sample_value(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def basic_type(self):
        pass


class FuzzyLabel(ABC):
    '''
    A generic trait to represent fuzzy labels or fuzzy data
    '''
    @abstractmethod
    def __getitem__(self, val):
        pass

    @abstractmethod
    def get_cut(self, alpha):
        pass


class ContinuousWeakLabel(GenericWeakLabel):
    '''
    A generic trait to represent continuous weak labels or data
    '''
    def basic_type(self):
        return np.float32

class GaussianFuzzyLabel(ContinuousWeakLabel, FuzzyLabel):
    '''
    A class to represent fuzzy labels or data based on Gaussian fuzzy numbers
    '''
    def __init__(self, mean, std):
        if std < 0:
            raise ValueError("Std cannot be lower than 0")
        self.mean = mean
        self.std = std

    def sample_value(self):
        alpha = np.random.random()
        interval = self.get_cut(alpha)
        return interval.sample_value()

    def __eq__(self, other):
        if isinstance(other, GaussianFuzzyLabel):
            return (self.mean == other.mean) and (self.std == other.std)
        else:
            return False

    def __getitem__(self, val):
        return np.exp(-(self.mean - val)**2/(2*self.std**2))

    def get_cut(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be between 0 and 1")
        else:
            return IntervalLabel( self.mean - self.std*np.sqrt(-2*np.log(alpha)), self.mean + self.std*np.sqrt(-2*np.log(alpha)) )

    def __str__(self):
        return "Gaussian[%f, %f]" % (self.mean, self.std)


class IntervalFuzzyLabel(ContinuousWeakLabel,FuzzyLabel):
    '''
    A generic trait to represent interval fuzzy labels or data
    '''
    pass

class IntervalLabel(IntervalFuzzyLabel):
    '''
    A class to represent interval-valued labels or data
    '''
    def __init__(self, lower, upper):
        if lower > upper:
            raise ValueError("Lower bound cannot be greater than upper bound")
        self.lower = lower
        self.upper = upper

    def sample_value(self):
        return np.random.random()*(self.upper - self.lower) + self.lower

    def __eq__(self, other):
        if isinstance(other, IntervalLabel):
            return (self.lower == other.lower) and (self.upper == other.upper)
        else:
            return False

    def __getitem__(self, val):
        return 1 if (val >= self.lower and val <= self.upper) else 0

    def get_cut(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be between 0 and 1")
        else:
            return self

    def __str__(self):
        return "[%f, %f]" % (self.lower, self.upper)


class DiscreteWeakLabel(GenericWeakLabel):
    '''
    A generic trait to represent discrete weak labels or data
    '''
    def basic_type(self):
        return np.int32
    
    @abstractmethod
    def to_probs(self):
        pass

class DiscreteFuzzyLabel(DiscreteWeakLabel, FuzzyLabel):
    '''
    A class to represent discrete fuzzy labels or data
    '''
    def __init__(self, classes, n_classes):
        self.n_classes = n_classes
        if isinstance(classes, np.ndarray) and classes.ndim == 1:
            if classes.shape[0] > self.n_classes:
                raise ValueError("Unexpected number of classes")
            self.classes = classes
        elif isinstance(classes, dict):
            self.classes = self.__to_array_encoding(classes)
        elif isinstance(classes, list):
            self.classes = np.zeros(self.n_classes)
            for i in classes:
                self.classes[i] = 1
        else:
            raise ValueError("Format not supported")

    def to_probs(self):
        probs= np.zeros(self.n_classes)
        values = np.unique(self.classes)
        sorted_values = np.sort(values)[::-1]
        if sorted_values[-1] != 0.0:
            sorted_values = np. append(sorted_values, 0.0)
        for j in range(len(sorted_values) - 1):
            val = sorted_values[j]
            idx = np.where(self.classes >= val)
            assign = (val - sorted_values[j+1])/len(idx[0])
            probs[idx] += assign
        return probs

    def sample_value(self):
        probs= self.to_probs()
        return np.random.choice(range(self.n_classes), p=probs)

    def __to_array_encoding(self, classes):
        out = np.zeros(self.n_classes)
        for i in classes:
          if out[i] > 1:
            raise ValueError("Values greater than 1 are not allowed")
          out[i] = classes[i]
        return out

    def __getitem__(self, val):
        return self.classes[val]

    def get_cut(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be between 0 and 1")
        return DiscreteSetLabel((self.classes >= alpha) * 1.0, self.n_classes)

    def __eq__(self, other):
        if isinstance(other, DiscreteWeakLabel):
            return np.array_equal(self.classes, other.classes)
        else:
            return False

    def __str__(self):
        return str(self.classes)

    def is_agnostic(self):
        return np.all(np.equal(self.classes, np.ones_like(self.classes)))


class DiscreteSetLabel(DiscreteFuzzyLabel):
    '''
    A class to represent discrete set-valued labels or data
    '''
    def __init__(self, classes, n_classes):
      super().__init__(classes, n_classes)
      for i in self.classes:
        if i not in [0.0, 1.0]:
          raise ValueError("Only values equal to 1 are allowed")
    
    def sample_value(self):
        return np.random.choice(range(self.n_classes), p=self.classes/np.sum(self.classes))

class RandomLabel(ContinuousWeakLabel):
    '''
    A class to represent continuous, probabilistic distribution-valued labels or data
    '''
    def __init__(self, distrib):
        self.distrib = distrib

    def sample_value(self):
        return self.distrib.rvs()

    def __eq__(self, other):
        if isinstance(other, RandomLabel):
            return (self.distrib == other.distrib)
        else:
            return False

    def __getitem__(self, val):
        return self.distrib.pdf(val)

    def __str__(self):
        return str(self.distrib)