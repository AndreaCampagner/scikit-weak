
from abc import ABC, abstractmethod
import numpy as np

class GenericWeakLabel(ABC):
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
    @abstractmethod
    def __getitem__(self, val):
        pass

    @abstractmethod
    def get_cut(self, alpha):
        pass


class ContinuousWeakLabel(GenericWeakLabel):
    def basic_type(self):
        return np.float32


class IntervalFuzzyLabel(ContinuousWeakLabel,FuzzyLabel):
    pass

class IntervalLabel(IntervalFuzzyLabel):
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
    def basic_type(self):
        return np.int32

class DiscreteFuzzyLabel(DiscreteWeakLabel, FuzzyLabel):
    def __init__(self, classes, n_classes):
        self.__n_classes = n_classes
        if isinstance(classes, np.ndarray) and classes.ndim == 1:
            if classes.shape[0] > self.__n_classes:
                raise ValueError("Unexpected number of classes")
            self.classes = classes
        elif isinstance(classes, dict):
            self.classes = self.__to_array_encoding(classes)
        elif isinstance(classes, list):
            self.classes = np.zeros(self.__n_classes)
            for i in classes:
                self.classes[i] = 1
        else:
            raise ValueError("Format not supported")

    def sample_value(self):
        probs= np.zeros(self.__n_classes)
        values = np.unique(self.classes)
        sorted_values = np.sort(values)[::-1]
        if sorted_values[-1] != 0.0:
            sorted_values = np. append(sorted_values, 0.0)
        for j in range(len(sorted_values) - 1):
            val = sorted_values[j]
            idx = np.where(self.classes >= val)
            assign = (val - sorted_values[j+1])/len(idx[0])
            probs[idx] += assign

        return np.random.choice(range(self.__n_classes), p=probs)

    def __to_array_encoding(self, classes):
        out = np.zeros(self.__n_classes)
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
        return DiscreteSetLabel((self.classes >= alpha) * 1.0, self.__n_classes)

    def __eq__(self, other):
        if isinstance(other, DiscreteWeakLabel):
            return np.array_equal(self.classes, other.classes)
        else:
            return False

    def __str__(self):
        return str(self.classes)


class DiscreteSetLabel(DiscreteFuzzyLabel):

    def __init__(self, classes, n_classes):
      super().__init__(classes, n_classes)
      for i in self.classes:
        if i not in [0.0, 1.0]:
          raise ValueError("Only values equal to 1 are allowed")
    
    def sample_value(self):
        return np.random.choice(range(self.n_classes), p=self.classes/np.sum(self.classes))