import pytest
import numpy as np
from ...feature_selection import GeneticRoughSetSelector
from ...data_representation import *
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

@pytest.fixture
def dataset():
    X, y = load_iris(return_X_y=True)
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X,y)
    y_fuzzy = clf.predict_proba(X)
    y_fuzzy = y_fuzzy/np.max(y_fuzzy, axis=1)[:, np.newaxis]
    y_soft = np.empty(y_fuzzy.shape[0], dtype=DiscreteFuzzyLabel)
    for i in range(y_fuzzy.shape[0]):
        y_soft[i] = DiscreteFuzzyLabel(y_fuzzy[i], 3)
    return (X, y, y_soft)

def test_simple(dataset):
    rss = GeneticRoughSetSelector(n_iters=10, epsilon=0.1, random_state=0)
    X, y = dataset[0], dataset[2]
    rss.fit(X,y)
    rss.transform(X)
    assert True

def test_random_state(dataset):
    rss = GeneticRoughSetSelector(n_iters=10, epsilon=0.1, random_state=0)
    X, y = dataset[0], dataset[2]
    res1 = rss.fit(X,y).best_features_[0]
    res2 = rss.fit(X,y).best_features_[0]
    assert np.array_equal(res1, res2) 

def test_transform(dataset):
    rss = GeneticRoughSetSelector(n_iters=10, epsilon=0.1, random_state=0)
    X, y = dataset[0], dataset[2]
    res1 = rss.fit_transform(X,y)
    res2 = rss.fit_transform(X,y)
    assert np.array_equal(res1, res2)