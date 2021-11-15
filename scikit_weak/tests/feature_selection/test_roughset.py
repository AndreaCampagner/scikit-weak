import pytest
import numpy as np
from sklearn.datasets import load_iris
from ...feature_selection import RoughSetSelector
from sklearn.neighbors import KNeighborsClassifier
from ...data_representation import *

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
    rss = RoughSetSelector(n_iters=10, epsilon=0.1, random_state=0)
    X, y = dataset[0], dataset[2]
    rss.fit(X,y)
    rss.transform(X)
    assert True