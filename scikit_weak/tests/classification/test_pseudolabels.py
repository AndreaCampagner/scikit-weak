import pytest
import numpy as np
from ...classification import PseudoLabelsClassifier
from sklearn.datasets import load_iris

from ...classification._pseudolabels import CSSLClassifier
from ...data_representation import *

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

def test_grm_linear(dataset):
    X, y_true, y = dataset[0], dataset[1], dataset[2]
    clf = PseudoLabelsClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert True

def test_cssl(dataset):
    X, y_true, y = dataset[0], dataset[1], dataset[2]

    # Make some instances agnostic
    for idx, label in enumerate(y):
        if np.random.random() < 0.3:
            label.classes = np.ones_like(label.classes)

    clf = CSSLClassifier(n_classes=3)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert True