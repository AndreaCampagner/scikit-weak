import pytest
from sklearn.datasets import load_iris
from sklearn import clone
from sklearn.model_selection import GridSearchCV

from ...classification._relaxation import LabelRelaxationNNClassifier


@pytest.fixture
def dataset():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_label_relaxation(dataset):
    """
    Simple label relaxation test.
    """

    X, y_true = dataset[0], dataset[1]
    clf = LabelRelaxationNNClassifier(lr_alpha=0.1, hidden_layer_sizes=(128,), activation="relu",
                                      l2_penalty=1e-8, learning_rate=1e-1, momentum=0.0, epochs=100,
                                      batch_size=128)
    clf.fit(X, y_true)
    _ = clf.predict(X)
    _ = clf.predict_proba(X)
    assert True


def test_clone():
    """
    As it is required to perform model selection routines from scikit-learn (e.g., RandomizedSearchCV).
    """
    clf = LabelRelaxationNNClassifier(lr_alpha=0.1, hidden_layer_sizes=(128,), activation="relu",
                                      l2_penalty=1e-8, learning_rate=1e-1, momentum=0.0, epochs=100,
                                      batch_size=128)
    _ = clone(clf)
    assert True


def test_cv(dataset):
    X, y_true = dataset[0], dataset[1]

    # Multi-class
    lr_int = LabelRelaxationNNClassifier(n_classes=3)
    lr_grid = {'lr_alpha': [0.1], 'learning_rate': [1e-1], 'momentum': [0.0, 0.9]}
    lr = GridSearchCV(estimator=lr_int, param_grid=lr_grid, cv=3, n_jobs=1, scoring='neg_log_loss', verbose=1)
    lr.fit(X, y_true)

    # Binary
    mask = y_true <= 1
    X = X[mask]
    y_true = y_true[mask]

    lr_int = LabelRelaxationNNClassifier(n_classes=2)
    lr_grid = {'lr_alpha': [0.1, 0.25], 'learning_rate': [1e-1], 'momentum': [0.0, 0.9]}
    lr = GridSearchCV(estimator=lr_int, param_grid=lr_grid, cv=3, n_jobs=1, scoring='neg_log_loss', verbose=1)
    lr.fit(X, y_true)

    assert True
