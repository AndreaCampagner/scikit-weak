import pytest
from sklearn.datasets import load_iris
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
