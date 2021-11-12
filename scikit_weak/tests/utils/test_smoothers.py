import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from ...utils import DiscreteRandomSmoother, DiscreteEstimatorSmoother

@pytest.fixture
def dataset():
    X, y = load_iris(return_X_y=True)
    return (X,y)

def test_simple_random(dataset):
    X, y = dataset[0], dataset[1]
    DiscreteRandomSmoother().fit(X,y)
    DiscreteRandomSmoother().fit_transform(X,y)
    assert True

def test_simple_estimator(dataset):
    X, y = dataset[0], dataset[1]
    est = KNeighborsClassifier()
    DiscreteEstimatorSmoother(estimator=est).fit(X,y)
    DiscreteEstimatorSmoother(estimator=est).fit_transform(X,y)
    assert True
