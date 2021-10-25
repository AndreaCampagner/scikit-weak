import pytest
import numpy as np
from ...data_representation import *


@pytest.fixture
def discrete_fuzzy_label():
    '''Returns a fuzzy label with 5 classes'''
    return DiscreteFuzzyLabel(np.array([0.0, 0.6, 1.0, 1.0, 0.8]), 5)

@pytest.fixture
def discrete_set_label():
    '''Returns a superset label with 5 classes'''
    return DiscreteSetLabel(np.array([0.0, 0.0, 1.0, 1.0, 1.0]), 5)

@pytest.fixture
def interval_label():
    '''Returns a default interval label'''
    return IntervalLabel(0, 3.5)


###Discrete Labels
def test_fuzzy_label_from_dict(discrete_fuzzy_label):
    assert DiscreteFuzzyLabel({1: 0.6, 2: 1.0, 3: 1.0, 4: 0.8}, 5) == discrete_fuzzy_label

def test_set_label_from_list(discrete_set_label):
    assert DiscreteFuzzyLabel([2,3,4], 5) == discrete_set_label

def test_alpha_cut_discrete(discrete_fuzzy_label, discrete_set_label):
    assert discrete_fuzzy_label.get_cut(0.7) == discrete_set_label

def test_membership_discrete(discrete_fuzzy_label):
    assert discrete_fuzzy_label[1] == 0.6

def test_sample_value_discrete(discrete_fuzzy_label):
    check = True
    for i in range(100):
        check = (check) | (discrete_fuzzy_label.sample_value != 0)
    assert check != False


### Interval Labels

def test_membership_interval_in(interval_label):
    assert interval_label[2.2] == 1.0

def test_membership_interval_border(interval_label):
    assert interval_label[3.5] == 1.0

def test_membership_interval_out(interval_label):
    assert interval_label[-1] == 0.0

def test_alpha_cut_interval(interval_label):
    assert interval_label == interval_label.get_cut(0.5)

def test_sample_value_interval(interval_label):
    check = True
    for i in range(100):
        check = (check) or ((interval_label.sample_value() >= interval_label.lower) and (interval_label.sample_value() <= interval_label.upper))
    assert check != False
