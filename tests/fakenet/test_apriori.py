import pytest
from tests.fakenet.apriori import apriori_prob
from fakenet.fakenet import FakeNet


def assert_sorted_probs(probs):
    assert all([0 <= prob <= 1 for prob in probs])
    assert sorted(probs) == probs


@pytest.mark.parametrize("n_paths", range(3, 6))
def test_apriori_multi_path(n_paths: int):
    """
    Checks the apriori probability for a network with a single load-balancer.
    """
    confidences = [0.80, 0.90, 0.95, 0.99]
    net = FakeNet.multi_single_paths(n_paths=n_paths, path_length=3)
    expected_probs = [apriori_prob(net, confidence=conf) for conf in confidences]
    assert_sorted_probs(expected_probs)
    assert len(set(expected_probs)) == len(expected_probs)
    for prob, confidence in zip(expected_probs, confidences):
        a, b = confidence * 0.95, confidence * 1.05
        # for confidence=0.95, we expect the probability to be between 0.9025 and 0.9975
        assert a <= prob <= b


@pytest.mark.parametrize("height", range(3, 6))
@pytest.mark.parametrize("length", range(3, 6))
def test_apriori_meshed(height: int, length: int):
    """
    Checks the apriori probability for a meshed network.
    """
    confidences = [0.80, 0.90, 0.95, 0.99]
    net = FakeNet.meshed([height for _ in range(length)])
    expected_probs = [apriori_prob(net, confidence=conf) for conf in confidences]
    assert_sorted_probs(expected_probs)
    assert len(set(expected_probs)) == len(expected_probs)
    n_nodes = height * (length - 1) + 1
    for prob, confidence in zip(expected_probs, confidences):
        a, b = confidence * 0.95, confidence * 1.05
        # for confidence=0.95, we expect the probability to be between 0.9025 and 0.9975
        # each bound is raised to the power of the number of nodes
        assert a**n_nodes <= prob <= b**n_nodes


@pytest.mark.parametrize("seed", range(10))
def test_apriori_random(seed: int):
    """
    Checks the apriori probability for a random network.
    """
    confidences = [0.80, 0.90, 0.95, 0.99]
    net = FakeNet.random_graph(p_edge=0.7, depth=8, seed=seed)
    expected_probs = [apriori_prob(net, confidence=conf) for conf in confidences]

    assert all([0 <= prob <= 1 for prob in expected_probs])
    assert sorted(expected_probs) == expected_probs
