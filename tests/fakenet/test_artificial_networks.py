import pytest
from fakenet.fakenet import FakeNet
from tests.fakenet.apriori import (
    apriori_prob,
)
from tests.fakenet.config import (
    DEFAULT_CONFIDENCE,
    N_TRIES,
    DELTA_THRESHOLD,
)
from tests.fakenet.utils import eval_diamond_miner


@pytest.mark.parametrize("height", range(2, 5))
@pytest.mark.parametrize("length", range(3, 6))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_meshed_networks(height, length, optimal_jump):
    """
    Checks that the algorithm succeeds on meshed networks
    with a probability higher than the expected apriori threshold.
    """
    net = FakeNet.meshed([height for _ in range(length)])
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=DEFAULT_CONFIDENCE / 100)
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )

    assert OK / N_TRIES > expected


@pytest.mark.parametrize("n_paths", range(3, 6))
@pytest.mark.parametrize("length", range(3, 6))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_multi_single_paths_networks(n_paths, length, optimal_jump):
    """
    Checks that the algorithm succeeds on a simple network with a single load-balancer.
    """
    net = FakeNet.multi_single_paths(n_paths=n_paths, path_length=length)
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=DEFAULT_CONFIDENCE / 100)

    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > expected


@pytest.mark.parametrize("length", range(1, 20))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_single_path_network(length, optimal_jump):
    """
    Checks that the algorithm succeeds on a simple network with no load-balancer.
    """
    net = FakeNet.multi_single_paths(n_paths=1, path_length=length)
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=DEFAULT_CONFIDENCE / 100)

    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > expected


@pytest.mark.parametrize("confidence", [60.0, 70.0, 80.0, 90.0])
def test_varying_confidence(confidence):
    """
    Checks that the algorithm succeeds on a simple network
    more often than the expected apriori probability.
    """
    net = FakeNet.multi_single_paths(n_paths=4, path_length=5)
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=confidence / 100)

    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=confidence,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > expected


@pytest.mark.parametrize("confidence", [10.0, 30.0, 50.0])
def test_low_confidence_fails(confidence):
    """
    Checks that the algorithm fails when the confidence is too low,
    more often than the expected threshold.
    """
    net = FakeNet.meshed([4 for _ in range(4)])
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=0.95)

    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=confidence,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES < expected


@pytest.mark.parametrize("density", [0.5, 0.7, 0.9])
@pytest.mark.parametrize("depth", range(2, 6))
@pytest.mark.parametrize("graph_seed", range(5))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_random_network(density, depth, graph_seed, optimal_jump):
    """
    Checks that the algorithm succeeds on random networks
    with a probability higher than the expected apriori threshold.
    """
    net = FakeNet.random_graph(p_edge=density, depth=depth, seed=graph_seed)
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=DEFAULT_CONFIDENCE / 100)

    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > expected
