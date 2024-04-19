import pytest
from fakenet.fakenet import FakeNet
from tests.fakenet.config import ACCEPTANCE_THRESHOLD, DEFAULT_CONFIDENCE, N_TRIES
from tests.fakenet.utils import eval_diamond_miner


@pytest.mark.parametrize("height", range(2, 5))
@pytest.mark.parametrize("length", range(3, 6))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_meshed_networks(height, length, optimal_jump):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.meshed([height for _ in range(length)]),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("n_paths", range(3, 6))
@pytest.mark.parametrize("length", range(3, 6))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_multi_single_paths_networks(n_paths, length, optimal_jump):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.multi_single_paths(n_paths=n_paths, path_length=length),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("length", range(1, 20))
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_single_path_network(length, optimal_jump):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.multi_single_paths(n_paths=1, path_length=length),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("confidence", [60.0, 70.0, 80.0, 90.0])
def test_varying_confidence(confidence):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.multi_single_paths(n_paths=4, path_length=5),
                confidence=confidence,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    lower_threshold = confidence * (ACCEPTANCE_THRESHOLD / DEFAULT_CONFIDENCE)
    # for a confidence of 60
    # lower = 60 * (75 / 95) = 47
    assert lower_threshold < OK / N_TRIES


@pytest.mark.parametrize("confidence", [10.0, 30.0, 50.0])
def test_low_confidence_fails(confidence):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.meshed([4 for _ in range(4)]),
                confidence=confidence,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES < ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("density", [0.5, 0.7, 0.9])
@pytest.mark.parametrize("depth", range(2, 6))
@pytest.mark.parametrize("graph_seed", [0, 1, 2])
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_random_network(density, depth, graph_seed, optimal_jump):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.random_graph(p_edge=density, depth=depth, seed=graph_seed),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
                optimal_jump=optimal_jump,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD
