import pytest
from fakenet.fakenet import FakeNet
from tests.fakenet.apriori import apriori_prob
from tests.fakenet.config import (
    DEFAULT_CONFIDENCE,
    DELTA_THRESHOLD,
    N_TRIES,
    SAMPLE_FILES,
)
from tests.fakenet.utils import eval_diamond_miner


@pytest.mark.parametrize("filepath", SAMPLE_FILES)
@pytest.mark.parametrize("optimal_jump", [False, True])
def test_real_networks(filepath, optimal_jump):
    net = FakeNet.from_file(filepath)
    expected = DELTA_THRESHOLD * apriori_prob(net, confidence=DEFAULT_CONFIDENCE / 100)
    if len(net.nodes()) > 80 or len(net.edges()) > 100:
        assert False, f"Skipping {filepath}, got {len(net.nodes())} nodes and {len(net.edges())} edges"
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
