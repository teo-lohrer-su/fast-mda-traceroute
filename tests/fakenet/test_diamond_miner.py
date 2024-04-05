import glob
import os

import networkx as nx
import pytest
from fakenet.fakenet import FakeNet, FakeProber, graph_from_links
from pycaracal import Probe, Reply

from fast_mda_traceroute.algorithms.diamond_miner import DiamondMiner
from fast_mda_traceroute.typing import Protocol

ACCEPTANCE_THRESHOLD = 0.75
DEFAULT_CONFIDENCE = 95.0
N_TRIES = 100
SAMPLE_NET_DATA = "tests/fakenet/data"
SAMPLE_FILES = [
    f for f in glob.glob(os.path.join(SAMPLE_NET_DATA, "*")) if os.path.isfile(f)
]


def eval_diamond_miner(net: FakeNet, confidence: float = 95.0, seed: int = 0) -> bool:
    prober = FakeProber(net, seed=seed)

    protocol = Protocol.ICMP
    src_port = 12345
    dst_port = 33434
    max_round = 100
    min_ttl = 1
    # fetch the distance between the start and end nodes
    # NOTE: we cannot use the shortest path because the network may be unbalanced
    max_ttl = nx.dag_longest_path_length(net) + 1

    src_addr = net.start
    dst_addr = net.end

    alg = DiamondMiner(
        dst_addr=dst_addr,
        min_ttl=min_ttl,
        max_ttl=max_ttl,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol.value,
        confidence=confidence,
        max_round=max_round,
    )

    last_replies: list[Reply] = []
    rnd = 0
    while True:
        rnd += 1
        probes = [Probe(*x) for x in alg.next_round(last_replies)]
        if not probes:
            break
        last_replies = prober.probe(probes)
    # transform links_by_ttl to a dict of lists
    links = {
        ttl: list((x, y) for _, x, y in alg.links_by_ttl[ttl] if x and y)
        for ttl in range(1, max_ttl)
    }
    # filter links to only show the ttl that have links
    links = {k: v for k, v in links.items() if v}

    # create a graph from the links
    nx_graph = graph_from_links(src_addr, links)

    # check if nx_graph is equal to net
    return nx.is_isomorphic(nx_graph, net)


@pytest.mark.parametrize("height", range(2, 5))
@pytest.mark.parametrize("length", range(3, 6))
def test_meshed_networks(height, length):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.meshed([height for _ in range(length)]),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("n_paths", range(3, 6))
@pytest.mark.parametrize("length", range(3, 6))
def test_multi_single_paths_networks(n_paths, length):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.multi_single_paths(n_paths=n_paths, path_length=length),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("length", range(1, 20))
def test_single_path_network(length):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.multi_single_paths(n_paths=1, path_length=length),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
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
def test_random_network(density, depth, graph_seed):
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=FakeNet.random_graph(p_edge=density, depth=depth, seed=graph_seed),
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD


@pytest.mark.parametrize("filepath", SAMPLE_FILES)
def test_on_data_files(filepath):
    net = FakeNet.from_file(filepath)
    if len(net.nodes()) > 50 or len(net.edges()) > 80:
        assert (
            False
        ), f"Skipping {filepath}, got {len(net.nodes())} nodes and {len(net.edges())} edges"
    OK = sum(
        (
            1
            if eval_diamond_miner(
                net=net,
                confidence=DEFAULT_CONFIDENCE,
                seed=seed,
            )
            else 0
        )
        for seed in range(N_TRIES)
    )
    assert OK / N_TRIES > ACCEPTANCE_THRESHOLD
