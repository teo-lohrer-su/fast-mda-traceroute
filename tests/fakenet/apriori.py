import logging
from scipy.special import stirling2
from collections import defaultdict
import concurrent.futures
from functools import cache
import math
import random

from tqdm import tqdm
from fast_mda_traceroute.logger import logger
from fast_mda_traceroute.algorithms.utils.stopping_point import (
    estimate_total_interfaces,
    reach_prob,
    stopping_point,
)
from tests.fakenet.config import DEFAULT_CONFIDENCE, N_TRIES
from tests.fakenet.utils import eval_diamond_miner


@cache
def apriori_single_node(degree, confidence=DEFAULT_CONFIDENCE / 100):
    tries = 100_000
    ok = 0
    for _ in range(tries):
        estimation = 1
        total_probes = 0
        discovered_interfaces = set()

        while True:
            stop = stopping_point(estimation, failure_probability=1 - confidence)
            send_probes = stop - total_probes
            if send_probes <= 0:
                break
            total_probes = stop

            round_interfaces = set(random.choices(range(degree), k=send_probes))
            discovered_interfaces.update(round_interfaces)
            estimation = len(discovered_interfaces)
        if len(discovered_interfaces) == degree:
            ok += 1
    return ok / tries


@cache
def step(degree: int, new_probes: int, found: int) -> dict[int, float]:
    # We have found `found` interfaces.
    # During this round, we are sending `new_probes` additional probes.
    # We output the probability vector for the number of new interfaces found.

    res = defaultdict(float)
    for new_interfaces in range(0, min(new_probes, degree - found) + 1):
        for n_relapse in range(new_probes - new_interfaces + 1):
            ways_to_pick_relapsing_probes = math.comb(new_probes, n_relapse)
            ways_to_distribute_relapsing_probes = found**n_relapse

            remaining_probes = new_probes - n_relapse
            ways_to_distribute_remaining_probes = stirling2(
                remaining_probes, new_interfaces
            )

            ways_to_pick_new_interfaces = math.comb(degree - found, new_interfaces)
            ways_to_distribute_subsets = math.factorial(new_interfaces)
            x = (
                ways_to_pick_relapsing_probes
                * ways_to_distribute_relapsing_probes
                * ways_to_distribute_remaining_probes
                * ways_to_pick_new_interfaces
                * ways_to_distribute_subsets
                / degree**new_probes
            )
            res[found + new_interfaces] += x

    return res


def rec_step(degree, estimation, found, sent, confidence=DEFAULT_CONFIDENCE / 100):
    stop = stopping_point(estimation, failure_probability=1 - confidence)
    if sent >= stop:
        reach = reach_prob(found, stop)
        return {found: reach}

    s = step(degree, stop - sent, found)
    res = defaultdict(float)
    for new_found, prob in s.items():
        # we have found `new_found` new interfaces with probability `prob`
        # we update the estimation
        new_estimation = new_found + 1
        for k, v in rec_step(
            degree, new_estimation, new_found, stop, confidence
        ).items():
            # given that we have found `new_found` new interfaces,
            # we have a probability `v` to find `k` interfaces at the next step
            res[k] += prob * v
    return res


@cache
def apriori_direct_single_node_rec(degree, confidence=DEFAULT_CONFIDENCE / 100):
    run = rec_step(degree, 1, 0, 0, confidence=confidence)
    return run[degree]


def apriori_direct(net, confidence=DEFAULT_CONFIDENCE / 100):
    expected = 1.0

    for node in net.nodes:
        degree = net.out_degree(node)
        if degree < 1:
            continue
        direct = apriori_direct_single_node_rec(degree, confidence=confidence)
        expected *= direct
    return expected


@cache
def apriori_single_node_opti(degree, confidence=DEFAULT_CONFIDENCE / 100):
    tries = 100_000
    ok = 0
    for _ in range(tries):
        estimation = 1
        total_probes = 0
        discovered_interfaces = set()

        while True:
            stop = stopping_point(estimation, failure_probability=1 - confidence)
            send_probes = stop - total_probes
            if send_probes <= 0:
                break
            total_probes = stop

            round_interfaces = set(random.choices(range(degree), k=send_probes))
            discovered_interfaces.update(round_interfaces)
            estimation = estimate_total_interfaces(
                k=total_probes, n=len(discovered_interfaces)
            )
        if len(discovered_interfaces) == degree:
            ok += 1
    return ok / tries


def apriori_prob(net, confidence=DEFAULT_CONFIDENCE / 100):
    expected = 1.0

    for node in net.nodes:
        # fetch degree of node
        degree = net.out_degree(node)
        if degree < 1:
            continue
        expected *= apriori_single_node(degree, confidence=confidence)
    return expected


def apriori_oracle(net, confidence=DEFAULT_CONFIDENCE / 100):
    expected = 1.0

    for node in net.nodes:
        # fetch degree of node
        degree = net.out_degree(node)
        if degree < 1:
            continue
        stop = stopping_point(degree, failure_probability=1 - confidence)
        p = reach_prob(degree, stop)
        expected *= p
    return expected


def worker(seed, net, confidence, optimal_jump):
    logger.setLevel(logging.ERROR)
    return eval_diamond_miner(
        net=net, confidence=confidence * 100, seed=seed, optimal_jump=optimal_jump
    )


def simple_trial(
    net,
    optimal_jump,
    n_tries=N_TRIES,
    confidence=DEFAULT_CONFIDENCE / 100,
    magic: int = 0,
    n_workers=4,
):
    seeds = range(magic, magic + n_tries)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    worker,
                    seeds,
                    [net] * n_tries,
                    [confidence] * n_tries,
                    [optimal_jump] * n_tries,
                ),
                total=n_tries,
            )
        )

    OK = sum(results)
    return OK / n_tries
