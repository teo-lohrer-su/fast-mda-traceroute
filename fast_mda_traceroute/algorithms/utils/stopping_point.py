from functools import cache
from math import comb


@cache
def reach_prob(total_interfaces: int, n_probes: int) -> float:
    # Hypothesis: there are total_interfaces interfaces in total, and sent n_probes.
    # what is the probability to reach all interfaces?
    big_sum = sum(
        comb(total_interfaces, i) * (i**n_probes) * (-1) ** (total_interfaces - i - 1)
        for i in range(total_interfaces)
    )
    return 1 - big_sum / total_interfaces**n_probes


@cache
def stopping_point(n_interfaces: int, failure_probability: float) -> int:
    # computes the minimal number of probes required to have reach_prob > a threshold
    n_probes = 0

    while reach_prob(n_interfaces + 1, n_probes) < (1 - failure_probability):
        n_probes += 1

    return n_probes


@cache
def stopping_point_prob(n_interfaces: int, failure_probability: float) -> int:
    # computes the minimal number of probes required to have reach_prob > a threshold
    n_probes = 0

    while reach_prob(n_interfaces + 1, n_probes) < (1 - failure_probability):
        n_probes += 1

    return n_probes, reach_prob(n_interfaces + 1, n_probes)


if __name__ == "__main__":
    # Example
    failure_probability = 0.05
    for n_interfaces in range(1, 21):
        print(
            f"{n_interfaces}: {stopping_point_prob(n_interfaces, failure_probability)}"
        )