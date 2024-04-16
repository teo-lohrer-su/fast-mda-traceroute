from functools import cache
from math import comb
import json
import os


class PreComputedP:
    mat: list[list[list[float]]]
    n_max: int

    def __init__(self, k_max: int, n_max: int):
        self.mat = [
            [[0 for _ in range(n_max)] for _ in range(n_max)] for _ in range(k_max)
        ]
        self.mat[1][1] = [1 for _ in range(n_max)]
        self.mat[0][0] = [1 for _ in range(n_max)]
        self.complete(k_max)

    @property
    def n_max(self):
        return len(self.mat[0][0])

    @property
    def k_max(self):
        return len(self.mat)

    def complete(self, k_max):
        for k in range(len(self.mat), k_max):
            for n in range(1, self.n_max):
                for N in range(1, self.n_max):
                    p_found_all = self.mat[k - 1][n][N]
                    p_not_find_new = n / N
                    p_found_all_but_one = self.mat[k - 1][n - 1][N]
                    p_find_new = (N - n + 1) / N
                    self.mat[k][n][N] = (
                        p_found_all * p_not_find_new + p_found_all_but_one * p_find_new
                    )

    @cache
    def __getitem__(self, key):
        if key >= len(self.mat):
            self.complete(key)
        return self.mat[key]


N_MAX = 256

P_FILENAME = "p_k1000_n150.json"

if os.path.exists(P_FILENAME):
    with open(P_FILENAME, "r") as f:
        P = json.load(f)
else:
    P = PreComputedP(200, N_MAX)


@cache
def optimal_N(k, n, likelihood_threshold=0.2):
    if k == n:
        for N in range(n, N_MAX):
            prob = P[k][n][N]
            if prob > likelihood_threshold:
                return N
    if k < n:
        raise ValueError("k must be greater or equal to n")
    N = n - 1
    prev_prob = 0
    for N in range(n - 1, N_MAX):
        prob = P[k][n][N]
        if prob < prev_prob:
            return N - 1
        prev_prob = prob
    return N


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
