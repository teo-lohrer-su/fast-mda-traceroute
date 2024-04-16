from functools import cache
from math import comb


def compute_p(k_max, n_max, N_max):
    res = [[[0 for _ in range(N_max)] for _ in range(n_max)] for _ in range(k_max)]
    res[1][1] = [1 for _ in range(N_max)]
    res[0][0] = [1 for _ in range(N_max)]
    for k in range(2, k_max):
        for n in range(1, n_max):
            for N in range(1, N_max):
                p_found_all = res[k - 1][n][N]
                p_not_find_new = n / N
                p_found_all_but_one = res[k - 1][n - 1][N]
                p_find_new = (N - n + 1) / N
                res[k][n][N] = (
                    p_found_all * p_not_find_new + p_found_all_but_one * p_find_new
                )
    return res


K_MAX = 1000
N_MAX = 150
P = compute_p(K_MAX, N_MAX, N_MAX)


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
