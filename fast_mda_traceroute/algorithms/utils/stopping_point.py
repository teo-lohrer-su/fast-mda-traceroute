from datetime import datetime
from functools import cache

import math
from scipy.special import comb, stirling2

N_MAX = 120


@cache
def p_k_n(k, n, N):
    if n > N or k <= 0 or n <= 0 or n > k:
        return 0.0

    # We choose n interfaces among N
    # We then have to pick a distribution of the k probes in n non-empty sets
    # We have n! to match the n interfaces with the n sets
    # There is a total N^k ways to distribute the k probes among the N interfaces

    # stirling = sum(
    #     (-1) ** (n - i) * int(comb(n, i, exact=False)) * i**k for i in range(n + 1)
    # )

    stirling = stirling2(k, n, exact=False)

    if binom := comb(N, n, exact=False) == float("inf"):
        return 1.0

    return int(stirling * binom) / N**k

    # t0 = datetime.now()

    # if stirling := stirling2(k, n, exact=False) == float("inf"):
    #     return 1.0

    # denominator = N**k

    # stirling = int(stirling) / denominator

    # time = datetime.now() - t0
    # if time.total_seconds() > 1:
    #     print(f"{k=}, {n=}, {N=}: stirling took {time.total_seconds()}")

    # if factorial := math.factorial(n) == float("inf"):
    #     return 1.0

    # numerator = binom * stirling * factorial

    # if numerator == float("inf"):
    #     return 1.0

    # return int(numerator)  # / denominator


@cache
def optimal_N(k, n, likelihood_threshold=0.4):
    if k < n:
        raise ValueError("k must be greater or equal to n")

    if k == n:
        for N in range(n, N_MAX):
            prob = p_k_n(k, n, N)
            if prob > likelihood_threshold:
                return N
        else:
            return n

    prev_prob = 0
    for N in range(n - 1, N_MAX):
        prob = p_k_n(k, n, N)
        if prob > likelihood_threshold:
            return N
        if prob < prev_prob:
            return N - 1
        prev_prob = prob
    else:
        return n


@cache
def reach_prob(total_interfaces: int, n_probes: int) -> float:
    # Hypothesis: there are total_interfaces interfaces in total, and sent n_probes.
    # what is the probability to reach all interfaces?
    try:
        big_sum = sum(
            comb(total_interfaces, i, exact=True)
            * (i**n_probes)
            * (-1) ** (total_interfaces - i - 1)
            for i in range(total_interfaces)
        )
        res = 1 - big_sum / total_interfaces**n_probes
        return res
    except OverflowError:
        return p_k_n(n_probes, total_interfaces, total_interfaces)


@cache
def stopping_point(n_interfaces: int, failure_probability: float) -> int:
    # computes the minimal number of probes required to have reach_prob > a threshold
    n_probes = 0
    growth_factor = 3
    upper_bound = 6
    lower_bound = 1

    while reach_prob(n_interfaces + 1, upper_bound) < (1 - failure_probability):
        lower_bound = upper_bound
        upper_bound = int(upper_bound * growth_factor)

    # dichotomy search for the lowest n_probes that satisfies the condition

    while upper_bound - lower_bound > 1:
        n_probes = (upper_bound + lower_bound) // 2
        if reach_prob(n_interfaces + 1, n_probes) > (1 - failure_probability):
            upper_bound = n_probes
        else:
            lower_bound = n_probes
    return upper_bound


# @cache
# def new_stopping_point(n_interfaces: int, failure_probability: float) -> int:
#     # computes the minimal number of probes required to have reach_prob > a threshold
#     n_probes = 0

#     while p_k_n(k=n_probes, n=n_interfaces + 1, N=n_interfaces + 1) < (
#         1 - failure_probability
#     ):
#         n_probes += 1

#     return n_probes


if __name__ == "__main__":
    # Example
    # k, n, N = 2, 2, 3
    # print(f"{k=}, {n=}, {N=}: {f(k, n, N):.2f}")
    # print(f"{k=}, {n=}, {N=}: {g(k, n, N):.2f}")
    # for N in range(2, 6):
    #     print("---------")
    #     for k in range(0, 10):
    #         print()
    #         for n in range(0, min(k, N) + 1):
    #             print(f"{k=}, {n=}, {N=}")
    #             print(f" p_k_n : {p_k_n(k, n, N):.5f}")
    #             print(f" reach_prob : {reach_prob(N, n):.5f}")
    # p_k_n(20000, 300, 301)
    # for N in range(2, 6):
    #     print("---------")
    #     for k in range(0, 10):
    #         print(f"{k=}, {N=}")
    #         print(f" p_k_n : {p_k_n(k, N, N):.5f}")
    #         print(f" p_k_n+1 : {p_k_n(k, N+1, N+1):.5f}")
    #         print(f" reach_prob : {reach_prob(N, k):.5f}")

    for N in range(1, 20):
        print(f"{stopping_point(N, 0.05)}")
    #     print(f"{new_stopping_point(N, 0.05)}")
    #     print("---------")

    # for n in range(1, N + 1):
    #     for k in range(n, 10):
    #         print(f"{k=} {n=} -> {optimal_N(k, n)=}")

    # failure_probability = 0.05
    # for n_interfaces in range(1, 21):
    #     print(
    #         f"{n_interfaces}: {stopping_point_prob(n_interfaces, failure_probability)}"
    #     )
