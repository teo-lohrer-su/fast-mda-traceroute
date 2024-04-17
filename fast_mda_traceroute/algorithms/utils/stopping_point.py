from functools import cache

# from math import comb
import math
from scipy.special import comb, stirling2


# @cache
# def old_p_k_n(k, n, N):
#     if k in P and n in P[k] and N in P[k][n]:
#         return P[k][n][N]
#     mat = [[[0 for _ in range(N + 2)] for _ in range(n + 2)] for _ in range(k + 2)]
#     mat[1][1] = [1 for _ in range(N + 1)]
#     mat[0][0] = [1 for _ in range(N + 1)]
#     for x in range(2, k + 1):
#         for y in range(1, n + 1):
#             for z in range(1, N + 1):
#                 p_found_all = mat[x - 1][y][z]
#                 p_not_find_new = y / z
#                 p_found_all_but_one = mat[x - 1][y - 1][z]
#                 p_find_new = (z - y + 1) / z
#                 mat[x][y][z] = (
#                     p_found_all * p_not_find_new + p_found_all_but_one * p_find_new
#                 )
#                 # cache the value in the global dict P
#                 if x not in P:
#                     P[x] = dict()
#                 if y not in P[x]:
#                     P[x][y] = dict()
#                 P[x][y][z] = mat[x][y][z]
#     return mat[k][n][N]


@cache
def stirling(n, k):
    return stirling2(n, k, exact=True)


@cache
def p_k_n(k, n, N):
    if n > N or k <= 0 or n <= 0 or n > k:
        return 0.0

    numerator = comb(N, n, exact=True) * stirling(k, n) * math.factorial(n)
    denominator = N**k

    return numerator / denominator


N_MAX = 256


@cache
def optimal_N(k, n, likelihood_threshold=0.2):
    if k == n:
        for N in range(n, N_MAX):
            # prob = P[k][n][N]
            prob = p_k_n(k, n, N)
            if prob > likelihood_threshold:
                return N
    if k < n:
        raise ValueError("k must be greater or equal to n")
    N = n - 1
    prev_prob = 0
    for N in range(n - 1, N_MAX):
        prob = p_k_n(k, n, N)
        if prob < prev_prob:
            return N - 1
        prev_prob = prob
    return N


# @cache
# def reach_prob(total_interfaces: int, n_probes: int) -> float:
#     # Hypothesis: there are total_interfaces interfaces in total, and sent n_probes.
#     # what is the probability to reach all interfaces?
#     try:
#         big_sum = sum(
#             comb(total_interfaces, i, exact=False)
#             * (i**n_probes)
#             * (-1) ** (total_interfaces - i - 1)
#             for i in range(total_interfaces)
#         )
#         res = 1 - big_sum / total_interfaces**n_probes
#         return res
#     except OverflowError:
#         return p_k_n(n_probes, total_interfaces, total_interfaces)
# chunk the sum
# big_sum = 0
# interfaces = list(range(total_interfaces))
# chunk_size = 4
# for chunk in (interfaces[i::chunk_size] for i in range(chunk_size)):
#     small_sum = 0
#     for i in chunk:
#         small_sum += (
#             comb(total_interfaces, i)
#             * (i**n_probes)
#             * (-1) ** (total_interfaces - i - 1)
#         )
#     for _ in range(n_probes):
#         small_sum /= total_interfaces
#     big_sum += small_sum
# res = 1 - big_sum
# return res


@cache
def stopping_point(n_interfaces: int, failure_probability: float) -> int:
    # computes the minimal number of probes required to have reach_prob > a threshold
    n_probes = 0

    # while reach_prob(n_interfaces + 1, n_probes) < (1 - failure_probability):
    while p_k_n(n_probes, n_interfaces + 1, n_interfaces + 1) < (
        1 - failure_probability
    ):
        n_probes += 1

    return n_probes


if __name__ == "__main__":
    # Example
    # k, n, N = 2, 2, 3
    # print(f"{k=}, {n=}, {N=}: {f(k, n, N):.2f}")
    # print(f"{k=}, {n=}, {N=}: {g(k, n, N):.2f}")
    for N in range(2, 6):
        print("---------")
        for k in range(0, 10):
            print()
            for n in range(0, min(k, N) + 1):
                print(f"{k=}, {n=}, {N=}: {p_k_n(k, n, N):.5f}")
    # failure_probability = 0.05
    # for n_interfaces in range(1, 21):
    #     print(
    #         f"{n_interfaces}: {stopping_point_prob(n_interfaces, failure_probability)}"
    #     )
