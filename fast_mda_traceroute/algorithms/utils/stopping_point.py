from functools import cache
import math

from scipy.special import comb, stirling2

MAX_INTERFACES = 1024


def event_prob(n_probes, observed_interfaces, total_interfaces):
    # Computes the probability of observing exactly observed_interfaces interfaces
    # after sending n_probes probes to total_interfaces interfaces
    if (
        observed_interfaces > total_interfaces
        or n_probes <= 0
        or observed_interfaces <= 0
        or observed_interfaces > n_probes
    ):
        return 0.0

    # We choose k interfaces among K
    # We then have to pick a distribution of the n probes in k non-empty sets
    # We have k! to match the k interfaces with the k sets
    # There is a total K^n ways to distribute the n probes among the K interfaces

    stirling = stirling2(n_probes, observed_interfaces, exact=False)

    binom = comb(total_interfaces, observed_interfaces, exact=False)
    if binom == float("inf") or stirling == float("inf"):
        return 1.0

    return (
        int(stirling * binom)
        * math.factorial(observed_interfaces)
        / total_interfaces**n_probes
    )


def estimate_total_interfaces(n_probes, observed_interfaces, likelihood_threshold=0.95):
    # Gives the optimal estimation for the number of interfaces
    # given the number of probes and the number of observed interfaces
    if n_probes < observed_interfaces:
        raise ValueError(
            f"{n_probes=} must be greater or equal to {observed_interfaces=}"
        )

    if n_probes == observed_interfaces:
        for total_interfaces in range(observed_interfaces, MAX_INTERFACES):
            prob = event_prob(n_probes, observed_interfaces, total_interfaces)
            if prob > likelihood_threshold:
                return total_interfaces
        else:
            return observed_interfaces

    prev_prob = 0
    for total_interfaces in range(observed_interfaces - 1, MAX_INTERFACES):
        prob = event_prob(n_probes, observed_interfaces, total_interfaces)
        if prob > likelihood_threshold:
            return total_interfaces
        if prob < prev_prob:
            return total_interfaces - 1
        prev_prob = prob
    else:
        return observed_interfaces


@cache
def reach_prob(total_interfaces: int, n_probes: int) -> float:
    # Hypothesis: there are total_interfaces interfaces in total, and sent n_probes.
    # what is the probability to reach all interfaces?
    return event_prob(n_probes, total_interfaces, total_interfaces)


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


if __name__ == "__main__":
    # Examples

    for total_interfaces in range(1, 20):
        print(f"{stopping_point(total_interfaces, 0.05)}")

    for n_probes in range(1, 20):
        print("---")
        for observed_interfaces in range(1, n_probes + 1):
            estimate = estimate_total_interfaces(
                n_probes, observed_interfaces, likelihood_threshold=0.95
            )
            print(f"n={n_probes} k={observed_interfaces} -> {estimate}")
