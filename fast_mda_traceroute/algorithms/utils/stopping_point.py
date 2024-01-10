from functools import cache

@cache
def reach_prob(total_interfaces: int, n_probes: int, target_interfaces: int) -> float:
    """
    Given a router with a known total number of interfaces, and a number of sent probes,
    computes the probability to have reached exactly a given target number of interfaces.
    """

    # Initialization:
    #   if we sent no probes, we can reach 0 interfaces with probability 1.0
    #                         we can reach n > 0 interfaces with probability 0.0
    if n_probes == 0:
        return 1.0 if target_interfaces == 0 else 0.0

    # We now define our recursion primarily on the number of sent probes.
    # We suppose we already sent (n_probes - 1) probes and distinguish between
    # the two following cases:
    #
    # - A: we already reached our target number of interfaces with (n_probes - 1) probes ; or
    # - B: we still need to reach one new interface with our last probe.
    #
    # A :  [x] [x] [x] [ ] ------> [x] [x] [x] [ ]
    #     (after n-1 probes)      (after n probes)
    #
    # B :  [x] [x] [ ] [ ] ------> [x] [x] [X] [ ]
    #     (after n-1 probes)      (after n probes)
    # 
    prob_discovered_all_targets = reach_prob(total_interfaces, n_probes - 1, target_interfaces)
    prob_hit_discovered_interface = (target_interfaces / total_interfaces)
    
    prob_one_target_left = reach_prob(total_interfaces, n_probes - 1, target_interfaces - 1)
    prob_hit_new_interface = (total_interfaces - (target_interfaces - 1)) / total_interfaces

    return prob_discovered_all_targets * prob_hit_discovered_interface \
        + prob_one_target_left * prob_hit_new_interface


@cache
def stopping_point(n_interfaces: int, failure_probability: float) -> int:
    n_probes = 0

    while reach_prob(n_interfaces, n_probes, n_interfaces) < (1 - failure_probability):
        n_probes += 1
    
    return n_probes

print(",".join(str(stopping_point(i, 0.05)) for i in range(64)))