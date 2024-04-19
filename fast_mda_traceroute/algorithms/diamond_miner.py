from collections import defaultdict
from math import ceil
from typing import Dict, List, Set

from collections import Counter

from diamond_miner.generators import probe_generator
from diamond_miner.mappers import SequentialFlowMapper
from diamond_miner.typing import Probe
from more_itertools import flatten
from pycaracal import Reply

from fast_mda_traceroute.logger import logger
from fast_mda_traceroute.algorithms.utils.stopping_point import (
    optimal_N,
    stopping_point,
)
from fast_mda_traceroute.links import get_links_by_ttl
from fast_mda_traceroute.typing import Link
from fast_mda_traceroute.utils import is_ipv4


class DiamondMiner:
    """A standalone, in-memory, version of Diamond-Miner."""

    def __init__(
        self,
        dst_addr: str,
        min_ttl: int,
        max_ttl: int,
        src_port: int,
        dst_port: int,
        protocol: str,
        confidence: float,
        max_round: int,
    ):
        if protocol == "icmp" and not is_ipv4(dst_addr):
            protocol = "icmp6"
        self.failure_probability = 1.0 - (confidence / 100.0)
        self.dst_addr = dst_addr
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        # We only use a prefix_size of 1 for direct
        # point-to-point paths
        self.mapper_v4 = SequentialFlowMapper(prefix_size=1)
        self.mapper_v6 = SequentialFlowMapper(prefix_size=1)
        self.max_round = max_round
        # Diamond-Miner state
        self.current_round = 0
        self.probes_sent: Dict[int, int] = defaultdict(int)
        self.replies_by_round: Dict[int, List[Reply]] = {}
        self.n_unresolved = [0 for _ in range(self.max_ttl + 1)]

    @property
    def links_by_ttl(self) -> Dict[int, Set[Link]]:
        return get_links_by_ttl(tuple(self.time_exceeded_replies))

    @property
    def links(self) -> Set[Link]:
        return set(flatten(self.links_by_ttl.values()))

    @property
    def replies(self) -> List[Reply]:
        return tuple(flatten(self.replies_by_round.values()))

    @property
    def time_exceeded_replies(self) -> List[Reply]:
        return (x for x in self.replies if x.time_exceeded)

    @property
    def destination_unreachable_replies(self) -> List[Reply]:
        return [x for x in self.replies if x.destination_unreachable]

    @property
    def echo_replies(self) -> List[Reply]:
        return [x for x in self.replies if x.echo_reply]

    def nodes_distribution_at_ttl(self, nodes: List[str], ttl: int) -> Dict[str, float]:
        # a routine to fetch the number of replies from a given node at a given TTL
        # NOTE: a node may appear at multiple TTLs
        def node_replies(node, ttl):
            return len(
                [
                    r
                    for r in self.time_exceeded_replies
                    if r.reply_src_addr == node and r.probe_ttl == ttl
                ]
            )

        # total number of observations of links reaching nodes at the current ttl.
        # since links are stored with the 'near_ttl',
        # we need to fetch them at ttl-1
        # all_replies = len(self.links_by_ttl.get(ttl - 1, []))
        link_dist = {node: node_replies(node, ttl) for node in nodes}
        total = sum(link_dist.values())
        if total:
            link_dist = {k: v / total for k, v in link_dist.items()}

            # if all_replies:
            #     # compute the probability distribution of nodes at the current ttl
            #     link_dist = {node: node_replies(node, ttl) for node in nodes}
            #     logger.debug(
            #         "link_dist at ttl (abs) %d: %s / total: %d", ttl, link_dist, all_replies
            #     )
            #     link_dist = {node: node_replies(node, ttl) / all_replies for node in nodes}
            logger.debug("link_dist at ttl %d: %s", ttl, link_dist)
        else:
            # if we did not observe links at the previous ttl
            # we won't apply weights to the n_k afterwards
            logger.debug("No links at ttl %d", ttl)
            link_dist = {node: 1.0 / len(nodes) for node in nodes}

        return link_dist

    def unresolved_nodes_at_ttl(
        self, ttl: int, optimal_jump: bool = False
    ) -> tuple[list[str], int]:
        # returns the list of unresolved nodes at a given TTL
        # a node is said to be unresolved if not enough probes
        # have observed the outgoing links of this node.
        # This threshold is given by the stopping_point routine.
        # Resolved vertices correspond to nodes where all
        # outgoing load balanced links have been discovered
        # with high probability toward the destination.
        unresolved = []
        weighted_thresholds = []

        # for every discovered node at this TTL (non None)
        nodes_at_ttl = set(filter(bool, (x[1] for x in self.links_by_ttl[ttl])))
        if nodes_at_ttl:
            logger.debug("Nodes at TTL %d: %s", ttl, nodes_at_ttl)

        # fetch the distribution of the nodes at this TTL.
        # this is important to determine what percentage of probes
        # sent to this TTL will eventually reach each specific node.
        link_dist = self.nodes_distribution_at_ttl(nodes_at_ttl, ttl)

        for node in nodes_at_ttl:
            # number of unique nodes at the next TTL that share a link with the node
            n_successors = len(
                set(
                    [
                        x[2]
                        for x in self.links_by_ttl[ttl]
                        if x[1] == node and x[2] is not None
                    ]
                )
            )

            # the minimum number of probes to send to confirm we got all successors
            # We try to dismiss the hypothesis that there are more successors than we observed
            logger.debug("Detected %d successors for node %s", n_successors, node)
            n_k = stopping_point(n_successors, self.failure_probability)

            # number of outgoing probes that went through the node
            n_probes = len([x for x in self.links_by_ttl[ttl] if x[1] == node and x[2]])
            # other = len([x for x in self.links_by_ttl[ttl] if x[0] == node and x[1]])
            logger.debug(
                "Detected %d outgoing links (probes) for node %s at ttl %d",
                n_probes,
                node,
                ttl,
            )
            # n_probes = other

            logger.debug(
                "Expected %d probes for node %s at ttl %d and already sent %d",
                n_k,
                node,
                ttl,
                n_probes,
            )

            # if we have not sent enough probes for this node
            if n_probes < n_k:
                logger.debug("|> Node %s is therefore unresolved", node)
                # mark the node as unresolved
                unresolved.append(node)

                # we store the total number of probes to send to get confirmation:
                # it is the threshold n_k weighted by how difficult it is to reach
                # this node, i.e. the distribution of probes that reach this node
                if optimal_jump:
                    opti_N = optimal_N(n_probes, n_successors)
                    opti_n_k = stopping_point(opti_N + 1, self.failure_probability)
                    weighted_thresholds.append(opti_n_k / link_dist[node])
                else:
                    weighted_thresholds.append(n_k / link_dist[node])
                logger.debug(
                    "|> At ttl %d, the distribution of nodes is %s", ttl, link_dist
                )
                logger.debug("|> Its distribution is %f", link_dist[node])
                logger.debug(
                    "|> Node %s is unresolved, with a weighted threshold of %d",
                    node,
                    n_k / link_dist[node],
                )

        # we store the number of unresolved nodes at each TTL for logging purposes
        self.n_unresolved[ttl] = len(unresolved)
        if unresolved:
            logger.debug("Unresolved nodes at TTL %d: %s", ttl, unresolved)
            logger.debug(
                "|> Weighted thresholds at TTL %d: %s",
                ttl,
                ceil(max(weighted_thresholds, default=0)),
            )

        return unresolved, ceil(max(weighted_thresholds, default=0))

    def next_round(
        self, replies: List[Reply], optimal_jump: bool = False
    ) -> List[Probe]:
        self.current_round += 1

        self.replies_by_round[self.current_round] = replies
        logger.debug("######### Round %d: %d replies", self.current_round, len(replies))
        replies_by_ttl = defaultdict(list)
        for reply in replies:
            replies_by_ttl[reply.probe_ttl].append(reply)
        for ttl, ttl_replies in sorted(replies_by_ttl.items(), key=lambda x: x[0]):
            # log replies at each TTL
            # log content of replies at each TTL
            logger.debug(
                "Replies @ TTL %d from: %s",
                ttl,
                Counter(x.reply_src_addr for x in ttl_replies),
            )

        if self.current_round > self.max_round:
            return []

        max_flows_by_ttl = defaultdict(int)

        if self.current_round == 1:
            # NOTE: we cannot reliably infer the destination TTL because it may not be unique.

            # we could send only one probe per TTL, but that would not resolve any node.
            # max_flow = 1
            max_flow = stopping_point(1, self.failure_probability)
            max_flows_by_ttl = {
                ttl: max_flow for ttl in range(self.min_ttl, self.max_ttl + 1)
            }
        else:
            max_flows_by_ttl = {
                ttl: self.unresolved_nodes_at_ttl(ttl, optimal_jump)[1]
                for ttl in range(self.min_ttl, self.max_ttl + 1)
            }

        # See Proposition 1 in the original Diamond Miner paper.
        # The max flow for a TTL is the one computed for unresolved nodes at this TTL
        # or the one computed at the previous TTL to traverse the previous TTL:
        # we take the max of both values.

        def combined_max_flow(ttl):
            return min(
                max(max_flows_by_ttl[ttl], max_flows_by_ttl.get(ttl - 1, 0)), 2**14
            )

        flows_by_ttl = {
            ttl: range(self.probes_sent[ttl], combined_max_flow(ttl))
            for ttl in range(self.min_ttl, self.max_ttl + 1)
        }

        probes = []
        for ttl, flows in flows_by_ttl.items():
            probes_for_ttl = list(
                probe_generator(
                    [(self.dst_addr, self.protocol)],
                    flow_ids=flows,
                    ttls=[ttl],
                    prefix_len_v4=32,
                    prefix_len_v6=128,
                    probe_src_port=self.src_port,
                    probe_dst_port=self.dst_port,
                    mapper_v4=self.mapper_v4,
                    mapper_v6=self.mapper_v6,
                )
            )
            logger.debug(
                "Round %d: Sending %d probes to TTL %d",
                self.current_round,
                len(probes_for_ttl),
                ttl,
            )

            self.probes_sent[ttl] += len(probes_for_ttl)
            probes.extend(probes_for_ttl)

        # shuffle(probes)
        # from tests.fakenet.fakenet import graph_from_links

        # log the current topology discovered so far
        # links = {
        #     ttl: list((x, y) for _, x, y in self.links_by_ttl[ttl] if x and y)
        #     for ttl in range(1, self.max_ttl)
        # }
        # filter links to only show the ttl that have links
        # links = {k: v for k, v in links.items() if v}
        # if links:
        #     graph = graph_from_links("4.4.4.4", links)
        #     logger.debug(">>> Current topology:")
        #     for start, end in graph.edges:
        #         logger.debug(">   %s --> %s", start, end)
        # else:
        #     logger.debug("No links discovered so far")
        return probes
