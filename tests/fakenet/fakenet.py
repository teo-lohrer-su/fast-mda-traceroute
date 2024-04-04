from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import xxhash
import random

import networkx as nx
from ipaddress import ip_address
from pycaracal import Probe, Reply

# A graph that represents a network topology
# where nodes are routers and edges existing links.
# Nodes can have multiple outgoing edges, representing multiple interfaces.
# Edges are not weighted.
# The topology is loaded from a text file with two IPs per line, separated by a space.


class FakeNet(nx.DiGraph):
    """
    A fake network topology that emulates a network graph.
    The graph is a directed graph with a start and end node.
    The class includes various methods to generate fake network topologies.
    One can generate:
    - a meshed topology with a given topology list
    - a random graph with a given edge probability and depth
    - a network with multiple single paths
    - a network from a file
    """

    DEFAULT_START = "4.4.4.4"
    DEFAULT_END = "44.44.44.44"

    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None

    @classmethod
    def meshed(cls, topology: list[int]) -> FakeNet:
        # each value in the topology list represents the number of nodes at that ttl
        # the first value is the number of nodes at ttl 1
        # the second value is the number of nodes at ttl 2, etc.
        net = FakeNet()
        routers = [
            [f"192.{ttl+1}.{i}.0" for i in range(n)] for ttl, n in enumerate(topology)
        ]
        net.start = FakeNet.DEFAULT_START
        net.end = FakeNet.DEFAULT_END
        # use itertools.product to create all possible paths and add them to the graph
        for ttl in range(len(topology) - 1):
            for i in range(topology[ttl]):
                for j in range(topology[ttl + 1]):
                    net.add_edge(f"192.{ttl+1}.{i}.0", f"192.{ttl+2}.{j}.0")
        # for path in itertools.product(*routers):
        #     nx.add_path(net, path)
        for dest in routers[0]:
            net.add_edge(net.start, dest)
        for dest in routers[-1]:
            net.add_edge(dest, net.end)

        return net

    @classmethod
    def multi_single_paths(cls, n_paths: int, path_length: int) -> FakeNet:
        # create a network with n_paths paths of length path_length
        net = FakeNet()
        net.start = FakeNet.DEFAULT_START
        net.end = FakeNet.DEFAULT_END
        for i in range(n_paths):
            net.add_edge(net.start, f"192.{i}.0.0")
            for j in range(0, path_length):
                start = f"192.{i}.{j}.0"
                dest = f"192.{i}.{j+1}.0"
                net.add_edge(start, dest)
            net.add_edge(dest, net.end)
        return net

    @classmethod
    def from_file(cls, filepath: str) -> FakeNet:
        """
        Create a network topology from a file.
        The file should contain two IP addresses per line, separated by a space.
        Comments are allowed and start with a #.
        The graph is a directed graph with a start and end node.
        The start node is the first node in a topological sort; the end node is the last node.
        """
        net = FakeNet()
        # open the file and add edges
        with open(filepath) as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                # we skip empty lines and comments
                if line and not line.startswith("#"):
                    try:
                        (start, end) = line.split(" ", maxsplit=1)
                        # We still try to validate the IP addresses
                        _, _ = ip_address(start), ip_address(end)
                        net.add_edge(start, end)
                    except ValueError as e:
                        raise ValueError(f"Error on line {i+1} in file {filepath}: {e}")
        # set start to the single node in the graph with no incoming edges
        # i.e. the first node in a topological sort
        net.start = next(iter(nx.topological_sort(net)))
        # set end to the single node in the graph with no outgoing edges
        # i.e. the last node in a topological sort
        net.end = list(nx.topological_sort(net))[-1]
        return net

    @classmethod
    def random_graph(cls, p_edge: float, depth: int, seed: int = 0) -> FakeNet:
        """
        Generate a random graph with a given edge probability and depth (number of ttl levels)
        The graph is a directed graph with a start and end node
        The start node is connected to all nodes at ttl 1
        The end node is connected to all nodes at last ttl
        :param p_edge: probability of an edge existing between two nodes
        :param depth: number of ttl levels
        :return: a FakeNet instance
        """
        # we seed the random number generator to get reproducible results
        random.seed(seed)

        # TODO: parametrize the topology generation
        # generate a random topology with a random number of nodes at each ttl
        # the first value is the number of nodes at ttl 1, the second value at ttl 2, etc.
        # the topology is generated by picking a random number of nodes between 1 and 5,
        # with a bias toward 1 (2/3 of values will be 1)

        topology = [max(random.randint(-4, 5), 1) for _ in range(depth)]

        # generate a list of routers IPs for each ttl
        # routers = [ [list of routers at ttl 1], [list of routers at ttl 2], ...]
        routers = [
            [f"192.{ttl+1}.{i}.0" for i in range(n)] for ttl, n in enumerate(topology)
        ]

        random_graph = nx.DiGraph()

        for ttl in range(len(topology) - 1):
            for i, src in enumerate(routers[ttl]):
                for j, dst in enumerate(routers[ttl + 1]):
                    # we always add an edge between the first nodes at each ttl
                    # this is to ensure that the end node is reachable from the start node
                    if i == 0 and j == 0:
                        random_graph.add_edge(src, dst)
                    elif random.random() < p_edge:
                        random_graph.add_edge(src, dst)
        start = FakeNet.DEFAULT_START
        end = FakeNet.DEFAULT_END
        for dst in routers[0]:
            random_graph.add_edge(start, dst)
        for src in routers[-1]:
            random_graph.add_edge(src, end)

        # filter edges such that all nodes are reachable from the start node
        # and the end node is reachable from all nodes
        random_graph = nx.DiGraph(
            [
                (u, v)
                for (u, v) in random_graph.edges()
                if nx.has_path(random_graph, start, u)
                and nx.has_path(random_graph, v, end)
            ]
        )
        # TODO: add logging
        # print(
        #     f"Topology: {[1]+[len([r for r in rr if r in random_graph]) for rr in routers]+[1]}"
        # )
        # # print number of paths from start to end
        # print(
        #     f"Paths from {start} to {end}: {len(list(nx.all_simple_paths(random_graph, start, end)))}"
        # )

        net = FakeNet()
        net.start = start
        net.end = end
        net.add_edges_from(random_graph.edges())

        return net


class FakeProber:
    """
    A fake prober that generates fake replies for a given list of probes.
    It aims to emulate the prober part (caracal) of the fast_mda_traceroute tool.
    It is initialized with a network topology and a seed.
    The `probe` method simulates the network traversal of the probes and returns a list of replies.
    The seed parameter is used to generate random paths for the probes, based on the probe's attributes.
    """

    def __init__(self, net: FakeNet, seed: int = 0):
        """
        Initialize the prober with a network topology and a seed.
        :param net: a FakeNet instance
        :param seed: an integer seed
        """
        self.net = net
        self.seed = seed

    @classmethod
    def flow_id(cls, probe: Probe, local_seed: int) -> int:
        """
        Generate a unique identifier for a flow based on the probe attributes and a local seed.
        It uses the following attributes of the probe:
        - dst_addr
        - src_port
        - protocol
        - dst_port
        The xxhash algorithm is used to generate the hash.

        :param probe: a Probe instance
        :param local_seed: an integer seed
        """
        # select the probe attributes that uniquely identify a flow
        attributes = (
            probe.dst_addr,
            probe.src_port,
            probe.protocol,
            probe.dst_port,
        )
        # use a fast hash
        return xxhash.xxh32(str(attributes).encode(), seed=local_seed).intdigest()

    def node_seed(self, node_id: str) -> int:
        """
        Generate a unique seed for a node based on its id.
        Nodes are routers in the network topology.
        The hash is computed using the xxhash algorithm.
        The prober's seed is used as a seed for the hash.
        :param node_id: a string representing the node
        """
        # use a fast hash
        return xxhash.xxh32(node_id.encode(), seed=self.seed).intdigest()

    def probe(self, probes: list[Probe]) -> list[Reply]:
        """
        Simulate the network traversal of the probes and generate fake replies.
        Replies can be ICMP time exceeded or ICMP echo replies.
        :param probes: a list of Probe instances

        :return: a list of Reply instances
        """
        replies = []
        for probe in probes:
            # walks the graph from the start node to the probe's destination
            # monitors the ttl and returns a reply when the destination is reached or the ttl is exceeded
            # a random path is chosen using the id of the flow modulo the number of successors of the current node
            dst = probe.dst_addr
            ttl = probe.ttl
            current = self.net.start
            while ttl > 0 and current != dst:
                succ = list(self.net.successors(current))

                # fetch a unique identifier for the current node
                local_seed = self.node_seed(current)
                # from the probe attributes and the current node's hash, get a unique identifier for the flow
                flow_id = FakeProber.flow_id(probe, local_seed=local_seed)

                # select the successor based on the flow id
                current = succ[flow_id % len(succ)]
                ttl -= 1
            if ttl == 0:
                replies.append(FakeReply.icmp_time_exceeded_reply(current, probe))
            else:
                replies.append(FakeReply.icmp_echo_reply(current, probe))

        return replies


@dataclass
class FakeReply:
    """
    A fake reply that emulates the replies generated by the prober.
    The class provides two class methods to generate ICMP time exceeded and ICMP echo replies.
    """

    # a fake reply that has the following:
    probe_protocol: str
    probe_dst_addr: str
    probe_src_port: int
    probe_dst_port: int
    probe_ttl: int
    reply_ttl: int
    reply_src_addr: str
    reply_icmp_type: int
    reply_icmp_code: int
    reply_id: int
    quoted_ttl: int
    capture_timestamp: int
    rtt: int

    @property
    def time_exceeded(self):
        # This method is used by the DiamondMiner class to filter time exceeded replies
        return self.reply_icmp_type == 11

    @property
    def echo_reply(self):
        # This method is used by the DiamondMiner class to filter echo replies
        return self.reply_icmp_type == 0

    @classmethod
    def icmp_time_exceeded_reply(cls, src: int, probe: Probe) -> FakeReply:
        """
        Generate a fake ICMP time exceeded reply for a given probe.
        The probe's TTL is normally embedded in the IP ID field.
        Here, we directly use the probe's TTL.
        :param src: the source address of the reply
        :param probe: a Probe instance

        :return: a FakeReply instance
        """

        # return a fake ICMP time exceeded reply using information from the probe
        return FakeReply(
            probe_protocol=probe.protocol,
            probe_dst_addr=probe.dst_addr,
            probe_src_port=probe.src_port,
            probe_dst_port=probe.dst_port,
            probe_ttl=probe.ttl,
            reply_ttl=255 - probe.ttl,
            reply_src_addr=src,
            reply_icmp_type=11,
            reply_icmp_code=0,
            reply_id=FakeProber.flow_id(probe, 0) % 2**16,
            quoted_ttl=1,
            capture_timestamp=0,
            rtt=0,
        )

    @classmethod
    def icmp_echo_reply(cls, src: int, probe: Probe) -> FakeReply:
        """
        Generate a fake ICMP echo reply for a given probe.
        Same as the ICMP time exceeded reply, but with a different ICMP type.
        :param src: the source address of the reply
        :param probe: a Probe instance

        :return: a FakeReply instance
        """
        # return a fake ICMP time ECHO reply using information from the probe
        # re-uses the icmp_time_exceeded_reply method
        reply = FakeReply.icmp_time_exceeded_reply(src, probe)
        reply.reply_icmp_type = 0
        return reply


def graph_from_links(
    src_addr: str, links: dict[int, list[tuple[str, str]]]
) -> nx.DiGraph:
    """
    Create a networkx graph from a dictionary of links.
    The dictionary has the following structure:
    {
        ttl: [(src, dst), ...],
        ...
    }
    Since the links dictionnary starts at TTL 1,
    we need to connect the source address to all nodes at TTL 1.
    :param src_addr: the source address of the network
    :param links: a dictionary of links by TTL

    :return: a networkx DiGraph instance
    """
    nx_graph = nx.DiGraph()
    # link source address to all nodes at ttl 1
    for link in links[1]:
        nx_graph.add_edge(src_addr, link[0])

    # add all edges to the graph
    for _, links in links.items():
        nx_graph.add_edges_from(list(set(links)))

    return nx_graph
