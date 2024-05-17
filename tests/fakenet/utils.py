import networkx as nx
from pycaracal import Probe, Reply
from fast_mda_traceroute.algorithms.diamond_miner import DiamondMiner
from fast_mda_traceroute.typing import Protocol
from fakenet.fakenet import FakeNet, FakeProber


def eval_diamond_miner(
    net: FakeNet, confidence: float = 95.0, seed: int = 0, optimal_jump=False
) -> bool:
    prober = FakeProber(net, seed=seed)

    protocol = Protocol.ICMP
    src_port = 12345
    dst_port = 33434
    max_round = 100
    min_ttl = 1
    # fetch the distance between the start and end nodes
    # NOTE: we cannot use the shortest path because the network may be unbalanced
    max_ttl = nx.dag_longest_path_length(net) + 1

    src_addr = net.start
    dst_addr = net.end

    alg = DiamondMiner(
        dst_addr=dst_addr,
        min_ttl=min_ttl,
        max_ttl=max_ttl,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol.value,
        confidence=confidence,
        max_round=max_round,
    )

    last_replies: list[Reply] = []
    rnd = 0
    while True:
        rnd += 1
        probes = [
            Probe(*x) for x in alg.next_round(last_replies, optimal_jump=optimal_jump)
        ]
        if not probes:
            break
        last_replies = prober.probe(probes)
    # transform links_by_ttl to a dict of lists
    links = {
        ttl: list((x, y) for _, x, y in alg.links_by_ttl[ttl] if x and y)
        for ttl in range(1, max_ttl)
    }
    # filter links to only show the ttl that have links
    links = {k: v for k, v in links.items() if v}

    # create a graph from the links
    nx_graph = graph_from_links(src_addr, links)

    # check if nx_graph is equal to net
    res = nx.is_isomorphic(nx_graph, net)
    return res


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
