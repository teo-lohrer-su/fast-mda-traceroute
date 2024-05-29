from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set

from more_itertools import map_reduce
from pycaracal import Reply

from fast_mda_traceroute.typing import Flow, Link, Pair


@cache
def get_replies_by_flow(replies: List[Reply]) -> Dict[Flow, List[Reply]]:
    return map_reduce(
        replies,
        lambda x: (
            x.probe_protocol,
            x.probe_dst_addr,
            x.probe_src_port,
            x.probe_dst_port,
        ),
    )


@cache
def get_replies_by_ttl(replies: List[Reply]) -> Dict[int, List[Reply]]:
    return map_reduce(replies, lambda x: x.probe_ttl)  # type: ignore


def get_successors(replies: List[Reply], src: str) -> Set[str]:
    successors = set()
    # fetch the addresses that are one hop away from the source and that have been reached
    # with a flow id that also reached the source
    ttls = [x.probe_ttl for x in replies if x.reply_src_addr == src]
    for flow, replies in get_replies_by_flow(replies).items():
        if any(x.reply_src_addr == src for x in replies):
            for reply in replies:
                if reply.probe_ttl - 1 in ttls:
                    successors.add(reply.reply_src_addr)

    return successors


def get_flow_ids(replies: List[Reply], src: str) -> Set[Flow]:
    flow_ids = set()
    for flow, replies in get_replies_by_flow(replies).items():
        if any(x.reply_src_addr == src for x in replies):
            flow_ids.add(flow)
    return flow_ids


@cache
def get_pairs_by_flow(replies: List[Reply]) -> Dict[Flow, List[Pair]]:
    pairs_by_flow = defaultdict(list)
    replies_by_flow = get_replies_by_flow(replies)

    for flow, replies in replies_by_flow.items():
        replies_by_ttl = get_replies_by_ttl(tuple(replies))
        for near_ttl in range(min(replies_by_ttl) - 1, max(replies_by_ttl)):
            near_replies = replies_by_ttl.get(near_ttl, [None])
            far_replies = replies_by_ttl.get(near_ttl + 1, [None])
            for near_reply in near_replies:
                for far_reply in far_replies:
                    pairs_by_flow[flow].append((near_ttl, near_reply, far_reply))
    return pairs_by_flow


def get_links_by_flow_by_ttl(replies: List[Reply]) -> Dict[int, Dict[Flow, Set[Link]]]:
    links_by_ttl = defaultdict(dict)
    pairs_by_flow = get_pairs_by_flow(replies)
    for flow, pairs in pairs_by_flow.items():
        for near_ttl, near_reply, far_reply in pairs:
            if near_reply:
                links_ttl = links_by_ttl[near_ttl]
                if near_reply.reply_src_addr not in links_ttl:
                    links_ttl[near_reply.reply_src_addr] = set()
                links_by_ttl[near_ttl][near_reply.reply_src_addr].add(
                    (
                        near_ttl,
                        near_reply.reply_src_addr if near_reply else None,
                        far_reply.reply_src_addr if far_reply else None,
                    )
                )
    return links_by_ttl


@cache
def get_links_by_ttl(replies: List[Reply]) -> Dict[int, Set[Link]]:
    # links_by_ttl = defaultdict(set)
    links_by_ttl = defaultdict(list)
    pairs_by_flow = get_pairs_by_flow(tuple(replies))

    for flow, pairs in pairs_by_flow.items():
        for near_ttl, near_reply, far_reply in pairs:
            links_by_ttl[near_ttl].append(
                (
                    near_ttl,
                    near_reply.reply_src_addr if near_reply else None,
                    far_reply.reply_src_addr if far_reply else None,
                )
            )
    return links_by_ttl


def get_scamper_links(
    replies: List[Reply],
) -> Dict[str, Dict[int, Dict[Optional[str], List[Reply]]]]:
    """Data structure used in Scamper's JSON format."""
    links: Dict[str, Dict[int, Dict[Optional[str], List[Reply]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    replies_by_flow = get_replies_by_flow(tuple(replies))
    for flow, flow_replies in replies_by_flow.items():
        replies_by_ttl = get_replies_by_ttl(tuple(flow_replies))
        for near_ttl in range(min(replies_by_ttl), max(replies_by_ttl)):
            # TODO: Handle per-packet load-balancing.
            far_ttl = near_ttl + 1
            near_reply = replies_by_ttl.get(near_ttl, [None])[0]
            far_reply = replies_by_ttl.get(far_ttl, [None])[0]
            if not near_reply:
                continue
            while not far_reply and far_ttl <= max(replies_by_ttl):
                links[near_reply.reply_src_addr][far_ttl][None].append(None)
                far_ttl += 1
                far_reply = replies_by_ttl.get(far_ttl, [None])[0]
            if far_reply:
                links[near_reply.reply_src_addr][far_ttl][
                    far_reply.reply_src_addr
                ].append(far_reply)
    return links
