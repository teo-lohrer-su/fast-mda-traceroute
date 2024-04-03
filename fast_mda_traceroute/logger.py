import sys
import logging

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)

logger = logging.getLogger("fast_mda_traceroute")
logger.setLevel(logging.INFO)
