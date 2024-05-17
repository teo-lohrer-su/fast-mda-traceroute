import glob
import os

DELTA_THRESHOLD = 0.95
DEFAULT_CONFIDENCE = 95.0
N_TRIES = 100
SAMPLE_NET_DATA = "tests/fakenet/data"
SAMPLE_FILES = [
    f for f in glob.glob(os.path.join(SAMPLE_NET_DATA, "*")) if os.path.isfile(f)
]
