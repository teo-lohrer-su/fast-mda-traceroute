import glob
import os

ACCEPTANCE_THRESHOLD = 0.75
DEFAULT_CONFIDENCE = 95.0
N_TRIES = 100
SAMPLE_NET_DATA = "tests/fakenet/data"
SAMPLE_FILES = [
    f for f in glob.glob(os.path.join(SAMPLE_NET_DATA, "*")) if os.path.isfile(f)
]
SAMPLE_FILES = [
    "tests/fakenet/data/puri.mimuw.edu.pl_103.37.81.246_2018-02-08 12_38_30.754221+01_00"
]
