"""Stage 3: exit timing. Not built yet -- owned by another agent.

The runner calls this with `--in` (the previous stage's out-of-fold file) and
`--out`. It exits non-zero so a pipeline run cannot silently report success
past a stage that does not exist.
"""

import argparse
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.parse_args()
sys.exit("stage 3 (exit timing) is not implemented yet")
