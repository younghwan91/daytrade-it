"""Seeded shuffle of the backfill code order, recorded so a partial run is explicable.

The backfill walks codes in ascending order, so an interrupted run leaves the
alphabetically-early codes covered and the rest empty. KRX codes are not random
with respect to listing vintage, market or sector, so that gap would be a
systematic bias in every downstream analysis and nearly undetectable after the
fact. A fixed-seed shuffle makes any partial state an unbiased sample.

Writes the seed and the full order next to the data, not just to the console, so
an interrupted multi-day run can be explained without reconstruction.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts/eval")
from backfill_toss_company_news import universe_codes

SEED = 20260917
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/toss_universe_24m")
out_dir.mkdir(parents=True, exist_ok=True)

codes = universe_codes()
order = list(codes)
random.Random(SEED).shuffle(order)

(out_dir / "_order.json").write_text(
    json.dumps(
        {
            "seed": SEED,
            "n_codes": len(order),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "source": "universe_codes(): SELECT DISTINCT code FROM stocks WHERE code ~ '^[0-9]{6}$'",
            "note": "Processing order. A partial run covers a uniform random sample of the "
            "universe, not an alphabetical prefix.",
            "order": order,
        },
        indent=1,
    )
    + "\n"
)
Path("/tmp/codes_24m.txt").write_text(",".join(order))
print(f"seed {SEED}, {len(order)} codes, order written to {out_dir / '_order.json'}")
print("first 8:", order[:8])
print("ascending would have started:", sorted(codes)[:8])
