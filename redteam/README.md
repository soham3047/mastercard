# VINEYARD — Red Team / Attack Generation (Person A)

Generates synthetic fraud transaction graphs — directed cyclic rings
(layering), hub-and-spoke fan-outs (mule networks), and Hawkes-clustered
decoy noise — with GenAI-produced KYC identities, for the Blue Team
detector (Person B), the adversarial loop (Person C), and the dashboard
(Person D) to consume.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional, for real GenAI identity generation (Stage A5):

```bash
export ANTHROPIC_API_KEY=sk-...
```

Without a key set, identity generation automatically falls back to an
offline templated generator — the rest of the pipeline (topology,
timing, amounts) is unaffected either way, so the demo still works with
no network access (e.g. flaky venue wifi during judging).

## Run

```bash
# Full run: rings + fan-outs + decoys + LLM identities
python -m src.main --out output/run0

# Fast smoke test, no LLM calls
python -m src.main --out output/smoketest --skip-llm

# Reproducible run with a specific seed
python -m src.main --out output/run0 --seed 7

# Override any theta knob via a JSON file (used by Person C's loop)
python -m src.main --out output/round3 --theta path/to/theta_overrides.json
```

Output written to `<out>/`:
- `transaction_graph.json` — the interface-contract nodes/edges schema
  (consumed by B, C, D exactly as specified).
- `identities.json` — extra KYC-style content keyed by account id (used
  by D's dashboard "explain this account" panel — not part of the
  strict graph schema).
- `theta_used.json` — the exact config this run used (for reproducibility
  and for C to log alongside detection results).

## Sanity checks

```bash
python -m tests.test_smoke      # schema validity, ring closure, threshold checks, Hawkes clustering
python -m src.calibration       # print the calibration derivation + sanity numbers
```

## Stages (see `PROGRESS_A.md` for full synopses)

| Stage | What | File(s) |
|---|---|---|
| A0 | Scaffold + stub writer | `src/schema.py`, `src/writer.py` |
| A1 | Ring generator | `src/ring_generator.py` |
| A2 | Fan-out/star generator | `src/fanout_generator.py` |
| A3 | Decoy noise + Hawkes realism | `src/decoy_generator.py` |
| A4 | Calibration to public stats | `src/calibration.py` |
| A5 | LLM identity + narrative generation | `src/identity_llm.py` |
| A6 | Parameterization for the adversarial loop | `src/theta.py` |

## Handoff to Person C (adversarial loop)

Call `src.main.build_graph(theta)` directly (don't shell out to the CLI)
for speed across many rounds:

```python
from src.theta import Theta
from src.main import build_graph
from src.identity_llm import generate_identities

theta = Theta(ring_size_lambda=8, decoy_density=0.9, fan_out_ratio=0.6)  # mutate any field
graph = build_graph(theta)  # -> TransactionGraph; .to_dict() for the raw JSON shape
```

`theta.as_dict()` is the flat config object referenced in the shared
interface contract as "a flat config dict θ ... expect roughly:
`ring_size`, `decoy_density`, `hop_timing_params`, `fan_out_ratio`" — see
`src/theta.py` for the exact key names actually used (they differ
slightly from that rough sketch; this file is the source of truth).
